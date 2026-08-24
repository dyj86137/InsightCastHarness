"""Bilibili backend routing for discovery and transcript inputs.

The routing model follows Agent-Reach's channel design without importing its
CLI package: prefer the structured ``bili`` command, then fall back to the
public Bilibili APIs.  The API fallback is intentionally read-only and does
not pretend to provide audio extraction.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from insightcast.platform_cookies import resolve_cookie_header


BILIBILI_SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
BILIBILI_VIEW_URL = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_PLAYER_URL = "https://api.bilibili.com/x/player/v2"
DEFAULT_BILIBILI_BACKENDS = ("bili-cli", "bilibili-api")
SUPPORTED_BILIBILI_BACKENDS = frozenset(DEFAULT_BILIBILI_BACKENDS)
_BILIBILI_ID_PATTERN = re.compile(r"(?:/video/|^)(BV[0-9A-Za-z]+|av[0-9]+)", re.IGNORECASE)
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}


class BilibiliRoutingError(RuntimeError):
    """所有已配置的 Bilibili 后端都不可用或执行失败。"""


@dataclass(frozen=True)
class BilibiliSearchRoute:
    """一次搜索使用的后端和规范化前的结果项。"""

    backend: str
    items: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class BilibiliVideoRoute:
    """一次视频详情请求使用的后端和 payload。"""

    backend: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class BilibiliAudioRoute:
    """一次音频下载使用的后端和输出文件。"""

    backend: str
    audio_path: Path


JsonFetcher = Callable[[str, Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]]
CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess]


class BilibiliBackendRouter:
    """按能力选择 Bilibili 后端，并保留实际使用的 backend。"""

    def __init__(
        self,
        *,
        env: Optional[Mapping[str, str]] = None,
        cookie_header: Optional[str] = None,
        fetch_json: Optional[JsonFetcher] = None,
        command_runner: Optional[CommandRunner] = None,
    ) -> None:
        self.env = dict(os.environ if env is None else env)
        self.cookie_header = cookie_header or resolve_cookie_header(
            self.env,
            domains=("bilibili.com",),
        )
        self.fetch_json = fetch_json or self._fetch_json
        self.command_runner = command_runner or _run_command
        self.command = self.env.get("INSIGHTCAST_BILIBILI_CLI", "bili").strip() or "bili"
        self.backends = self._ordered_backends()

    def _ordered_backends(self) -> Tuple[str, ...]:
        configured = self.env.get("INSIGHTCAST_BILIBILI_BACKENDS")
        if configured:
            values = tuple(
                _normalise_backend(value)
                for value in configured.split(",")
                if _normalise_backend(value)
            )
        else:
            values = DEFAULT_BILIBILI_BACKENDS

        values = tuple(value for value in values if value in SUPPORTED_BILIBILI_BACKENDS)
        if not values:
            values = DEFAULT_BILIBILI_BACKENDS

        preferred = _normalise_backend(self.env.get("INSIGHTCAST_BILIBILI_BACKEND", ""))
        if preferred in values:
            values = (preferred,) + tuple(value for value in values if value != preferred)
        return tuple(dict.fromkeys(values))

    def headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        }
        if self.cookie_header:
            headers["Cookie"] = self.cookie_header
        return headers

    def search(
        self,
        keyword: str,
        *,
        max_results: int,
        order: str = "totalrank",
        timeout_seconds: Optional[float] = None,
    ) -> BilibiliSearchRoute:
        errors = []
        for backend in self.backends:
            try:
                if backend == "bili-cli":
                    if not self._cli_available():
                        continue
                    payload = self._cli_json(
                        [
                            "search",
                            keyword,
                            "--type",
                            "video",
                            "--max",
                            str(max_results),
                            "--json",
                        ],
                        timeout_seconds=timeout_seconds,
                    )
                    return BilibiliSearchRoute(
                        backend=backend,
                        items=tuple(_normalise_search_item(item) for item in _search_items(payload)),
                    )

                if backend == "bilibili-api":
                    payload = self.fetch_json(
                        BILIBILI_SEARCH_URL,
                        {
                            "search_type": "video",
                            "keyword": keyword,
                            "page": 1,
                            "order": order,
                        },
                        self.headers(),
                    )
                    _ensure_api_success(payload, "Bilibili search")
                    data = payload.get("data") or {}
                    items = data.get("result") if isinstance(data, Mapping) else None
                    if not isinstance(items, list):
                        items = []
                    return BilibiliSearchRoute(
                        backend=backend,
                        items=tuple(items[:max_results]),
                    )
            except Exception as exc:  # noqa: BLE001 - continue to the next backend
                errors.append(f"{backend}: {exc}")

        raise BilibiliRoutingError(_routing_message("search", errors))

    def video(
        self,
        bvid: str,
        *,
        page_number: int = 1,
        timeout_seconds: Optional[float] = None,
    ) -> BilibiliVideoRoute:
        errors = []
        for backend in self.backends:
            try:
                if backend == "bili-cli":
                    if not self._cli_available():
                        continue
                    payload = self._cli_json(
                        ["video", bvid, "--subtitle", "--json"],
                        timeout_seconds=timeout_seconds,
                    )
                    if _has_usable_subtitle(payload) or "bilibili-api" not in self.backends:
                        return BilibiliVideoRoute(backend=backend, payload=payload)
                    errors.append(f"{backend}: no usable subtitle in response")
                    continue

                if backend == "bilibili-api":
                    payload = self._api_video_payload(bvid, page_number=page_number)
                    return BilibiliVideoRoute(backend=backend, payload=payload)
            except Exception as exc:  # noqa: BLE001 - continue to the next backend
                errors.append(f"{backend}: {exc}")

        raise BilibiliRoutingError(_routing_message("video", errors))

    def download_audio(
        self,
        bvid: str,
        *,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[BilibiliAudioRoute]:
        """下载 B 站音频。

        The public API backend deliberately does not claim audio support. If
        ``bili-cli`` is unavailable, callers receive ``None`` and can record a
        metadata-only transcript instead of silently falling back to yt-dlp.
        """

        errors = []
        for backend in self.backends:
            if backend != "bili-cli":
                continue
            if not self._cli_available():
                continue

            candidate_dir = output_dir / _safe_component(bvid)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = self.command_runner(
                    (
                        self.command,
                        "audio",
                        bvid,
                        "--no-split",
                        "-o",
                        str(candidate_dir),
                    ),
                    timeout_seconds or 1800.0,
                )
            except Exception as exc:  # noqa: BLE001 - route failure is reported below
                errors.append(f"{backend}: {exc}")
                continue

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "command failed").strip()
                errors.append(f"{backend}: {detail[:400]}")
                continue

            files = sorted(
                (
                    path
                    for path in candidate_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if files:
                return BilibiliAudioRoute(backend=backend, audio_path=files[0])
            errors.append(f"{backend}: command succeeded but produced no audio file")

        if errors:
            raise BilibiliRoutingError(_routing_message("audio", errors))
        return None

    def _cli_available(self) -> bool:
        return bool(shutil.which(self.command))

    def _cli_json(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Mapping[str, Any]:
        result = self.command_runner(
            tuple((self.command, *args)),
            timeout_seconds or 60.0,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "command failed").strip()
            raise BilibiliRoutingError(detail[:400])
        payload = _parse_json_output(result.stdout or "")
        if payload.get("ok") is False:
            error = payload.get("error") or {}
            if isinstance(error, Mapping):
                raise BilibiliRoutingError(
                    str(error.get("message") or error.get("code") or "bili-cli returned an error")
                )
            raise BilibiliRoutingError("bili-cli returned an error")
        return payload

    def _api_video_payload(self, bvid: str, *, page_number: int) -> Mapping[str, Any]:
        view = self.fetch_json(
            BILIBILI_VIEW_URL,
            {"bvid": bvid},
            self.headers(),
        )
        _ensure_api_success(view, "Bilibili video detail")
        video = view.get("data") or {}
        if not isinstance(video, Mapping):
            raise BilibiliRoutingError("Bilibili video detail data is not an object")

        pages = video.get("pages")
        page = pages[0] if isinstance(pages, list) and pages else {}
        if isinstance(pages, list):
            for item in pages:
                if isinstance(item, Mapping) and _page_value(item.get("page")) == page_number:
                    page = item
                    break
        cid = page.get("cid") if isinstance(page, Mapping) else None
        player: Mapping[str, Any] = {}
        if cid:
            player = self.fetch_json(
                BILIBILI_PLAYER_URL,
                {"bvid": bvid, "cid": cid},
                self.headers(),
            )
        return {"video": video, "player": player}

    def _fetch_json(
        self,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        request_url = f"{url}?{query}" if query else url
        request = Request(request_url, headers=dict(headers))
        timeout = float(self.env.get("INSIGHTCAST_BILIBILI_FETCH_TIMEOUT_SECONDS", "10"))
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise BilibiliRoutingError("Bilibili API response is not an object")
        return payload


def extract_bilibili_id(value: Any) -> Optional[str]:
    """从 CandidateItem、BVID、BV URL 或 av URL 中提取平台 ID。"""

    platform_item_id = getattr(value, "platform_item_id", None)
    url = getattr(value, "effective_url", None) or getattr(value, "url", None)
    for raw in (platform_item_id, url, value):
        text = str(raw or "").strip()
        if not text:
            continue
        match = _BILIBILI_ID_PATTERN.search(text)
        if match:
            return match.group(1)
    return None


def _run_command(
    command: Sequence[str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _parse_json_output(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise BilibiliRoutingError("bili-cli output is not valid JSON") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise BilibiliRoutingError("bili-cli output is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise BilibiliRoutingError("bili-cli JSON root is not an object")
    return payload


def _search_items(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        return ()
    for key in ("items", "videos", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _has_usable_subtitle(payload: Mapping[str, Any]) -> bool:
    """判断 CLI 视频 payload 是否包含正文或字幕下载地址。"""

    def has_value(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(has_value(item) for item in value)
        if isinstance(value, Mapping):
            return any(
                has_value(value.get(key))
                for key in (
                    "text",
                    "content",
                    "transcript",
                    "subtitle_url",
                    "url",
                    "body",
                    "subtitles",
                    "items",
                    "segments",
                )
                if key in value
            )
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in {"subtitle", "subtitles"} and has_value(item):
                    return True
                if str(key).lower() in {"data", "player", "video"} and walk(item):
                    return True
        elif isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(payload)


def _normalise_search_item(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """把 bili-cli 的规范化结果映射到现有 Bilibili discovery 字段。"""

    data = dict(item)
    bvid = str(
        item.get("bvid")
        or item.get("bvid_id")
        or item.get("video_id")
        or item.get("id")
        or ""
    ).strip()
    if bvid.lower().startswith("av") and not bvid[2:].isdigit():
        bvid = ""
    title = str(item.get("title") or item.get("name") or "").strip()
    description = str(
        item.get("description") or item.get("desc") or item.get("summary") or ""
    ).strip()
    url = str(item.get("url") or item.get("arcurl") or "").strip()
    if not url and bvid:
        url = f"https://www.bilibili.com/video/{bvid}"
    data.update(
        {
            "bvid": bvid,
            "title": title,
            "description": description,
            "arcurl": url,
            "pubdate": item.get("published_at") or item.get("pubdate"),
            "duration": item.get("duration") or item.get("duration_seconds"),
        }
    )
    return data


def _ensure_api_success(payload: Mapping[str, Any], operation: str) -> None:
    if payload.get("code") not in (0, "0"):
        raise BilibiliRoutingError(
            f"{operation} failed: "
            f"{payload.get('message') or payload.get('msg') or payload.get('code')}"
        )


def _page_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalise_backend(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "bili": "bili-cli",
        "bili_cli": "bili-cli",
        "bilibili-cli": "bili-cli",
        "api": "bilibili-api",
        "search-api": "bilibili-api",
        "b站搜索 api": "bilibili-api",
    }
    return aliases.get(text, text)


def _routing_message(operation: str, errors: Sequence[str]) -> str:
    if errors:
        return f"Bilibili {operation} backends failed: {'; '.join(errors)}"
    return f"Bilibili {operation} has no available backend"


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text or "bilibili"


__all__ = [
    "BILIBILI_SEARCH_URL",
    "BILIBILI_VIEW_URL",
    "BILIBILI_PLAYER_URL",
    "BilibiliAudioRoute",
    "BilibiliBackendRouter",
    "BilibiliRoutingError",
    "BilibiliSearchRoute",
    "BilibiliVideoRoute",
    "DEFAULT_BILIBILI_BACKENDS",
    "extract_bilibili_id",
]
