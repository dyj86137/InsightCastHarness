"""Transcript 获取和规范化。"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from pydantic import Field

from insightcast.domain.enums import SourceType, TranscriptSource, TranscriptStatus
from insightcast.domain.models import CandidateItem, DomainModel, Interview, Transcript
from insightcast.integrations.bilibili import (
    BilibiliBackendRouter,
    extract_bilibili_id,
)
from insightcast.platform_cookies import build_ytdlp_cookie_options


DEFAULT_PREFERRED_CAPTION_LANGUAGES = (
    "zh-Hans",
    "zh-CN",
    "zh",
    "zh-TW",
    "en",
    "en-US",
    "en-GB",
)


class TranscriptFetchDependencyError(RuntimeError):
    """Transcript 获取依赖缺失或不可用。"""


class PlatformCaptionFetchResult(DomainModel):
    """一次平台字幕抓取结果。"""

    candidate_id: str
    text: str
    source: TranscriptSource = TranscriptSource.PLATFORM_CAPTION
    language: Optional[str] = None
    segments: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioDownloadResult(DomainModel):
    """一次音频下载结果。"""

    candidate_id: str
    audio_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioTranscriptionResult(DomainModel):
    """一次音频转写结果。"""

    text: str
    language: Optional[str] = None
    segments: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TranscriptFetchOutcome(DomainModel):
    """一次 transcript 获取的结果。"""

    transcript: Transcript
    used_candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)


PlatformCaptionFetcher = Callable[[CandidateItem], Optional[PlatformCaptionFetchResult]]
AudioDownloader = Callable[[CandidateItem], Optional[AudioDownloadResult]]
AudioTranscriber = Callable[
    [AudioDownloadResult, Interview, CandidateItem],
    Optional[AudioTranscriptionResult],
]


class TranscriptProvider:
    """Transcript 获取接口。"""

    def fetch(
        self,
        interview: Interview,
        candidates: Iterable[CandidateItem],
    ) -> TranscriptFetchOutcome:
        """为一个 Interview 获取可用文本。"""

        raise NotImplementedError


class CompositeTranscriptProvider(TranscriptProvider):
    """按顺序尝试多个 TranscriptProvider，直到拿到可用 transcript。"""

    def __init__(self, providers: Iterable[TranscriptProvider]) -> None:
        self.providers = tuple(providers)

    def fetch(
        self,
        interview: Interview,
        candidates: Iterable[CandidateItem],
    ) -> TranscriptFetchOutcome:
        candidate_list = tuple(candidates)
        last_unavailable: Optional[TranscriptFetchOutcome] = None
        for provider in self.providers:
            outcome = provider.fetch(interview, candidate_list)
            if outcome.transcript.status != TranscriptStatus.UNAVAILABLE:
                return outcome
            last_unavailable = outcome
        return last_unavailable or TranscriptFetchOutcome(
            transcript=unavailable_transcript(
                interview,
                provider="composite",
                reason="no transcript providers configured",
            )
        )


class PlatformCaptionTranscriptProvider(TranscriptProvider):
    """优先从平台已有字幕中获取完整 transcript。"""

    def __init__(
        self,
        *,
        fetcher: Optional[PlatformCaptionFetcher] = None,
        preferred_languages: Sequence[str] = DEFAULT_PREFERRED_CAPTION_LANGUAGES,
        timeout_seconds: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.preferred_languages = tuple(preferred_languages)
        self.timeout_seconds = timeout_seconds
        self.env = env
        self.fetcher = fetcher or (
            lambda candidate: fetch_platform_caption(
                candidate,
                preferred_languages=self.preferred_languages,
                timeout_seconds=self.timeout_seconds,
                env=self.env,
            )
        )

    def fetch(
        self,
        interview: Interview,
        candidates: Iterable[CandidateItem],
    ) -> TranscriptFetchOutcome:
        errors: List[Dict[str, Any]] = []
        for candidate in supported_platform_candidates(candidates):
            try:
                result = self.fetcher(candidate)
            except Exception as exc:
                errors.append(fetch_error_payload(candidate, exc))
                continue
            if result is None or not result.text.strip():
                continue
            transcript = Transcript(
                interview_id=interview.id,
                status=TranscriptStatus.READY,
                source=result.source,
                text=result.text.strip(),
                language=result.language,
                segments=result.segments,
                content_hash=content_hash(result.text),
                metadata={
                    "provider": "platform_caption",
                    "candidate_id": candidate.id,
                    "source_type": candidate.source_type.value,
                    **result.metadata,
                },
            )
            return TranscriptFetchOutcome(
                transcript=transcript,
                used_candidate_ids=(candidate.id,),
            )

        return TranscriptFetchOutcome(
            transcript=unavailable_transcript(
                interview,
                provider="platform_caption",
                reason="no platform captions available",
                errors=tuple(errors),
            )
        )


class AudioTranscriptionTranscriptProvider(TranscriptProvider):
    """平台字幕不可用时，下载音频并调用外部转写能力。"""

    def __init__(
        self,
        *,
        downloader: AudioDownloader,
        transcriber: AudioTranscriber,
    ) -> None:
        self.downloader = downloader
        self.transcriber = transcriber

    def fetch(
        self,
        interview: Interview,
        candidates: Iterable[CandidateItem],
    ) -> TranscriptFetchOutcome:
        errors: List[Dict[str, Any]] = []
        for candidate in supported_platform_candidates(candidates):
            try:
                download = self.downloader(candidate)
                if download is None:
                    continue
                transcription = self.transcriber(download, interview, candidate)
            except Exception as exc:
                errors.append(fetch_error_payload(candidate, exc))
                continue
            if transcription is None or not transcription.text.strip():
                continue
            transcript = Transcript(
                interview_id=interview.id,
                status=TranscriptStatus.READY,
                source=TranscriptSource.AUDIO_TRANSCRIPTION,
                text=transcription.text.strip(),
                language=transcription.language,
                segments=transcription.segments,
                content_hash=content_hash(transcription.text),
                metadata={
                    "provider": "audio_transcription",
                    "candidate_id": candidate.id,
                    "audio_path": download.audio_path,
                    "download": dict(download.metadata),
                    **transcription.metadata,
                },
            )
            return TranscriptFetchOutcome(
                transcript=transcript,
                used_candidate_ids=(candidate.id,),
            )

        return TranscriptFetchOutcome(
            transcript=unavailable_transcript(
                interview,
                provider="audio_transcription",
                reason="audio transcription unavailable",
                errors=tuple(errors),
            )
        )


class CandidateMetadataTranscriptProvider(TranscriptProvider):
    """从候选内容的标题、简介和平台元数据中提取可用文本。"""

    def fetch(
        self,
        interview: Interview,
        candidates: Iterable[CandidateItem],
    ) -> TranscriptFetchOutcome:
        candidate_list = tuple(candidates)
        text = build_source_metadata_text(interview, candidate_list)
        if text:
            transcript = Transcript(
                interview_id=interview.id,
                status=TranscriptStatus.PARTIAL,
                source=TranscriptSource.SOURCE_METADATA,
                text=text,
                content_hash=content_hash(text),
                metadata={
                    "provider": "candidate_metadata",
                    "is_full_transcript": False,
                },
            )
            return TranscriptFetchOutcome(
                transcript=transcript,
                used_candidate_ids=tuple(candidate.id for candidate in candidate_list),
            )

        transcript = unavailable_transcript(
            interview,
            provider="candidate_metadata",
            reason="no candidate title, description, or metadata text available",
        )
        return TranscriptFetchOutcome(transcript=transcript)


def create_default_transcript_provider(
    *,
    env: Optional[Mapping[str, str]] = None,
    platform_caption_fetcher: Optional[PlatformCaptionFetcher] = None,
    audio_downloader: Optional[AudioDownloader] = None,
    audio_transcriber: Optional[AudioTranscriber] = None,
    audio_output_dir: Optional[Path] = None,
) -> TranscriptProvider:
    """创建 daily pipeline 默认的 transcript provider 链。"""

    source = os.environ if env is None else env
    preferred_languages = env_csv(
        source,
        "INSIGHTCAST_TRANSCRIPT_LANGUAGES",
        DEFAULT_PREFERRED_CAPTION_LANGUAGES,
    )
    providers: List[TranscriptProvider] = []
    if env_flag(source, "INSIGHTCAST_TRANSCRIPT_PLATFORM_CAPTIONS", True):
        providers.append(
            PlatformCaptionTranscriptProvider(
                fetcher=platform_caption_fetcher,
                preferred_languages=preferred_languages,
                env=source,
            )
        )

    audio_enabled = env_flag(
        source,
        "INSIGHTCAST_TRANSCRIPT_AUDIO_TRANSCRIPTION",
        audio_transcriber is not None,
    )
    if audio_enabled and audio_transcriber is not None:
        output_dir = audio_output_dir or Path(".insightcast") / "transcript_audio"
        providers.append(
            AudioTranscriptionTranscriptProvider(
                downloader=audio_downloader
                or (
                    lambda candidate: download_candidate_audio(
                        candidate,
                        output_dir=output_dir,
                        env=source,
                    )
                ),
                transcriber=audio_transcriber,
            )
        )

    providers.append(CandidateMetadataTranscriptProvider())
    return CompositeTranscriptProvider(providers)


def fetch_platform_caption(
    candidate: CandidateItem,
    *,
    preferred_languages: Sequence[str] = DEFAULT_PREFERRED_CAPTION_LANGUAGES,
    timeout_seconds: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[PlatformCaptionFetchResult]:
    """按平台获取已有字幕；Bilibili 使用专用后端。"""

    if candidate.source_type == SourceType.BILIBILI:
        return fetch_bilibili_caption(
            candidate,
            preferred_languages=preferred_languages,
            timeout_seconds=timeout_seconds,
            env=env,
        )

    if not is_supported_platform_candidate(candidate):
        return None
    info = extract_media_info(candidate, timeout_seconds=timeout_seconds, env=env)
    selected = select_caption_track(
        info,
        candidate,
        preferred_languages=preferred_languages,
    )
    if selected is None:
        return None

    raw = fetch_text_url(
        str(selected["url"]),
        timeout_seconds=timeout_seconds,
    )
    text = parse_caption_payload(raw, ext=str(selected.get("ext") or ""))
    if not text:
        return None
    source = (
        TranscriptSource.OFFICIAL_CAPTION
        if selected.get("source_kind") == "manual"
        else TranscriptSource.PLATFORM_CAPTION
    )
    return PlatformCaptionFetchResult(
        candidate_id=candidate.id,
        text=text,
        source=source,
        language=str(selected.get("language") or "") or None,
        segments=tuple(line for line in text.splitlines() if line.strip()),
        metadata={
            "fetcher": "yt_dlp",
            "caption_source_kind": selected.get("source_kind"),
            "caption_language": selected.get("language"),
            "caption_ext": selected.get("ext"),
            "media_title": info.get("title"),
            "webpage_url": info.get("webpage_url") or str(candidate.url),
        },
    )


def fetch_bilibili_caption(
    candidate: CandidateItem,
    *,
    preferred_languages: Sequence[str] = DEFAULT_PREFERRED_CAPTION_LANGUAGES,
    timeout_seconds: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[PlatformCaptionFetchResult]:
    """通过 bili-cli -> B站 player API 获取字幕。"""

    bvid = extract_bilibili_id(candidate)
    if not bvid:
        return None

    router = BilibiliBackendRouter(env=env)
    page_number = _bilibili_page_number(candidate)
    route = router.video(
        bvid,
        page_number=page_number,
        timeout_seconds=timeout_seconds,
    )
    entries = _bilibili_subtitle_entries(route.payload)
    language_order = caption_language_order(candidate, preferred_languages)
    for entry in _order_bilibili_subtitles(entries, language_order):
        language = _bilibili_subtitle_language(entry)
        text = _bilibili_inline_subtitle_text(entry)
        if not text:
            subtitle_url = _bilibili_subtitle_url(entry)
            if subtitle_url:
                raw = fetch_text_url(
                    subtitle_url,
                    timeout_seconds=timeout_seconds,
                    headers=router.headers(),
                )
                text = parse_caption_payload(raw)
        if not text:
            continue
        return PlatformCaptionFetchResult(
            candidate_id=candidate.id,
            text=text,
            source=TranscriptSource.PLATFORM_CAPTION,
            language=language,
            segments=tuple(line for line in text.splitlines() if line.strip()),
            metadata={
                "fetcher": route.backend,
                "backend": route.backend,
                "backend_route": list(router.backends),
                "bvid": bvid,
                "caption_source_kind": "platform",
                "webpage_url": str(candidate.url),
            },
        )
    return None


def download_candidate_audio(
    candidate: CandidateItem,
    *,
    output_dir: Path,
    timeout_seconds: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[AudioDownloadResult]:
    """按平台下载候选音频；Bilibili 使用专用后端。"""

    if candidate.source_type == SourceType.BILIBILI:
        return download_bilibili_audio(
            candidate,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            env=env,
        )

    if not is_supported_platform_candidate(candidate):
        return None
    yt_dlp = import_yt_dlp()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = safe_filename(candidate.id)
    output_template = str(output_dir / f"{safe_id}.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if timeout_seconds is not None:
        options["socket_timeout"] = timeout_seconds
    options.update(build_ytdlp_cookie_options(env))

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(str(candidate.url), download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        matches = sorted(output_dir.glob(f"{safe_id}.*"))
        if not matches:
            return None
        path = matches[-1]
    return AudioDownloadResult(
        candidate_id=candidate.id,
        audio_path=str(path),
        metadata={
            "downloader": "yt_dlp",
            "source_type": candidate.source_type.value,
            "media_title": info.get("title") if isinstance(info, Mapping) else None,
            "ext": path.suffix.lstrip("."),
        },
    )


def download_bilibili_audio(
    candidate: CandidateItem,
    *,
    output_dir: Path,
    timeout_seconds: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[AudioDownloadResult]:
    """通过 bili-cli 下载 B 站音频，失败时不静默退回 yt-dlp。"""

    bvid = extract_bilibili_id(candidate)
    if not bvid:
        return None
    router = BilibiliBackendRouter(env=env)
    route = router.download_audio(
        bvid,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )
    if route is None:
        return None
    return AudioDownloadResult(
        candidate_id=candidate.id,
        audio_path=str(route.audio_path),
        metadata={
            "downloader": route.backend,
            "backend": route.backend,
            "backend_route": list(router.backends),
            "source_type": candidate.source_type.value,
            "bvid": bvid,
            "ext": route.audio_path.suffix.lstrip("."),
        },
    )


def extract_media_info(
    candidate: CandidateItem,
    *,
    timeout_seconds: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    """通过 yt-dlp 提取支持平台的媒体元数据但不下载媒体文件。"""

    if candidate.source_type == SourceType.BILIBILI:
        raise TranscriptFetchDependencyError(
            "Bilibili media info must use BilibiliBackendRouter, not yt-dlp."
        )

    yt_dlp = import_yt_dlp()
    options: Dict[str, Any] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if timeout_seconds is not None:
        options["socket_timeout"] = timeout_seconds
    options.update(build_ytdlp_cookie_options(env))
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(str(candidate.url), download=False)
    if not isinstance(info, Mapping):
        raise RuntimeError("yt-dlp returned non-mapping media info")
    return info


def select_caption_track(
    info: Mapping[str, Any],
    candidate: CandidateItem,
    *,
    preferred_languages: Sequence[str] = DEFAULT_PREFERRED_CAPTION_LANGUAGES,
) -> Optional[Mapping[str, Any]]:
    """从 yt-dlp 媒体信息里选择最合适的一条字幕轨。"""

    language_order = caption_language_order(candidate, preferred_languages)
    for source_kind, field_name in (
        ("manual", "subtitles"),
        ("automatic", "automatic_captions"),
    ):
        tracks = info.get(field_name)
        if not isinstance(tracks, Mapping):
            continue
        language = select_language(tracks, language_order)
        if language is None:
            continue
        entry = select_subtitle_entry(tracks.get(language))
        if entry is None:
            continue
        return {
            "source_kind": source_kind,
            "language": language,
            "url": entry.get("url"),
            "ext": entry.get("ext"),
        }
    return None


def build_source_metadata_text(
    interview: Interview,
    candidates: Iterable[CandidateItem],
) -> str:
    """把候选内容中的可用文本整理成 partial transcript。"""

    candidate_sections = [
        candidate_text
        for candidate in candidates
        for candidate_text in (candidate_source_text(candidate),)
        if candidate_text
    ]
    if not candidate_sections:
        return ""

    sections: List[str] = []
    if interview.title:
        sections.append(f"Interview title: {interview.title}")
    sections.extend(candidate_sections)
    return "\n\n".join(dedupe_text_sections(sections)).strip()


def candidate_source_text(candidate: CandidateItem) -> str:
    """提取单条候选内容中的标题、简介和少量平台文本。"""

    evidence_parts: List[str] = []
    if candidate.description:
        evidence_parts.append(f"Description: {candidate.description}")

    for key in ("description", "desc", "summary", "subtitle", "dynamic"):
        value = candidate.raw_metadata.get(key)
        if isinstance(value, str) and value.strip():
            evidence_parts.append(f"{key}: {value.strip()}")

    snippet = candidate.raw_metadata.get("snippet")
    if isinstance(snippet, dict):
        value = snippet.get("description")
        if isinstance(value, str) and value.strip():
            evidence_parts.append(f"snippet.description: {value.strip()}")

    if not evidence_parts:
        return ""
    parts: List[str] = []
    if candidate.title:
        parts.append(f"Candidate title: {candidate.title}")
    parts.extend(evidence_parts)
    return "\n".join(dedupe_text_sections(parts)).strip()


def parse_caption_payload(raw: str, *, ext: str = "") -> str:
    """解析常见字幕格式为纯文本。"""

    stripped = raw.strip()
    if not stripped:
        return ""
    normalized_ext = ext.strip().lower()
    if normalized_ext in ("json", "json3") or stripped.startswith(("{", "[")):
        text = parse_json_caption(stripped)
        if text:
            return text
    if normalized_ext in ("srv1", "srv2", "srv3", "ttml", "xml") or stripped.startswith("<"):
        text = parse_xml_caption(stripped)
        if text:
            return text
    return parse_line_caption(stripped)


def _bilibili_page_number(candidate: CandidateItem) -> int:
    for key in ("page", "p", "page_number"):
        value = candidate.raw_metadata.get(key)
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return 1


def _bilibili_subtitle_entries(payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    """提取 bili-cli 和 player API 两种 payload 中的字幕条目。"""

    entries: List[Any] = []

    def collect_container(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                collect_container(item)
            return
        if isinstance(value, str):
            if value.strip():
                entries.append(value)
            return
        if not isinstance(value, Mapping):
            return

        if any(
            key in value
            for key in (
                "text",
                "content",
                "transcript",
                "subtitle_url",
                "url",
                "body",
            )
        ):
            entries.append(value)
        for key in ("subtitles", "items", "body", "segments"):
            if key in value:
                collect_container(value.get(key))

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in {"subtitle", "subtitles"}:
                    collect_container(item)
                elif lowered in {"data", "player", "video"}:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return tuple(entries)


def _order_bilibili_subtitles(
    entries: Sequence[Any],
    preferred_languages: Sequence[str],
) -> Tuple[Any, ...]:
    preference = {
        normalize_language(language): index
        for index, language in enumerate(preferred_languages)
    }

    def rank(entry: Any) -> Tuple[int, int]:
        language = normalize_language(_bilibili_subtitle_language(entry) or "")
        if language in preference:
            return (0, preference[language])
        base = language.split("-", 1)[0] if language else ""
        for preferred, index in preference.items():
            if base and base == preferred.split("-", 1)[0]:
                return (1, index)
        return (2, 0)

    return tuple(sorted(entries, key=rank))


def _bilibili_subtitle_language(entry: Any) -> Optional[str]:
    if not isinstance(entry, Mapping):
        return None
    for key in ("language", "lang", "lan", "lan_doc", "language_name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _bilibili_inline_subtitle_text(entry: Any) -> str:
    if isinstance(entry, str):
        return parse_caption_payload(entry)
    if not isinstance(entry, Mapping):
        return ""

    body = entry.get("body")
    if isinstance(body, list):
        values = []
        for item in body:
            if isinstance(item, Mapping):
                value = item.get("content") or item.get("text") or item.get("utf8")
                if isinstance(value, str):
                    values.append(value)
        if values:
            return "\n".join(clean_caption_lines(values))

    for key in ("text", "content", "transcript"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return parse_caption_payload(value)
    return ""


def _bilibili_subtitle_url(entry: Any) -> Optional[str]:
    if not isinstance(entry, Mapping):
        return None
    for key in ("subtitle_url", "url", "download_url"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            url = value.strip()
            return f"https:{url}" if url.startswith("//") else url
    return None


def parse_json_caption(raw: str) -> str:
    """解析 YouTube json3 和 Bilibili JSON 字幕。"""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""

    lines: List[str] = []
    if isinstance(payload, Mapping):
        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, Mapping):
                    continue
                segs = event.get("segs")
                if not isinstance(segs, list):
                    continue
                lines.append(
                    "".join(
                        str(seg.get("utf8") or "")
                        for seg in segs
                        if isinstance(seg, Mapping)
                    )
                )

        body = payload.get("body")
        if isinstance(body, list):
            for item in body:
                if isinstance(item, Mapping):
                    lines.append(str(item.get("content") or item.get("text") or ""))

        if not lines:
            for key in ("text", "transcript", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    lines.append(value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, Mapping):
                lines.append(str(item.get("content") or item.get("text") or ""))

    return "\n".join(clean_caption_lines(lines))


def parse_xml_caption(raw: str) -> str:
    """解析 srv/ttml/xml 字幕。"""

    try:
        root = ElementTree.fromstring(raw.encode("utf-8"))
    except ElementTree.ParseError:
        return ""

    lines: List[str] = []
    for element in root.iter():
        tag = element.tag.split("}")[-1].lower()
        if tag in ("text", "p") and element.text:
            lines.append(element.text)
    return "\n".join(clean_caption_lines(lines))


def parse_line_caption(raw: str) -> str:
    """解析 vtt/srt 等基于行的字幕格式。"""

    lines: List[str] = []
    skip_note = False
    for line in raw.replace("\r", "").split("\n"):
        text = line.strip()
        upper = text.upper()
        if not text:
            skip_note = False
            continue
        if upper.startswith(("WEBVTT", "STYLE", "REGION")):
            continue
        if upper.startswith("NOTE"):
            skip_note = True
            continue
        if skip_note:
            continue
        if "-->" in text:
            continue
        if text.isdigit():
            continue
        if re.match(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{3}$", text):
            continue
        lines.append(text)
    return "\n".join(clean_caption_lines(lines))


def clean_caption_lines(values: Iterable[str]) -> Tuple[str, ...]:
    """清理字幕行并去掉连续重复内容。"""

    cleaned: List[str] = []
    previous = ""
    for value in values:
        text = clean_caption_text(value)
        if not text or text == previous:
            continue
        cleaned.append(text)
        previous = text
    return tuple(cleaned)


def clean_caption_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(value))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe_text_sections(values: Iterable[str]) -> Tuple[str, ...]:
    """按文本内容去重，同时保留原顺序。"""

    seen = set()
    result: List[str] = []
    for value in values:
        text = " ".join(str(value).strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(str(value).strip())
    return tuple(result)


def content_hash(text: str) -> str:
    """生成文本内容 hash，用于后续去重和变更检测。"""

    from hashlib import sha1

    return sha1(text.encode("utf-8")).hexdigest()


def supported_platform_candidates(
    candidates: Iterable[CandidateItem],
) -> Tuple[CandidateItem, ...]:
    return tuple(candidate for candidate in candidates if is_supported_platform_candidate(candidate))


def is_supported_platform_candidate(candidate: CandidateItem) -> bool:
    return candidate.source_type in {
        SourceType.YOUTUBE,
        SourceType.BILIBILI,
        SourceType.VIMEO,
        SourceType.DAILYMOTION,
        SourceType.TWITCH,
        SourceType.PEERTUBE,
    }


def import_yt_dlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise TranscriptFetchDependencyError(
            "yt-dlp is required for supported-platform captions and audio fallback."
        ) from exc
    return yt_dlp


def fetch_text_url(
    url: str,
    *,
    timeout_seconds: Optional[float] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> str:
    timeout = timeout_seconds or 20.0
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }
    request_headers.update(dict(headers or {}))
    request = Request(
        url,
        headers=request_headers,
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def caption_language_order(
    candidate: CandidateItem,
    preferred_languages: Sequence[str],
) -> Tuple[str, ...]:
    values: List[str] = []
    for key in ("language", "lang", "default_language"):
        value = candidate.raw_metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    values.extend(preferred_languages)
    if candidate.source_type == SourceType.BILIBILI:
        values.extend(("zh-Hans", "zh-CN", "zh"))
    return tuple(dedupe_text_sections(values))


def select_language(
    tracks: Mapping[str, Any],
    preferred_languages: Sequence[str],
) -> Optional[str]:
    available = [key for key, value in tracks.items() if value]
    if not available:
        return None
    normalized = {normalize_language(key): key for key in available}
    for preferred in preferred_languages:
        key = normalized.get(normalize_language(preferred))
        if key is not None:
            return key
    for preferred in preferred_languages:
        preferred_base = normalize_language(preferred).split("-")[0]
        for language in available:
            if normalize_language(language).split("-")[0] == preferred_base:
                return language
    return sorted(available)[0]


def normalize_language(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def select_subtitle_entry(entries: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(entries, Mapping):
        entry_list = [entries]
    elif isinstance(entries, list):
        entry_list = [entry for entry in entries if isinstance(entry, Mapping)]
    else:
        return None

    preferred_exts = ("json3", "json", "vtt", "srv3", "srv2", "srv1", "ttml", "srt")
    for ext in preferred_exts:
        for entry in entry_list:
            if str(entry.get("ext") or "").lower() == ext and entry.get("url"):
                return entry
    for entry in entry_list:
        if entry.get("url"):
            return entry
    return None


def unavailable_transcript(
    interview: Interview,
    *,
    provider: str,
    reason: str,
    errors: Tuple[Dict[str, Any], ...] = (),
) -> Transcript:
    metadata: Dict[str, Any] = {
        "provider": provider,
        "reason": reason,
    }
    if errors:
        metadata["errors"] = list(errors)
    return Transcript(
        interview_id=interview.id,
        status=TranscriptStatus.UNAVAILABLE,
        source=TranscriptSource.UNKNOWN,
        metadata=metadata,
    )


def fetch_error_payload(candidate: CandidateItem, exc: Exception) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "source_type": candidate.source_type.value,
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }


def env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def env_csv(
    env: Mapping[str, str],
    key: str,
    default: Sequence[str],
) -> Tuple[str, ...]:
    value = env.get(key)
    if not value:
        return tuple(default)
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or tuple(default)


def safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text or "audio"


__all__ = [
    "AudioDownloader",
    "AudioDownloadResult",
    "AudioTranscriber",
    "AudioTranscriptionResult",
    "AudioTranscriptionTranscriptProvider",
    "CandidateMetadataTranscriptProvider",
    "CompositeTranscriptProvider",
    "DEFAULT_PREFERRED_CAPTION_LANGUAGES",
    "PlatformCaptionFetcher",
    "PlatformCaptionFetchResult",
    "PlatformCaptionTranscriptProvider",
    "TranscriptFetchDependencyError",
    "TranscriptFetchOutcome",
    "TranscriptProvider",
    "build_source_metadata_text",
    "candidate_source_text",
    "content_hash",
    "create_default_transcript_provider",
    "dedupe_text_sections",
    "download_bilibili_audio",
    "download_candidate_audio",
    "fetch_bilibili_caption",
    "fetch_platform_caption",
    "parse_caption_payload",
]
