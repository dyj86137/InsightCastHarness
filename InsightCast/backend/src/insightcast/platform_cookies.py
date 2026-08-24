"""平台登录态 Cookie 解析工具。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


BILIBILI_COOKIE_ENV = "INSIGHTCAST_BILIBILI_COOKIE"
BILIBILI_COOKIE_FILE_ENV = "INSIGHTCAST_BILIBILI_COOKIE_FILE"
YTDLP_COOKIE_ENV = "INSIGHTCAST_YTDLP_COOKIE"
YTDLP_COOKIES_FILE_ENV = "INSIGHTCAST_YTDLP_COOKIES_FILE"
YTDLP_COOKIES_FROM_BROWSER_ENV = "INSIGHTCAST_YTDLP_COOKIES_FROM_BROWSER"


def resolve_cookie_header(
    env: Optional[Mapping[str, str]] = None,
    *,
    cookie: Optional[str] = None,
    cookie_file: Optional[str] = None,
    cookie_env: str = BILIBILI_COOKIE_ENV,
    cookie_file_env: str = BILIBILI_COOKIE_FILE_ENV,
    domains: Sequence[str] = (),
) -> Optional[str]:
    """从直接配置、环境变量或 cookies 文件解析 HTTP Cookie header。"""

    source = os.environ if env is None else env
    direct = _first_non_empty(
        cookie,
        source.get(cookie_env),
    )
    if direct:
        return direct

    file_value = _first_non_empty(
        cookie_file,
        source.get(cookie_file_env),
    )
    if not file_value:
        return None
    return cookie_header_from_file(Path(file_value), domains=domains)


def cookie_header_from_file(
    path: Path,
    *,
    domains: Sequence[str] = (),
) -> Optional[str]:
    """从 raw Cookie header 或 Netscape cookies.txt 文件生成 Cookie header。"""

    cookie_path = path.expanduser()
    if not cookie_path.exists():
        raise FileNotFoundError(f"Cookie 文件不存在：{cookie_path}")

    text = cookie_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    pairs = parse_netscape_cookie_pairs(text, domains=domains)
    if pairs:
        return "; ".join(pairs)

    raw_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("cookie:"):
            line = line.split(":", 1)[1].strip()
        raw_lines.append(line)

    if not raw_lines:
        return None
    return "; ".join(raw_lines)


def parse_netscape_cookie_pairs(
    text: str,
    *,
    domains: Sequence[str] = (),
) -> Tuple[str, ...]:
    """解析 Netscape cookies.txt 格式，返回 name=value 列表。"""

    result = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_") :]
        elif line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain = parts[0].strip()
        name = parts[5].strip()
        value = parts[6].strip()
        if not name:
            continue
        if domains and not any(domain_matches(domain, expected) for expected in domains):
            continue
        result.append(f"{name}={value}")
    return tuple(result)


def build_ytdlp_cookie_options(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """把 InsightCast Cookie 环境变量转换为 yt-dlp options。"""

    source = os.environ if env is None else env
    options: Dict[str, Any] = {}

    cookie_file = _first_non_empty(source.get(YTDLP_COOKIES_FILE_ENV))
    if cookie_file:
        cookie_path = Path(cookie_file).expanduser()
        if not cookie_path.exists():
            raise FileNotFoundError(f"yt-dlp Cookie 文件不存在：{cookie_path}")
        options["cookiefile"] = str(cookie_path)
    else:
        bilibili_cookie_file = _first_non_empty(source.get(BILIBILI_COOKIE_FILE_ENV))
        if bilibili_cookie_file:
            cookie_path = Path(bilibili_cookie_file).expanduser()
            if not cookie_path.exists():
                raise FileNotFoundError(f"yt-dlp Cookie 文件不存在：{cookie_path}")
            text = cookie_path.read_text(encoding="utf-8")
            if parse_netscape_cookie_pairs(text, domains=("bilibili.com",)):
                options["cookiefile"] = str(cookie_path)
            else:
                cookie_header = cookie_header_from_file(
                    cookie_path,
                    domains=("bilibili.com",),
                )
                if cookie_header:
                    options["http_headers"] = {"Cookie": cookie_header}

    browser = _first_non_empty(source.get(YTDLP_COOKIES_FROM_BROWSER_ENV))
    if browser:
        options["cookiesfrombrowser"] = parse_browser_cookie_spec(browser)

    raw_cookie = _first_non_empty(
        source.get(YTDLP_COOKIE_ENV),
        source.get(BILIBILI_COOKIE_ENV),
    )
    if raw_cookie:
        options["http_headers"] = {"Cookie": raw_cookie}

    return options


def parse_browser_cookie_spec(value: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """解析 yt-dlp cookiesfrombrowser 简写：browser[:profile[:keyring[:container]]]。"""

    parts = [part.strip() or None for part in value.split(":", 3)]
    while len(parts) < 4:
        parts.append(None)
    return parts[0], parts[1], parts[2], parts[3]


def domain_matches(actual: str, expected: str) -> bool:
    """判断 cookies.txt 中的 domain 是否属于期望域名。"""

    actual_domain = actual.strip().lower().lstrip(".")
    expected_domain = expected.strip().lower().lstrip(".")
    return bool(actual_domain and expected_domain) and (
        actual_domain == expected_domain
        or actual_domain.endswith(f".{expected_domain}")
    )


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


__all__ = [
    "BILIBILI_COOKIE_ENV",
    "BILIBILI_COOKIE_FILE_ENV",
    "YTDLP_COOKIE_ENV",
    "YTDLP_COOKIES_FILE_ENV",
    "YTDLP_COOKIES_FROM_BROWSER_ENV",
    "build_ytdlp_cookie_options",
    "cookie_header_from_file",
    "domain_matches",
    "parse_browser_cookie_spec",
    "parse_netscape_cookie_pairs",
    "resolve_cookie_header",
]
