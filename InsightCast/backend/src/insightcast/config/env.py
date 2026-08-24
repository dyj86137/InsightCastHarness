"""加载 InsightCast 本地 .env 文件。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def load_env_file(
    path: Optional[Path] = None,
    *,
    override: bool = False,
) -> Dict[str, str]:
    """把 .env 文件中的键值加载到 os.environ。"""

    env_path = Path(path) if path is not None else default_env_path()
    if not env_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)
    return loaded


def default_env_path() -> Path:
    """返回 backend 根目录下的默认 .env 路径。"""

    return Path(__file__).resolve().parents[3] / ".env"


def parse_env_line(line: str) -> Optional[tuple[str, str]]:
    """解析一行 KEY=value 格式的 .env 内容。"""

    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()
    if "=" not in text:
        return None

    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, parse_env_value(raw_value)


def parse_env_value(value: str) -> str:
    """解析 .env 右侧值，支持引号和未加引号的行尾注释。"""

    text = value.strip()
    if not text:
        return ""

    quote = text[0]
    if quote in ("'", '"'):
        chars = []
        escaped = False
        for char in text[1:]:
            if escaped:
                chars.append(_unescape_char(char))
                escaped = False
                continue
            if quote == '"' and char == "\\":
                escaped = True
                continue
            if char == quote:
                return "".join(chars)
            chars.append(char)
        return "".join(chars)

    return strip_inline_comment(text).strip()


def strip_inline_comment(value: str) -> str:
    """去掉未加引号值中的行尾注释。"""

    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def _unescape_char(char: str) -> str:
    return {
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }.get(char, char)


__all__ = [
    "default_env_path",
    "load_env_file",
    "parse_env_line",
    "parse_env_value",
    "strip_inline_comment",
]
