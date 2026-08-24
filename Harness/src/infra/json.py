"""JSON 文本提取与格式化工具。"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_value(text: str) -> object:
    """从文本中解析第一个 JSON 值。

    支持纯 JSON、Markdown code fence 包裹的 JSON，以及文本中夹带的
    JSON object/array。若无法解析则抛出 ValueError。
    """

    stripped = strip_json_code_fence(text).strip()
    decoder = json.JSONDecoder()
    try:
        value, index = decoder.raw_decode(stripped)
        if stripped[index:].strip():
            raise ValueError("Trailing content after JSON payload.")
        return value
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("No JSON payload found.")


def strip_json_code_fence(text: str) -> str:
    """移除包裹 JSON 的 Markdown code fence。"""

    stripped = (text or "").strip()
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1)
    return stripped


def compact_json(value: Any) -> str:
    """把结构化值紧凑格式化为 JSON。"""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


__all__ = [
    "compact_json",
    "parse_json_value",
    "strip_json_code_fence",
]
