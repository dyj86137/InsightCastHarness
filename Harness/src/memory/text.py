"""记忆检索文本规范化与分词工具。"""

from __future__ import annotations

import re
from typing import List, Set

from memory.base import MemoryRecord


ASCII_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def record_text(record: MemoryRecord) -> str:
    """返回用于检索和索引的记录文本。"""

    parts = [record.content, record.summary or "", " ".join(record.tags)]
    return "\n".join(part for part in parts if part)


def tokenize(text: str) -> List[str]:
    """把文本转换为适合轻量检索的 token 列表。"""

    normalized = (text or "").lower()
    tokens = [token for token in ASCII_TOKEN_PATTERN.findall(normalized) if token]
    cjk_chars = CJK_CHAR_PATTERN.findall(normalized)
    tokens.extend(cjk_chars)
    tokens.extend(
        "".join(pair)
        for pair in zip(cjk_chars, cjk_chars[1:])
    )
    return tokens


def token_set(text: str) -> Set[str]:
    """返回去重 token 集合。"""

    return set(tokenize(text))


__all__ = [
    "record_text",
    "token_set",
    "tokenize",
]
