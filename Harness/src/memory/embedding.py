"""记忆向量化抽象与本地 fallback 实现。"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from memory.text import tokenize


class EmbeddingProvider(ABC):
    """文本向量化 Provider 抽象。"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回 Provider 名称。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称。"""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """返回向量维度。"""

    async def embed_query(self, text: str) -> Tuple[float, ...]:
        """向量化查询文本。"""

        embeddings = await self.embed_texts([text])
        return embeddings[0]

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> Tuple[Tuple[float, ...], ...]:
        """批量向量化文本。"""


class HashEmbeddingProvider(EmbeddingProvider):
    """可离线运行的确定性 embedding fallback。

    该实现用于验证向量索引流水线，不提供真实语义泛化能力。
    """

    def __init__(self, *, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "local_hash"

    @property
    def model_name(self) -> str:
        return f"hash-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: Sequence[str]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> Tuple[float, ...]:
        vector = [0.0 for _ in range(self._dimensions)]
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self._dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        return normalize_vector(vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """计算余弦相似度。"""

    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def normalize_vector(values: Sequence[float]) -> Tuple[float, ...]:
    """返回 L2 归一化后的向量。"""

    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "cosine_similarity",
    "normalize_vector",
]
