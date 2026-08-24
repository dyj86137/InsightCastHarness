"""记忆向量索引。"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import Field

from infra.exception import SerializationError
from infra.serialization import SerializableModel
from memory.base import MemoryError, MemoryRecord
from memory.embedding import EmbeddingProvider, cosine_similarity
from memory.text import record_text


class VectorSearchResult(SerializableModel):
    """向量召回结果。"""

    memory_id: str
    score: float
    rank: int

    model_config = SerializableModel.config(frozen=True)


class MemoryIndexEntry(SerializableModel):
    """一条可重建的向量索引条目。"""

    memory_id: str
    content_hash: str
    text: str
    embedding: Tuple[float, ...]
    embedding_provider: str
    embedding_model: str
    metadata: Dict[str, object] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = SerializableModel.config(frozen=True)


class MemoryVectorIndex(ABC):
    """向量索引统一接口。"""

    @abstractmethod
    def upsert(self, entry: MemoryIndexEntry) -> None:
        """新增或更新索引条目。"""

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryIndexEntry]:
        """按 memory_id 读取索引条目。"""

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """删除索引条目。"""

    @abstractmethod
    def list_entries(self) -> List[MemoryIndexEntry]:
        """返回所有索引条目。"""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[VectorSearchResult]:
        """按余弦相似度检索。"""


class InMemoryVectorIndex(MemoryVectorIndex):
    """内存向量索引。"""

    def __init__(self, entries: Optional[Iterable[MemoryIndexEntry]] = None) -> None:
        self._entries: Dict[str, MemoryIndexEntry] = {}
        for entry in entries or ():
            self._entries[entry.memory_id] = entry

    def upsert(self, entry: MemoryIndexEntry) -> None:
        self._entries[entry.memory_id] = entry

    def get(self, memory_id: str) -> Optional[MemoryIndexEntry]:
        return self._entries.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self._entries.pop(memory_id, None)

    def list_entries(self) -> List[MemoryIndexEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.updated_at, reverse=True)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[VectorSearchResult]:
        if top_k <= 0 or not query_vector:
            return []
        candidate_set = set(candidate_ids) if candidate_ids is not None else set(self._entries)
        scored = []
        for memory_id in candidate_set:
            entry = self._entries.get(memory_id)
            if entry is None:
                continue
            score = cosine_similarity(query_vector, entry.embedding)
            if score <= 0:
                continue
            scored.append((memory_id, score))
        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return [
            VectorSearchResult(memory_id=memory_id, score=score, rank=rank)
            for rank, (memory_id, score) in enumerate(scored[:top_k], start=1)
        ]


class JsonlVectorIndex(MemoryVectorIndex):
    """基于 JSONL 文件的本地向量索引。"""

    def __init__(
        self,
        directory: Union[str, Path] = "memories",
        *,
        filename: str = "memory_vectors.jsonl",
        create_dir: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.filename = filename
        self._index = InMemoryVectorIndex()
        if create_dir:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    def upsert(self, entry: MemoryIndexEntry) -> None:
        self._index.upsert(entry)
        self._persist()

    def get(self, memory_id: str) -> Optional[MemoryIndexEntry]:
        return self._index.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self._index.delete(memory_id)
        self._persist()

    def list_entries(self) -> List[MemoryIndexEntry]:
        return self._index.list_entries()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[VectorSearchResult]:
        return self._index.search(
            query_vector,
            top_k=top_k,
            candidate_ids=candidate_ids,
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        entries = []
        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entries.append(MemoryIndexEntry.from_json(stripped))
                    except SerializationError as exc:
                        raise MemoryError(
                            "Failed to parse memory vector index line.",
                            details={"path": str(self.path), "line_number": line_number},
                            cause=exc,
                        ) from exc
        except OSError as exc:
            raise MemoryError(
                "Failed to read memory vector index.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc
        self._index = InMemoryVectorIndex(entries)

    def _persist(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as file:
                for entry in self._index.list_entries():
                    file.write(entry.to_json(exclude_none=True))
                    file.write("\n")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise MemoryError(
                "Failed to write memory vector index.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc


async def build_memory_index_entry(
    record: MemoryRecord,
    embedding_provider: EmbeddingProvider,
) -> MemoryIndexEntry:
    """从 MemoryRecord 构建向量索引条目。"""

    text = record_text(record)
    embedding = await embedding_provider.embed_query(text)
    return MemoryIndexEntry(
        memory_id=record.id,
        content_hash=memory_content_hash(record),
        text=text,
        embedding=tuple(embedding),
        embedding_provider=embedding_provider.provider_name,
        embedding_model=embedding_provider.model_name,
        metadata={
            "kind": record.kind.value,
            "scope": record.scope.value,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "user_id": record.user_id,
            "tags": list(record.tags),
        },
    )


def memory_content_hash(record: MemoryRecord) -> str:
    """返回记录可检索内容的稳定 hash。"""

    digest = hashlib.sha256()
    digest.update(record_text(record).encode("utf-8"))
    digest.update(record.kind.value.encode("utf-8"))
    digest.update(record.scope.value.encode("utf-8"))
    return digest.hexdigest()


def index_entry_is_current(
    entry: MemoryIndexEntry,
    record: MemoryRecord,
    embedding_provider: EmbeddingProvider,
) -> bool:
    """判断索引条目是否仍匹配当前记录和 embedding 模型。"""

    return (
        entry.content_hash == memory_content_hash(record)
        and entry.embedding_provider == embedding_provider.provider_name
        and entry.embedding_model == embedding_provider.model_name
    )


__all__ = [
    "InMemoryVectorIndex",
    "JsonlVectorIndex",
    "MemoryIndexEntry",
    "MemoryVectorIndex",
    "VectorSearchResult",
    "build_memory_index_entry",
    "index_entry_is_current",
    "memory_content_hash",
]
