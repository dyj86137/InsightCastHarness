"""BM25 稀疏关键词召回。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pydantic import Field

from infra.serialization import SerializableModel
from memory.base import MemoryRecord
from memory.text import record_text, tokenize


class BM25SearchResult(SerializableModel):
    """BM25 单条召回结果。"""

    memory_id: str
    score: float
    rank: int
    matched_terms: Tuple[str, ...] = Field(default_factory=tuple)

    model_config = SerializableModel.config(frozen=True)


class BM25MemoryIndex:
    """面向 MemoryRecord 的轻量 BM25 索引。"""

    def __init__(
        self,
        records: Optional[Iterable[MemoryRecord]] = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._records: Dict[str, MemoryRecord] = {}
        self._term_freqs: Dict[str, Counter] = {}
        self._doc_freqs: Counter = Counter()
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_len = 0.0
        if records:
            self.build(records)

    def build(self, records: Iterable[MemoryRecord]) -> None:
        """从记录集合重建索引。"""

        self._records = {}
        self._term_freqs = {}
        self._doc_freqs = Counter()
        self._doc_lengths = {}
        for record in records:
            self._add_unchecked(record)
        self._refresh_average_length()

    def upsert(self, record: MemoryRecord) -> None:
        """新增或更新一条记录的索引。"""

        if record.id in self._records:
            self.delete(record.id)
        self._add_unchecked(record)
        self._refresh_average_length()

    def delete(self, memory_id: str) -> None:
        """删除一条记录的索引。"""

        term_freq = self._term_freqs.pop(memory_id, None)
        if term_freq is None:
            return
        for term in term_freq:
            self._doc_freqs[term] -= 1
            if self._doc_freqs[term] <= 0:
                del self._doc_freqs[term]
        self._records.pop(memory_id, None)
        self._doc_lengths.pop(memory_id, None)
        self._refresh_average_length()

    def search(
        self,
        query_text: str,
        *,
        top_k: int,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[BM25SearchResult]:
        """返回 BM25 top_k 召回结果。"""

        query_terms = tokenize(query_text)
        if not query_terms or top_k <= 0 or not self._records:
            return []

        candidate_set = set(candidate_ids) if candidate_ids is not None else set(self._records)
        query_counts = Counter(query_terms)
        scored: List[Tuple[str, float, Tuple[str, ...]]] = []
        for memory_id in candidate_set:
            if memory_id not in self._term_freqs:
                continue
            score, matched_terms = self._score_document(memory_id, query_counts)
            if score <= 0:
                continue
            scored.append((memory_id, score, tuple(sorted(matched_terms))))

        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return [
            BM25SearchResult(
                memory_id=memory_id,
                score=score,
                rank=rank,
                matched_terms=matched_terms,
            )
            for rank, (memory_id, score, matched_terms) in enumerate(scored[:top_k], start=1)
        ]

    def _add_unchecked(self, record: MemoryRecord) -> None:
        tokens = tokenize(record_text(record))
        term_freq = Counter(tokens)
        self._records[record.id] = record
        self._term_freqs[record.id] = term_freq
        self._doc_lengths[record.id] = len(tokens)
        for term in term_freq:
            self._doc_freqs[term] += 1

    def _score_document(
        self,
        memory_id: str,
        query_counts: Counter,
    ) -> Tuple[float, Set[str]]:
        term_freq = self._term_freqs[memory_id]
        doc_len = self._doc_lengths.get(memory_id, 0)
        if doc_len <= 0:
            return 0.0, set()

        score = 0.0
        matched_terms: Set[str] = set()
        for term, query_count in query_counts.items():
            frequency = term_freq.get(term, 0)
            if frequency <= 0:
                continue
            matched_terms.add(term)
            idf = self._idf(term)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1.0)
            )
            score += query_count * idf * frequency * (self.k1 + 1) / denominator
        return score, matched_terms

    def _idf(self, term: str) -> float:
        total_docs = len(self._records)
        doc_freq = self._doc_freqs.get(term, 0)
        if total_docs <= 0 or doc_freq <= 0:
            return 0.0
        return math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def _refresh_average_length(self) -> None:
        if not self._doc_lengths:
            self._avg_doc_len = 0.0
            return
        self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)


__all__ = [
    "BM25MemoryIndex",
    "BM25SearchResult",
]
