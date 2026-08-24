"""生产级增强的记忆检索流水线。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from memory.base import (
    MemoryKind,
    MemoryPolicy,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalMode,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
)
from memory.embedding import EmbeddingProvider, HashEmbeddingProvider
from memory.filter import filter_records, record_matches_query
from memory.fusion import FusedRecallCandidate, RecallHit, reciprocal_rank_fusion
from memory.index import (
    InMemoryVectorIndex,
    MemoryVectorIndex,
    build_memory_index_entry,
    index_entry_is_current,
)
from memory.lexical import BM25MemoryIndex
from memory.reranker import MemoryReranker, NoopReranker
from memory.text import record_text, token_set, tokenize


class WeightedKeywordMemoryRetriever:
    """轻量确定性检索器，作为 hybrid 不可用时的 fallback。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy(
            retrieval_mode=MemoryRetrievalMode.WEIGHTED_KEYWORD
        )

    async def retrieve(self, query: MemoryQuery) -> List[MemorySearchResult]:
        records = await self.store.list_records()
        return retrieve_from_records(records, query, policy=self.policy)


class MemoryRetriever:
    """完整记忆检索器。

    流水线：
    1. metadata / scope / user hard filter
    2. BM25 sparse recall
    3. vector dense recall
    4. RRF fusion
    5. recency / importance / scope business weighting
    6. optional rerank
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: Optional[MemoryPolicy] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_index: Optional[MemoryVectorIndex] = None,
        lexical_index: Optional[BM25MemoryIndex] = None,
        reranker: Optional[MemoryReranker] = None,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.vector_index = vector_index or InMemoryVectorIndex()
        self.lexical_index = lexical_index or BM25MemoryIndex()
        self.reranker = reranker or NoopReranker()

    async def retrieve(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """执行完整检索流水线。"""

        if self.policy.retrieval_mode == MemoryRetrievalMode.WEIGHTED_KEYWORD:
            records = await self.store.list_records()
            return retrieve_from_records(records, query, policy=self.policy)

        records = await self.store.list_records()
        effective_query = normalize_query(query, self.policy)
        filtered_records = filter_records(records, effective_query)
        if not filtered_records:
            return []

        await self.sync_indexes(records)
        record_by_id = {record.id: record for record in filtered_records}
        candidate_ids = tuple(record_by_id)
        fused_candidates = await self.recall(effective_query, candidate_ids)
        if not fused_candidates:
            fused_candidates = fallback_candidates(
                filtered_records,
                effective_query,
                top_k=self.policy.fusion_top_k,
            )

        results = self.rank_candidates(
            fused_candidates,
            record_by_id,
            effective_query,
        )
        if self.policy.enable_rerank:
            return await self.reranker.rerank(
                effective_query,
                results[: self.policy.rerank_top_k],
                top_k=effective_query.top_k,
            )
        return results[: effective_query.top_k]

    async def sync_indexes(self, records: Sequence[MemoryRecord]) -> None:
        """让 BM25 和向量索引与 store 中的记录保持一致。"""

        self.lexical_index.build(records)
        if not self.should_use_dense_recall():
            return

        record_ids = {record.id for record in records}
        for entry in self.vector_index.list_entries():
            if entry.memory_id not in record_ids:
                self.vector_index.delete(entry.memory_id)

        for record in records:
            entry = self.vector_index.get(record.id)
            if entry and index_entry_is_current(entry, record, self.embedding_provider):
                continue
            self.vector_index.upsert(
                await build_memory_index_entry(record, self.embedding_provider)
            )

    async def recall(
        self,
        query: MemoryQuery,
        candidate_ids: Sequence[str],
    ) -> List[FusedRecallCandidate]:
        """执行 sparse/dense 召回并融合。"""

        if not query.has_text:
            return []

        hit_lists: List[List[RecallHit]] = []
        if self.should_use_sparse_recall():
            sparse_hits = self.lexical_index.search(
                query.text,
                top_k=self.policy.sparse_recall_k,
                candidate_ids=candidate_ids,
            )
            hit_lists.append(
                [
                    RecallHit(
                        memory_id=hit.memory_id,
                        source="bm25",
                        rank=hit.rank,
                        score=hit.score,
                        metadata={"matched_terms": list(hit.matched_terms)},
                    )
                    for hit in sparse_hits
                ]
            )

        if self.should_use_dense_recall():
            query_vector = await self.embedding_provider.embed_query(query.text)
            dense_hits = self.vector_index.search(
                query_vector,
                top_k=self.policy.dense_recall_k,
                candidate_ids=candidate_ids,
            )
            hit_lists.append(
                [
                    RecallHit(
                        memory_id=hit.memory_id,
                        source="dense",
                        rank=hit.rank,
                        score=hit.score,
                    )
                    for hit in dense_hits
                ]
            )

        return reciprocal_rank_fusion(
            hit_lists,
            k=self.policy.rrf_k,
            top_k=self.policy.fusion_top_k,
        )

    def rank_candidates(
        self,
        candidates: Iterable[FusedRecallCandidate],
        record_by_id: Dict[str, MemoryRecord],
        query: MemoryQuery,
    ) -> List[MemorySearchResult]:
        """融合相关性分数和记忆业务信号，生成最终检索结果。"""

        results: List[MemorySearchResult] = []
        for candidate in candidates:
            record = record_by_id.get(candidate.memory_id)
            if record is None:
                continue
            business_score, signals, matched_tags = memory_business_score(record, query)
            score = clamp_score(
                self.policy.relevance_weight * candidate.score
                + self.policy.business_weight * business_score
            )
            if score < query.min_score:
                continue
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "retriever": "hybrid",
                    "relevance_score": candidate.score,
                    "business_score": business_score,
                    "signals": signals,
                    "sources": list(candidate.sources),
                    "relevance_weight": self.policy.relevance_weight,
                    "business_weight": self.policy.business_weight,
                }
            )
            results.append(
                MemorySearchResult(
                    record=record,
                    score=score,
                    reason=build_reason(signals, matched_tags, sources=candidate.sources),
                    matched_tags=tuple(matched_tags),
                    metadata=metadata,
                )
            )

        return sorted(results, key=search_result_sort_key, reverse=True)

    def should_use_sparse_recall(self) -> bool:
        mode = self.policy.retrieval_mode
        return (
            self.policy.enable_sparse_recall
            and mode in (MemoryRetrievalMode.SPARSE, MemoryRetrievalMode.HYBRID)
        )

    def should_use_dense_recall(self) -> bool:
        mode = self.policy.retrieval_mode
        return (
            self.policy.enable_dense_recall
            and mode in (MemoryRetrievalMode.DENSE, MemoryRetrievalMode.HYBRID)
        )


def retrieve_from_records(
    records: Iterable[MemoryRecord],
    query: MemoryQuery,
    *,
    policy: Optional[MemoryPolicy] = None,
) -> List[MemorySearchResult]:
    """对记忆记录执行轻量过滤、分层评分和排序。"""

    selected_policy = policy or MemoryPolicy()
    effective_query = normalize_query(query, selected_policy)
    results = []
    for record in records:
        if not record_matches_query(record, effective_query):
            continue
        score, reason, matched_tags, metadata = score_record(record, effective_query)
        if score < effective_query.min_score:
            continue
        results.append(
            MemorySearchResult(
                record=record,
                score=score,
                reason=reason,
                matched_tags=tuple(matched_tags),
                metadata=metadata,
            )
        )
    return sorted(results, key=search_result_sort_key, reverse=True)[: effective_query.top_k]


def normalize_query(query: MemoryQuery, policy: MemoryPolicy) -> MemoryQuery:
    """根据策略填充查询默认值。"""

    top_k = query.top_k or policy.default_top_k
    min_score = max(query.min_score, policy.min_retrieval_score)
    if top_k == query.top_k and min_score == query.min_score:
        return query
    return query.clone(top_k=top_k, min_score=min_score)


def score_record(
    record: MemoryRecord,
    query: MemoryQuery,
) -> Tuple[float, str, List[str], Dict[str, object]]:
    """按记忆层级给记录做轻量 fallback 评分。"""

    matched_tags = sorted(set(query.tags).intersection(record.tags))
    signals = {
        "text": token_overlap_score(query.text, record_text(record)) if query.has_text else 0.0,
        "tag": tag_match_score(matched_tags, query.tags),
        "scope": scope_match_score(record, query),
        "importance": record.importance,
        "confidence": record.confidence,
        "recency": recency_score(record),
    }
    weights = weights_for_kind(record.kind)
    weighted = sum(signals[name] * weights[name] for name in weights)
    score = clamp_score(weighted)
    reason = build_reason(signals, matched_tags)
    return score, reason, matched_tags, {
        "retriever": "weighted_keyword",
        "kind": record.kind.value,
        "signals": signals,
        "weights": weights,
    }


def memory_business_score(
    record: MemoryRecord,
    query: MemoryQuery,
) -> Tuple[float, Dict[str, float], List[str]]:
    """计算与记忆系统相关的业务信号分。"""

    matched_tags = sorted(set(query.tags).intersection(record.tags))
    signals = {
        "tag": tag_match_score(matched_tags, query.tags),
        "scope": scope_match_score(record, query),
        "importance": record.importance,
        "confidence": record.confidence,
        "recency": recency_score(record),
    }
    weights = weights_for_kind(record.kind)
    active_weights = {name: weights[name] for name in signals}
    denominator = sum(active_weights.values()) or 1.0
    weighted = sum(signals[name] * active_weights[name] for name in signals)
    return clamp_score(weighted / denominator), signals, matched_tags


def fallback_candidates(
    records: Iterable[MemoryRecord],
    query: MemoryQuery,
    *,
    top_k: int,
) -> List[FusedRecallCandidate]:
    """在没有文本召回结果时，用业务信号生成候选集。"""

    scored = []
    for record in records:
        score, _, _ = memory_business_score(record, query)
        scored.append((record.id, score))
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [
        FusedRecallCandidate(
            memory_id=memory_id,
            score=score,
            rank=rank,
            sources=("business",),
            metadata={"fallback": "business"},
        )
        for rank, (memory_id, score) in enumerate(scored[:top_k], start=1)
    ]


def weights_for_kind(kind: MemoryKind) -> Dict[str, float]:
    """返回不同记忆层的评分权重。"""

    if kind == MemoryKind.SHORT_TERM:
        return {
            "scope": 0.30,
            "recency": 0.25,
            "text": 0.20,
            "tag": 0.10,
            "importance": 0.10,
            "confidence": 0.05,
        }
    if kind == MemoryKind.LONG_TERM:
        return {
            "text": 0.30,
            "importance": 0.25,
            "tag": 0.15,
            "scope": 0.15,
            "confidence": 0.10,
            "recency": 0.05,
        }
    if kind == MemoryKind.SKILL:
        return {
            "tag": 0.30,
            "text": 0.25,
            "importance": 0.20,
            "confidence": 0.10,
            "scope": 0.10,
            "recency": 0.05,
        }
    return {
        "text": 0.25,
        "tag": 0.20,
        "scope": 0.20,
        "importance": 0.15,
        "confidence": 0.10,
        "recency": 0.10,
    }


def token_overlap_score(query_text: str, record_text_value: str) -> float:
    """基于 token overlap 的轻量文本相关性。"""

    query_tokens = token_set(query_text)
    if not query_tokens:
        return 0.0
    record_tokens = token_set(record_text_value)
    if not record_tokens:
        return 0.0
    return len(query_tokens.intersection(record_tokens)) / len(query_tokens)


def tag_match_score(matched_tags: Iterable[str], query_tags: Iterable[str]) -> float:
    """根据标签命中比例评分。"""

    query_tag_tuple = tuple(query_tags)
    if not query_tag_tuple:
        return 0.0
    return len(tuple(matched_tags)) / len(query_tag_tuple)


def scope_match_score(record: MemoryRecord, query: MemoryQuery) -> float:
    """根据作用域匹配情况评分。"""

    if query.run_id and record.run_id == query.run_id:
        return 1.0
    if query.session_id and record.session_id == query.session_id:
        return 0.9
    if query.user_id and record.user_id == query.user_id:
        return 0.75
    if record.scope == MemoryScope.GLOBAL and query.include_global:
        return 0.5
    return 0.0


def recency_score(record: MemoryRecord, *, now: Optional[datetime] = None) -> float:
    """根据更新时间给新近度评分。"""

    selected_now = now or datetime.utcnow()
    reference = record.last_accessed_at or record.updated_at or record.created_at
    age_seconds = max((selected_now - reference).total_seconds(), 0.0)
    age_days = age_seconds / 86400
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        return 0.8
    if age_days <= 30:
        return 0.5
    if age_days <= 180:
        return 0.25
    return 0.1


def build_reason(
    signals: Dict[str, float],
    matched_tags: Iterable[str],
    *,
    sources: Iterable[str] = (),
) -> str:
    """根据有效信号生成可诊断 reason。"""

    reasons = []
    source_tuple = tuple(sources)
    if "bm25" in source_tuple:
        reasons.append("bm25")
    if "dense" in source_tuple:
        reasons.append("dense")
    if signals.get("text", 0) > 0:
        reasons.append("text_overlap")
    if tuple(matched_tags):
        reasons.append("tag_match")
    if signals.get("scope", 0) > 0:
        reasons.append("scope_match")
    if signals.get("importance", 0) >= 0.7:
        reasons.append("important")
    if signals.get("recency", 0) >= 0.8:
        reasons.append("recent")
    return ",".join(reasons) or "low_signal_match"


def search_result_sort_key(result: MemorySearchResult) -> tuple:
    """生成检索结果排序 key。"""

    record = result.record
    accessed_at = record.last_accessed_at or record.updated_at
    return result.score, record.importance, accessed_at, record.created_at, record.id


def clamp_score(score: float) -> float:
    """把分数裁剪到 [0, 1]。"""

    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


__all__ = [
    "MemoryRetriever",
    "WeightedKeywordMemoryRetriever",
    "build_reason",
    "clamp_score",
    "fallback_candidates",
    "memory_business_score",
    "record_matches_query",
    "record_text",
    "recency_score",
    "retrieve_from_records",
    "scope_match_score",
    "score_record",
    "search_result_sort_key",
    "tag_match_score",
    "token_overlap_score",
    "tokenize",
    "weights_for_kind",
]
