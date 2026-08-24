"""多路召回结果融合。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from pydantic import Field

from infra.serialization import SerializableModel


class RecallHit(SerializableModel):
    """单路召回命中。"""

    memory_id: str
    source: str
    rank: int
    score: float
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class FusedRecallCandidate(SerializableModel):
    """融合后的候选记忆。"""

    memory_id: str
    score: float
    rank: int
    sources: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


def reciprocal_rank_fusion(
    hit_lists: Iterable[Iterable[RecallHit]],
    *,
    k: int = 60,
    top_k: Optional[int] = None,
    source_weights: Optional[Mapping[str, float]] = None,
) -> List[FusedRecallCandidate]:
    """使用 RRF 融合多路召回结果。"""

    if k <= 0:
        raise ValueError("RRF k must be positive.")

    weights = dict(source_weights or {})
    raw_scores: Dict[str, float] = defaultdict(float)
    source_map: Dict[str, set] = defaultdict(set)
    metadata_map: Dict[str, Dict[str, object]] = defaultdict(dict)

    for hits in hit_lists:
        for hit in hits:
            if hit.rank <= 0:
                continue
            weight = weights.get(hit.source, 1.0)
            contribution = weight / (k + hit.rank)
            raw_scores[hit.memory_id] += contribution
            source_map[hit.memory_id].add(hit.source)
            metadata_map[hit.memory_id][f"{hit.source}_rank"] = hit.rank
            metadata_map[hit.memory_id][f"{hit.source}_score"] = hit.score
            if hit.metadata:
                metadata_map[hit.memory_id][f"{hit.source}_metadata"] = hit.metadata

    if not raw_scores:
        return []

    max_score = max(raw_scores.values()) or 1.0
    sorted_items = sorted(
        raw_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    return [
        FusedRecallCandidate(
            memory_id=memory_id,
            score=raw_score / max_score,
            rank=rank,
            sources=tuple(sorted(source_map[memory_id])),
            metadata={
                "rrf_raw_score": raw_score,
                "rrf_k": k,
                **metadata_map[memory_id],
            },
        )
        for rank, (memory_id, raw_score) in enumerate(sorted_items, start=1)
    ]


__all__ = [
    "FusedRecallCandidate",
    "RecallHit",
    "reciprocal_rank_fusion",
]
