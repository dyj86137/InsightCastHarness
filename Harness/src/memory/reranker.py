"""记忆检索重排器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest
from memory.base import MemoryQuery, MemorySearchResult, validate_score


class RerankDecision(SerializableModel):
    """LLM 对单条候选记忆的重排判断。"""

    memory_id: str
    score: float
    reason: str = ""

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        validate_score(self.score, field_name="score", owner_id=self.memory_id)


class RerankResponse(SerializableModel):
    """LLM 重排响应。"""

    results: Tuple[RerankDecision, ...] = Field(default_factory=tuple)

    model_config = SerializableModel.config(frozen=True)


class MemoryReranker(ABC):
    """记忆重排器接口。"""

    @abstractmethod
    async def rerank(
        self,
        query: MemoryQuery,
        results: Iterable[MemorySearchResult],
        *,
        top_k: int,
    ) -> List[MemorySearchResult]:
        """对候选记忆进行重排。"""


class NoopReranker(MemoryReranker):
    """不改变排序的重排器。"""

    async def rerank(
        self,
        query: MemoryQuery,
        results: Iterable[MemorySearchResult],
        *,
        top_k: int,
    ) -> List[MemorySearchResult]:
        del query
        return list(results)[:top_k]


class LLMReranker(MemoryReranker):
    """基于 LLM 结构化输出的记忆重排器。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_candidates: int = 50,
    ) -> None:
        self.llm = llm
        self.max_candidates = max_candidates

    async def rerank(
        self,
        query: MemoryQuery,
        results: Iterable[MemorySearchResult],
        *,
        top_k: int,
    ) -> List[MemorySearchResult]:
        candidates = list(results)[: self.max_candidates]
        if not query.has_text or not candidates:
            return candidates[:top_k]

        request = ChatRequest.create(
            messages=[
                Message.system(
                    "You are a memory retrieval reranker. "
                    "Return JSON only. Score each candidate from 0 to 1 by relevance."
                ),
                Message.user(build_rerank_prompt(query, candidates)),
            ],
            temperature=0,
        )
        response = await self.llm.structured(request, RerankResponse)
        decision_by_id = {decision.memory_id: decision for decision in response.results}

        reranked = []
        for result in candidates:
            decision = decision_by_id.get(result.record.id)
            if decision is None:
                reranked.append(result)
                continue
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "reranker": "llm",
                    "rerank_score": decision.score,
                    "rerank_reason": decision.reason,
                }
            )
            reason = ",".join(part for part in (result.reason, "llm_rerank") if part)
            reranked.append(
                result.clone(
                    score=decision.score,
                    reason=reason,
                    metadata=metadata,
                )
            )

        reranked.sort(key=lambda item: (item.score, item.record.importance, item.record.id), reverse=True)
        return reranked[:top_k]


def build_rerank_prompt(
    query: MemoryQuery,
    results: Iterable[MemorySearchResult],
) -> str:
    """构造重排提示词。"""

    lines = [
        "Query:",
        query.text,
        "",
        "Candidates:",
    ]
    for index, result in enumerate(results, start=1):
        record = result.record
        lines.extend(
            [
                f"{index}. memory_id={record.id}",
                f"kind={record.kind.value} scope={record.scope.value} tags={','.join(record.tags)}",
                f"content={record.injection_text}",
                "",
            ]
        )
    lines.extend(
        [
            "Return JSON in this shape:",
            '{"results":[{"memory_id":"...", "score":0.0, "reason":"..."}]}',
        ]
    )
    return "\n".join(lines)


def apply_score_overrides(
    results: Iterable[MemorySearchResult],
    scores: Dict[str, float],
) -> List[MemorySearchResult]:
    """用外部分数覆盖检索结果，主要用于测试和自定义 reranker。"""

    reranked = []
    for result in results:
        score = scores.get(result.record.id)
        if score is None:
            reranked.append(result)
            continue
        reranked.append(result.clone(score=score))
    reranked.sort(key=lambda item: (item.score, item.record.importance, item.record.id), reverse=True)
    return reranked


__all__ = [
    "LLMReranker",
    "MemoryReranker",
    "NoopReranker",
    "RerankDecision",
    "RerankResponse",
    "apply_score_overrides",
    "build_rerank_prompt",
]
