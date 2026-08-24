"""基于 InsightCast 仓储的人物历史 Memory 适配器。"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

from memory import (
    MemoryError,
    MemoryInjection,
    MemoryKind,
    MemoryManager,
    MemoryPolicy,
    MemoryQuery,
    MemoryRecord,
    MemoryRetrievalMode,
    MemoryScope,
    MemoryStore,
    build_memory_query,
)

from insightcast.domain.models import Interview, InterviewSummary
from insightcast.storage.repositories import InsightCastRepositories


DEFAULT_PERSON_HISTORY_TOP_K = 3
DEFAULT_PERSON_HISTORY_MAX_TOKENS = 4000


class SummaryMemoryProvider(ABC):
    """为总结生成构建 myHarness memory 注入。"""

    @abstractmethod
    async def build_summary_memory(self, interview: Interview) -> MemoryInjection:
        """返回某次访谈相关的历史 memory。"""


class InsightCastSummaryMemoryStore(MemoryStore):
    """基于已持久化访谈总结的只读 MemoryStore 视图。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        exclude_interview_id: Optional[str] = None,
    ) -> None:
        self.repositories = repositories
        self.exclude_interview_id = exclude_interview_id

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        raise MemoryError("InsightCast summary memory store is read-only.")

    async def get(self, memory_id: str) -> MemoryRecord:
        for record in await self.list_records():
            if record.id == memory_id:
                return record
        raise MemoryError(
            "Memory not found.",
            details={"memory_id": memory_id},
        )

    async def list_records(self) -> List[MemoryRecord]:
        interviews = {
            interview.id: interview
            for interview in self.repositories.interviews.list()
        }
        records: List[MemoryRecord] = []
        for summary in self.repositories.interview_summaries.list():
            if summary.interview_id == self.exclude_interview_id:
                continue
            interview = interviews.get(summary.interview_id)
            if interview is None or not interview.person_names:
                continue
            records.append(summary_to_memory_record(interview, summary))
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        raise MemoryError("InsightCast summary memory store is read-only.")

    async def delete(self, memory_id: str) -> None:
        raise MemoryError("InsightCast summary memory store is read-only.")


class PersonHistoryMemoryProvider(SummaryMemoryProvider):
    """为总结 Prompt 检索同一人物的历史观点 memory。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        top_k: int = DEFAULT_PERSON_HISTORY_TOP_K,
        max_injected_tokens: int = DEFAULT_PERSON_HISTORY_MAX_TOKENS,
        policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.repositories = repositories
        self.top_k = top_k
        self.policy = policy or MemoryPolicy(
            retrieval_mode=MemoryRetrievalMode.WEIGHTED_KEYWORD,
            default_top_k=top_k,
            max_injected_memories=top_k,
            max_injected_tokens=max_injected_tokens,
            min_retrieval_score=0.0,
            enable_dense_recall=False,
        )

    async def build_summary_memory(self, interview: Interview) -> MemoryInjection:
        query = build_person_history_query(interview, top_k=self.top_k)
        if not query.tags:
            return MemoryInjection(
                query=query,
                results=(),
                content="",
                estimated_tokens=0,
                metadata={"reason": "interview_has_no_person_names"},
            )

        store = InsightCastSummaryMemoryStore(
            self.repositories,
            exclude_interview_id=interview.id,
        )
        manager = MemoryManager(
            memory_store=store,
            policy=self.policy,
        )
        injection = await manager.build_injection(
            query,
            touch=False,
            title="人物历史 memory",
        )
        return injection.clone(
            metadata={
                **injection.metadata,
                "provider": "insightcast_person_history",
                "person_names": list(interview.person_names),
            }
        )


def build_person_history_query(
    interview: Interview,
    *,
    top_k: int = DEFAULT_PERSON_HISTORY_TOP_K,
) -> MemoryQuery:
    """为同一人物的历史总结构建 myHarness MemoryQuery。"""

    tags = tuple(person_memory_tag(name) for name in interview.person_names)
    query_text = " ".join(
        part
        for part in (
            interview.title,
            " ".join(interview.person_names),
            " ".join(industry.value for industry in interview.industries),
        )
        if part
    )
    return build_memory_query(
        text=query_text,
        kinds=(MemoryKind.LONG_TERM,),
        tags=tags,
        top_k=top_k,
        min_score=0.0,
    )


def summary_to_memory_record(
    interview: Interview,
    summary: InterviewSummary,
) -> MemoryRecord:
    """将 InsightCast InterviewSummary 转换为 myHarness MemoryRecord。"""

    content = "\n".join(_memory_content_lines(interview, summary))
    return MemoryRecord(
        id=summary_memory_id(summary),
        kind=MemoryKind.LONG_TERM,
        content=content,
        summary=_memory_summary_text(interview, summary),
        scope=MemoryScope.GLOBAL,
        tags=summary_memory_tags(interview, summary),
        importance=_memory_importance(interview),
        confidence=0.85,
        metadata={
            "source": "insightcast_interview_summary",
            "interview_id": interview.id,
            "summary_id": summary.id,
            "title": interview.title,
            "url": str(interview.url),
            "person_names": list(interview.person_names),
            "published_at": (
                interview.published_at.isoformat()
                if interview.published_at
                else None
            ),
        },
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def summary_memory_id(summary: InterviewSummary) -> str:
    return f"insightcast_summary_{summary.id}"


def summary_memory_tags(
    interview: Interview,
    summary: InterviewSummary,
) -> Tuple[str, ...]:
    tags = ["insightcast", "interview_summary"]
    tags.extend(person_memory_tag(name) for name in interview.person_names)
    tags.extend(f"industry:{normalize_memory_tag(industry.value)}" for industry in interview.industries)
    tags.extend(f"company:{normalize_memory_tag(company)}" for company in summary.mentioned_companies)
    return tuple(dedupe_non_empty(tags))


def person_memory_tag(person_name: str) -> str:
    return f"person:{normalize_memory_tag(person_name)}"


def normalize_memory_tag(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    parts: List[str] = []
    previous_separator = False
    for char in text:
        if char.isalnum():
            parts.append(char)
            previous_separator = False
            continue
        if parts and not previous_separator:
            parts.append("_")
            previous_separator = True
    normalized = "".join(parts).strip("_")
    return normalized or "unknown"


def _memory_content_lines(
    interview: Interview,
    summary: InterviewSummary,
) -> Iterable[str]:
    yield f"标题：{interview.title}"
    yield f"人物：{', '.join(interview.person_names)}"
    yield f"总结：{summary.summary}"
    if summary.potential_opportunities:
        yield f"关注机会：{'; '.join(summary.potential_opportunities)}"
    if summary.industry_judgements:
        yield f"行业判断：{'; '.join(summary.industry_judgements)}"
    if summary.novelty_assessment:
        yield f"新颖性判断：{summary.novelty_assessment}"


def _memory_summary_text(interview: Interview, summary: InterviewSummary) -> str:
    parts = [f"{interview.title}: {summary.summary}"]
    if summary.potential_opportunities:
        parts.append(f"关注机会：{'; '.join(summary.potential_opportunities)}")
    if summary.novelty_assessment:
        parts.append(f"新颖性：{summary.novelty_assessment}")
    return " ".join(parts)


def _memory_importance(interview: Interview) -> float:
    if interview.importance_score > 0:
        return max(0.5, min(1.0, interview.importance_score))
    return 0.7


def dedupe_non_empty(values: Iterable[str]) -> Tuple[str, ...]:
    seen: Dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen[text] = None
    return tuple(seen)


__all__ = [
    "DEFAULT_PERSON_HISTORY_MAX_TOKENS",
    "DEFAULT_PERSON_HISTORY_TOP_K",
    "InsightCastSummaryMemoryStore",
    "PersonHistoryMemoryProvider",
    "SummaryMemoryProvider",
    "build_person_history_query",
    "normalize_memory_tag",
    "person_memory_tag",
    "summary_memory_id",
    "summary_memory_tags",
    "summary_to_memory_record",
]
