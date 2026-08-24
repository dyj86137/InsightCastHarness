"""把过滤后的 CandidateItem，转换成系统后续处理的正式访谈记录数据结构 Interview，
目的是因为：同一场访谈可能被多个来源发现，需要合并成一条正式记录 """

from __future__ import annotations

"""
具体流程是：
  1. 读取状态为 CandidateStatus.ACCEPTED 的候选内容
  2. 找到对应的 InterviewDecision
  3. 用 InterviewIndex 判断是否已经存在同一场访谈
  4. 如果没有，调用 build_interview() 创建新的 Interview
  5. 如果有，调用 merge_interview() 合并到已有 Interview
  6. 原候选会被标记为 ARCHIVED 或 DUPLICATE
"""

import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.domain.enums import CandidateStatus, InterviewStatus, TranscriptStatus
from insightcast.domain.models import (
    CandidateItem,
    DomainModel,
    Interview,
    InterviewDecision,
    RunRecord,
    dedupe_non_empty,
    new_id,
    utc_now,
)
from insightcast.sources import normalize_candidate_url
from insightcast.storage.repositories import InsightCastRepositories


class InterviewPromotionRunError(DomainModel):
    """单条候选内容提升失败时记录的错误。"""

    candidate_id: str
    title: str
    error_type: str
    message: str


class InterviewPromotionRunResult(DomainModel):
    """一次 Interview 提升运行的结果。"""

    run_id: str
    status: str
    processed_count: int = 0
    created_count: int = 0
    merged_count: int = 0
    failed_count: int = 0
    candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)
    interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[InterviewPromotionRunError, ...] = Field(default_factory=tuple)


class InterviewPromotionRunner:
    """读取 accepted 候选内容，创建或合并为规范化 Interview。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        fail_fast: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.repositories = repositories
        self.fail_fast = fail_fast
        self.limit = limit

    def run_once(self) -> InterviewPromotionRunResult:
        """执行一次候选内容提升任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "interview_promotion"},
            )
        )
        try:
            result = self._run(run)
        except Exception as exc:
            self.repositories.run_records.finish(
                run.id,
                status="failed",
                error=str(exc),
            )
            raise

        self.repositories.run_records.finish(
            run.id,
            status=result.status,
            error=_error_summary(result.errors),
            discovered_count=result.processed_count,
            accepted_count=result.created_count + result.merged_count,
            pushed_count=0,
            metadata={
                "stage": "interview_promotion",
                "processed_count": result.processed_count,
                "created_count": result.created_count,
                "merged_count": result.merged_count,
                "failed_count": result.failed_count,
                "candidate_ids": list(result.candidate_ids),
                "interview_ids": list(result.interview_ids),
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    def _run(self, run: RunRecord) -> InterviewPromotionRunResult:
        candidates = self._load_candidates()
        index = InterviewIndex(
            interviews=self.repositories.interviews.list(),
            candidates=self.repositories.candidates.list(),
        )
        created_count = 0
        merged_count = 0
        failed_count = 0
        processed_ids: List[str] = []
        interview_ids: List[str] = []
        errors: List[InterviewPromotionRunError] = []

        for candidate in candidates:
            try:
                decision = self._require_decision(candidate)
                matched = index.find(candidate)
                if matched is None:
                    interview = build_interview(candidate, decision)
                    saved = self.repositories.interviews.create(interview)
                    self.repositories.candidates.update(
                        candidate.id,
                        status=CandidateStatus.ARCHIVED,
                    )
                    index.add(saved)
                    created_count += 1
                    interview_ids.append(saved.id)
                else:
                    saved = self.repositories.interviews.save(
                        merge_interview(matched, candidate, decision)
                    )
                    self.repositories.candidates.update(
                        candidate.id,
                        status=CandidateStatus.DUPLICATE,
                    )
                    index.add(saved)
                    merged_count += 1
                    interview_ids.append(saved.id)
                processed_ids.append(candidate.id)
            except Exception as exc:
                failed_count += 1
                errors.append(
                    InterviewPromotionRunError(
                        candidate_id=candidate.id,
                        title=candidate.title,
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                    )
                )
                self.repositories.candidates.update(
                    candidate.id,
                    status=CandidateStatus.FAILED,
                )
                if self.fail_fast:
                    raise

        return InterviewPromotionRunResult(
            run_id=run.id,
            status=_final_status(
                processed_count=len(processed_ids),
                failed_count=failed_count,
                total_count=len(candidates),
            ),
            processed_count=len(processed_ids),
            created_count=created_count,
            merged_count=merged_count,
            failed_count=failed_count,
            candidate_ids=tuple(processed_ids),
            interview_ids=tuple(dedupe_non_empty(interview_ids)),
            errors=tuple(errors),
        )

    def _load_candidates(self) -> Sequence[CandidateItem]:
        candidates = self.repositories.candidates.list_by_status(CandidateStatus.ACCEPTED)
        if self.limit is not None:
            return candidates[: self.limit]
        return candidates

    def _require_decision(self, candidate: CandidateItem) -> InterviewDecision:
        decision = self.repositories.interview_decisions.get(candidate.id)
        if decision is None:
            raise ValueError(f"accepted candidate has no InterviewDecision: {candidate.id}")
        return decision


class InterviewIndex:
    """Interview 的轻量去重索引。"""

    def __init__(
        self,
        *,
        interviews: Iterable[Interview],
        candidates: Iterable[CandidateItem],
    ) -> None:
        self._items: Dict[str, Interview] = {}
        self._candidates = {candidate.id: candidate for candidate in candidates}
        for interview in interviews:
            self.add(interview)

    def add(self, interview: Interview) -> None:
        for key in interview_identity_keys(interview, self._candidates):
            self._items[key] = interview

    def find(self, candidate: CandidateItem) -> Optional[Interview]:
        for key in candidate_interview_keys(candidate):
            interview = self._items.get(key)
            if interview is not None:
                return interview
        return None


def build_interview(candidate: CandidateItem, decision: InterviewDecision) -> Interview:
    """根据候选内容和访谈识别结果创建 Interview。"""

    return Interview(
        id=new_id("interview"),
        title=candidate.title,
        url=candidate.url,
        canonical_url=candidate.canonical_url,
        type=decision.content_type,
        format=candidate.format,
        status=InterviewStatus.NEW,
        source_id=candidate.source_id,
        source_name=candidate.source_name,
        candidate_ids=(candidate.id,),
        person_ids=candidate.detected_person_ids,
        person_names=candidate.detected_person_names,
        published_at=candidate.published_at,
        duration_seconds=candidate.duration_seconds,
        transcript_status=TranscriptStatus.UNAVAILABLE,
        metadata={
            "promoted_from_candidate_id": candidate.id,
            "decision_confidence": decision.confidence,
            "decision_reason": decision.reason,
        },
    )


def merge_interview(
    interview: Interview,
    candidate: CandidateItem,
    decision: InterviewDecision,
) -> Interview:
    """把重复候选内容合并到已有 Interview。"""

    metadata = dict(interview.metadata)
    duplicate_ids = list(metadata.get("duplicate_candidate_ids") or [])
    duplicate_ids.append(candidate.id)
    metadata.update(
        {
            "duplicate_candidate_ids": list(dedupe_non_empty(duplicate_ids)),
            "last_merged_candidate_id": candidate.id,
            "last_decision_confidence": decision.confidence,
            "updated_at_reason": "candidate_duplicate_merge",
        }
    )
    return interview.clone(
        candidate_ids=dedupe_non_empty(list(interview.candidate_ids) + [candidate.id]),
        person_ids=dedupe_non_empty(
            list(interview.person_ids) + list(candidate.detected_person_ids)
        ),
        person_names=dedupe_non_empty(
            list(interview.person_names) + list(candidate.detected_person_names)
        ),
        published_at=interview.published_at or candidate.published_at,
        duration_seconds=interview.duration_seconds or candidate.duration_seconds,
        metadata=metadata,
        updated_at=utc_now(),
    )


def interview_identity_keys(
    interview: Interview,
    candidates_by_id: MappingLikeCandidates,
) -> Tuple[str, ...]:
    """返回已有 Interview 的去重 key。"""

    keys: List[str] = []
    for url in (interview.effective_url, str(interview.url)):
        normalized = normalize_candidate_url(url)
        if normalized:
            keys.append(f"url:{normalized}")
    keys.extend(_title_person_date_keys(interview.title, interview.person_ids, interview.person_names, interview.published_at))
    for candidate_id in interview.candidate_ids:
        candidate = candidates_by_id.get(candidate_id)
        if candidate is not None:
            keys.extend(candidate_interview_keys(candidate))
    return tuple(dedupe_non_empty(keys))


def candidate_interview_keys(candidate: CandidateItem) -> Tuple[str, ...]:
    """返回候选内容用于匹配 Interview 的去重 key。"""

    keys: List[str] = []
    if candidate.platform_item_id:
        keys.append(
            "platform:"
            f"{candidate.source_type.value}:"
            f"{candidate.platform_item_id.strip().lower()}"
        )
    for url in (candidate.effective_url, str(candidate.url)):
        normalized = normalize_candidate_url(url)
        if normalized:
            keys.append(f"url:{normalized}")
    keys.extend(
        _title_person_date_keys(
            candidate.title,
            candidate.detected_person_ids,
            candidate.detected_person_names,
            candidate.published_at,
        )
    )
    return tuple(dedupe_non_empty(keys))


def _title_person_date_keys(
    title: str,
    person_ids: Iterable[str],
    person_names: Iterable[str],
    published_at: Optional[datetime],
) -> Tuple[str, ...]:
    normalized_title = normalize_title(title)
    person_key = normalize_person_key(person_ids, person_names)
    if not normalized_title or not person_key or published_at is None:
        return ()
    return (f"title_person_date:{normalized_title}:{person_key}:{published_at.date().isoformat()}",)


def normalize_title(title: str) -> str:
    """归一化标题，用于轻量精确去重。"""

    text = re.sub(r"\s+", " ", title.strip().lower())
    text = re.sub(r"[^\w\u4e00-\u9fff ]+", "", text)
    return text.strip()


def normalize_person_key(
    person_ids: Iterable[str],
    person_names: Iterable[str],
) -> str:
    """归一化人物信息，用于标题级去重。"""

    values = list(person_ids) or list(person_names)
    normalized = sorted(value.strip().lower() for value in values if value.strip())
    return "|".join(normalized)


def _final_status(*, processed_count: int, failed_count: int, total_count: int) -> str:
    if failed_count and processed_count == 0:
        return "failed"
    if failed_count:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[InterviewPromotionRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} interview promotion errors"


MappingLikeCandidates = Dict[str, CandidateItem]


__all__ = [
    "InterviewIndex",
    "InterviewPromotionRunError",
    "InterviewPromotionRunResult",
    "InterviewPromotionRunner",
    "build_interview",
    "candidate_interview_keys",
    "interview_identity_keys",
    "merge_interview",
    "normalize_person_key",
    "normalize_title",
]
