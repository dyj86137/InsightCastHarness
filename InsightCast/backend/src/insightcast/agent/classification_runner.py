"""用于识别候选内容是否为访谈的业务 Runner。"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.agent.interview_classifier import InterviewClassifier
from insightcast.domain.enums import CandidateStatus
from insightcast.domain.models import CandidateItem, DomainModel, RunRecord
from insightcast.sources import candidate_video_identity_keys
from insightcast.storage.repositories import InsightCastRepositories


DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD = 0.75


class ClassificationRunError(DomainModel):
    """单条候选内容分类失败时记录的错误。"""

    candidate_id: str
    title: str
    error_type: str
    message: str


class ClassificationRunResult(DomainModel):
    """一次分类运行的结果。"""

    run_id: str
    status: str
    processed_count: int = 0
    accepted_count: int = 0
    classified_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0
    duplicate_count: int = 0
    candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)
    duplicate_candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[ClassificationRunError, ...] = Field(default_factory=tuple)


class ClassificationRunner:
    """读取待处理候选内容，执行分类，并持久化判断结果。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        classifier: InterviewClassifier,
        *,
        fail_fast: bool = False,
        limit: Optional[int] = None,
        accept_confidence_threshold: float = DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD,
        deduplicate: bool = True,
    ) -> None:
        self.repositories = repositories
        self.classifier = classifier
        self.fail_fast = fail_fast
        self.limit = limit
        self.accept_confidence_threshold = accept_confidence_threshold
        self.deduplicate = deduplicate

    def run_once(self) -> ClassificationRunResult:
        """同步执行一次分类任务。"""

        return asyncio.run(self.run_once_async())

    async def run_once_async(self) -> ClassificationRunResult:
        """异步执行一次分类任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "classification"},
            )
        )
        try:
            result = await self._run(run)
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
            discovered_count=(
                result.processed_count
                + result.failed_count
                + result.duplicate_count
            ),
            accepted_count=result.accepted_count,
            pushed_count=0,
            metadata={
                "stage": "classification",
                "processed_count": result.processed_count,
                "accepted_count": result.accepted_count,
                "classified_count": result.classified_count,
                "rejected_count": result.rejected_count,
                "failed_count": result.failed_count,
                "duplicate_count": result.duplicate_count,
                "candidate_ids": list(result.candidate_ids),
                "duplicate_candidate_ids": list(result.duplicate_candidate_ids),
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    async def _run(self, run: RunRecord) -> ClassificationRunResult:
        candidates, duplicate_count, duplicate_ids = self._prepare_candidates()
        accepted_count = 0
        classified_count = 0
        rejected_count = 0
        failed_count = 0
        processed_ids: List[str] = []
        errors: List[ClassificationRunError] = []

        for candidate in candidates:
            try:
                decision = await self.classifier.classify_async(candidate)
                self.repositories.interview_decisions.save(decision)
                status = decision_to_candidate_status(
                    decision,
                    accept_confidence_threshold=self.accept_confidence_threshold,
                )
                self.repositories.candidates.update(candidate.id, status=status)
                if status == CandidateStatus.ACCEPTED:
                    accepted_count += 1
                elif status == CandidateStatus.CLASSIFIED:
                    classified_count += 1
                else:
                    rejected_count += 1
                processed_ids.append(candidate.id)
            except Exception as exc:
                failed_count += 1
                error = ClassificationRunError(
                    candidate_id=candidate.id,
                    title=candidate.title,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
                errors.append(error)
                self.repositories.candidates.update(candidate.id, status=CandidateStatus.FAILED)
                if self.fail_fast:
                    raise

        return ClassificationRunResult(
            run_id=run.id,
            status=_final_status(
                processed_count=len(processed_ids),
                failed_count=failed_count,
                total_count=len(candidates),
            ),
            processed_count=len(processed_ids),
            accepted_count=accepted_count,
            classified_count=classified_count,
            rejected_count=rejected_count,
            failed_count=failed_count,
            duplicate_count=duplicate_count,
            candidate_ids=tuple(processed_ids),
            duplicate_candidate_ids=tuple(duplicate_ids),
            errors=tuple(errors),
        )

    def _load_candidates(self) -> Sequence[CandidateItem]:
        return self.repositories.candidates.list_by_status(CandidateStatus.DISCOVERED)

    def _prepare_candidates(
        self,
    ) -> Tuple[Sequence[CandidateItem], int, Tuple[str, ...]]:
        """在分类前合并跨平台原始视频重复项。"""

        candidates = sorted(
            self._load_candidates(),
            key=lambda candidate: (candidate.created_at, candidate.id),
        )
        if not self.deduplicate:
            if self.limit is not None:
                candidates = candidates[: self.limit]
            return candidates, 0, ()

        identity_to_keeper = {}
        unique_candidates = []
        duplicate_ids = []

        for candidate in candidates:
            identities = candidate_video_identity_keys(candidate)
            keeper = next(
                (
                    identity_to_keeper[identity]
                    for identity in identities
                    if identity in identity_to_keeper
                ),
                None,
            )
            if keeper is None:
                unique_candidates.append(candidate)
                for identity in identities:
                    identity_to_keeper[identity] = candidate
                continue

            matched_identity = next(
                identity
                for identity in identities
                if identity_to_keeper.get(identity) is keeper
            )
            metadata = dict(candidate.raw_metadata)
            metadata.update(
                {
                    "insightcast_duplicate_of": keeper.id,
                    "insightcast_duplicate_identity": matched_identity,
                }
            )
            self.repositories.candidates.update(
                candidate.id,
                status=CandidateStatus.DUPLICATE,
                raw_metadata=metadata,
            )
            duplicate_ids.append(candidate.id)
            for identity in identities:
                identity_to_keeper.setdefault(identity, keeper)

        if self.limit is not None:
            unique_candidates = unique_candidates[: self.limit]
        return unique_candidates, len(duplicate_ids), tuple(duplicate_ids)


def decision_to_candidate_status(
    decision,
    *,
    accept_confidence_threshold: float = DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD,
) -> CandidateStatus:
    """把内容分类结果映射为候选内容状态。"""

    if (
        decision.is_qualifying_content
        and decision.target_person_present
        and not decision.is_short_clip_or_commentary
    ):
        if decision.confidence < accept_confidence_threshold:
            return CandidateStatus.CLASSIFIED
        return CandidateStatus.ACCEPTED
    return CandidateStatus.REJECTED


def _final_status(*, processed_count: int, failed_count: int, total_count: int) -> str:
    if failed_count and processed_count == 0:
        return "failed"
    if failed_count:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[ClassificationRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} classification errors"


__all__ = [
    "ClassificationRunError",
    "ClassificationRunResult",
    "ClassificationRunner",
    "DEFAULT_ACCEPT_CONFIDENCE_THRESHOLD",
    "decision_to_candidate_status",
]
