"""Transcript 获取阶段的业务 Runner。"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.domain.enums import InterviewStatus, TranscriptStatus
from insightcast.domain.models import (
    CandidateItem,
    DomainModel,
    Interview,
    RunRecord,
    Transcript,
    utc_now,
)
from insightcast.storage.repositories import InsightCastRepositories
from insightcast.transcript import CandidateMetadataTranscriptProvider, TranscriptProvider


class TranscriptFetchRunError(DomainModel):
    """单条 Interview 获取 transcript 失败时记录的错误。"""

    interview_id: str
    title: str
    error_type: str
    message: str


class TranscriptFetchRunResult(DomainModel):
    """一次 transcript 获取运行的结果。"""

    run_id: str
    status: str
    processed_count: int = 0
    ready_count: int = 0
    partial_count: int = 0
    unavailable_count: int = 0
    failed_count: int = 0
    interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    transcript_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[TranscriptFetchRunError, ...] = Field(default_factory=tuple)


class TranscriptFetchRunner:
    """读取 Interview，获取可用文本，并保存 Transcript。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        provider: Optional[TranscriptProvider] = None,
        *,
        fail_fast: bool = False,
        limit: Optional[int] = None,
        interview_ids: Optional[Sequence[str]] = None,
        transcript_statuses: Optional[Sequence[TranscriptStatus]] = None,
        progress_callback: Optional[Callable[[str, int, int, Interview], None]] = None,
    ) -> None:
        self.repositories = repositories
        self.provider = provider or CandidateMetadataTranscriptProvider()
        self.fail_fast = fail_fast
        if limit is not None and limit <= 0:
            raise ValueError("transcript_fetch.limit must be positive.")
        self.limit = limit
        self.interview_ids = set(interview_ids or ())
        self.transcript_statuses = tuple(transcript_statuses or ())
        self.progress_callback = progress_callback

    def run_once(self) -> TranscriptFetchRunResult:
        """执行一次 transcript 获取任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "transcript_fetch"},
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
            accepted_count=result.ready_count + result.partial_count,
            pushed_count=0,
            metadata={
                "stage": "transcript_fetch",
                "processed_count": result.processed_count,
                "ready_count": result.ready_count,
                "partial_count": result.partial_count,
                "unavailable_count": result.unavailable_count,
                "failed_count": result.failed_count,
                "interview_ids": list(result.interview_ids),
                "transcript_ids": list(result.transcript_ids),
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    def _run(self, run: RunRecord) -> TranscriptFetchRunResult:
        interviews = self._load_interviews()
        candidates_by_id = {
            candidate.id: candidate for candidate in self.repositories.candidates.list()
        }
        ready_count = 0
        partial_count = 0
        unavailable_count = 0
        failed_count = 0
        interview_ids: List[str] = []
        transcript_ids: List[str] = []
        errors: List[TranscriptFetchRunError] = []

        total_count = len(interviews)
        for position, interview in enumerate(interviews, start=1):
            self._report_progress("started", position, total_count, interview)
            try:
                candidates = candidates_for_interview(interview, candidates_by_id)
                outcome = self.provider.fetch(interview, candidates)
                transcript = self._save_transcript(outcome.transcript)
                self._update_interview(interview, transcript)
                if transcript.status == TranscriptStatus.READY:
                    ready_count += 1
                elif transcript.status == TranscriptStatus.PARTIAL:
                    partial_count += 1
                else:
                    unavailable_count += 1
                interview_ids.append(interview.id)
                transcript_ids.append(transcript.id)
                self._report_progress("finished", position, total_count, interview)
            except Exception as exc:
                failed_count += 1
                errors.append(
                    TranscriptFetchRunError(
                        interview_id=interview.id,
                        title=interview.title,
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                    )
                )
                self.repositories.interviews.update(
                    interview.id,
                    transcript_status=TranscriptStatus.FAILED,
                    status=InterviewStatus.FAILED,
                )
                self._report_progress("failed", position, total_count, interview)
                if self.fail_fast:
                    raise

        return TranscriptFetchRunResult(
            run_id=run.id,
            status=_final_status(
                processed_count=len(interview_ids),
                failed_count=failed_count,
                total_count=len(interviews),
            ),
            processed_count=len(interview_ids),
            ready_count=ready_count,
            partial_count=partial_count,
            unavailable_count=unavailable_count,
            failed_count=failed_count,
            interview_ids=tuple(interview_ids),
            transcript_ids=tuple(transcript_ids),
            errors=tuple(errors),
        )

    def _report_progress(
        self,
        event: str,
        position: int,
        total: int,
        interview: Interview,
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, position, total, interview)

    def _load_interviews(self) -> Sequence[Interview]:
        statuses = self.transcript_statuses or (
            TranscriptStatus.UNAVAILABLE,
            TranscriptStatus.PENDING,
            TranscriptStatus.FAILED,
        )
        interviews = [
            interview
            for interview in self.repositories.interviews.list()
            if interview.transcript_status in statuses
            and (
                not self.interview_ids
                or interview.id in self.interview_ids
            )
        ]
        if self.limit is not None:
            return interviews[: self.limit]
        return interviews

    def _save_transcript(self, transcript: Transcript) -> Transcript:
        existing = self.repositories.transcripts.find_by_interview_id(transcript.interview_id)
        if existing is not None:
            transcript = transcript.clone(
                id=existing.id,
                created_at=existing.created_at,
                updated_at=utc_now(),
            )
        return self.repositories.transcripts.save(transcript)

    def _update_interview(self, interview: Interview, transcript: Transcript) -> None:
        status = interview.status
        if transcript.status in (TranscriptStatus.READY, TranscriptStatus.PARTIAL):
            status = InterviewStatus.TRANSCRIPT_READY
        self.repositories.interviews.update(
            interview.id,
            transcript_status=transcript.status,
            status=status,
        )


def candidates_for_interview(
    interview: Interview,
    candidates_by_id: Dict[str, CandidateItem],
) -> Tuple[CandidateItem, ...]:
    """按 Interview.candidate_ids 找到关联候选内容。"""

    return tuple(
        candidate
        for candidate_id in interview.candidate_ids
        for candidate in (candidates_by_id.get(candidate_id),)
        if candidate is not None
    )


def _final_status(*, processed_count: int, failed_count: int, total_count: int) -> str:
    if failed_count and processed_count == 0:
        return "failed"
    if failed_count:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[TranscriptFetchRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} transcript fetch errors"


__all__ = [
    "TranscriptFetchRunError",
    "TranscriptFetchRunResult",
    "TranscriptFetchRunner",
    "candidates_for_interview",
]
