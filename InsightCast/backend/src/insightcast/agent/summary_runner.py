"""结构化总结阶段的业务 Runner。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.agent.interview_summarizer import InterviewSummarizer
from insightcast.domain.enums import InterviewStatus, TranscriptStatus
from insightcast.domain.models import DomainModel, Interview, RunRecord, utc_now
from insightcast.evidence import (
    append_interview_evidence,
    default_evidence_dir,
    evidence_path,
    initialize_evidence_file,
)
from insightcast.storage.repositories import InsightCastRepositories


class SummaryRunError(DomainModel):
    """单条访谈生成摘要失败时记录的错误。"""

    interview_id: str
    title: str
    error_type: str
    message: str


class SummaryRunResult(DomainModel):
    """一次结构化总结运行的结果。"""

    run_id: str
    status: str
    processed_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    failed_count: int = 0
    interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    summary_ids: Tuple[str, ...] = Field(default_factory=tuple)
    evidence_path: Optional[str] = None
    errors: Tuple[SummaryRunError, ...] = Field(default_factory=tuple)


class SummaryRunner:
    """读取 transcript 可用的 Interview，生成并保存结构化摘要。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        summarizer: InterviewSummarizer,
        *,
        fail_fast: bool = False,
        limit: Optional[int] = None,
        evidence_dir: Optional[Path] = None,
        interview_ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[Callable[[str, int, int, Interview], None]] = None,
    ) -> None:
        self.repositories = repositories
        self.summarizer = summarizer
        self.fail_fast = fail_fast
        self.limit = limit
        self.evidence_dir = evidence_dir or default_evidence_dir(repositories.root_dir)
        self.interview_ids = set(interview_ids or ())
        self.progress_callback = progress_callback

    def run_once(self) -> SummaryRunResult:
        """同步执行一次结构化总结任务。"""

        return asyncio.run(self.run_once_async())

    async def run_once_async(self) -> SummaryRunResult:
        """异步执行一次结构化总结任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "summary"},
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
            discovered_count=result.processed_count,
            accepted_count=result.created_count + result.updated_count,
            pushed_count=0,
            metadata={
                "stage": "summary",
                "processed_count": result.processed_count,
                "created_count": result.created_count,
                "updated_count": result.updated_count,
                "failed_count": result.failed_count,
                "interview_ids": list(result.interview_ids),
                "summary_ids": list(result.summary_ids),
                "evidence_path": result.evidence_path,
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    async def _run(self, run: RunRecord) -> SummaryRunResult:
        interviews = self._load_interviews()
        created_count = 0
        updated_count = 0
        failed_count = 0
        interview_ids: List[str] = []
        summary_ids: List[str] = []
        errors: List[SummaryRunError] = []
        audit_path = initialize_evidence_file(evidence_path(self.evidence_dir, run.id))

        total_count = len(interviews)
        for position, interview in enumerate(interviews, start=1):
            self._report_progress("started", position, total_count, interview)
            try:
                transcript = self.repositories.transcripts.find_by_interview_id(interview.id)
                if transcript is None:
                    raise ValueError(f"interview has no transcript: {interview.id}")
                generated = await self.summarizer.summarize_async(interview, transcript)
                append_interview_evidence(
                    audit_path,
                    run_id=run.id,
                    interview=interview,
                    transcript=transcript,
                    claims=generated.evidence_claims,
                )
                summary = generated.summary
                existing = self.repositories.interview_summaries.find_by_interview_id(
                    interview.id
                )
                if existing is None:
                    saved = self.repositories.interview_summaries.create(summary)
                    created_count += 1
                else:
                    saved = self.repositories.interview_summaries.save(
                        summary.clone(
                            id=existing.id,
                            created_at=existing.created_at,
                            updated_at=utc_now(),
                        )
                    )
                    updated_count += 1
                self.repositories.interviews.update(
                    interview.id,
                    status=InterviewStatus.SUMMARY_READY,
                )
                interview_ids.append(interview.id)
                summary_ids.append(saved.id)
                self._report_progress("finished", position, total_count, interview)
            except Exception as exc:
                failed_count += 1
                errors.append(
                    SummaryRunError(
                        interview_id=interview.id,
                        title=interview.title,
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                    )
                )
                self.repositories.interviews.update(
                    interview.id,
                    status=InterviewStatus.FAILED,
                )
                self._report_progress("failed", position, total_count, interview)
                if self.fail_fast:
                    raise

        return SummaryRunResult(
            run_id=run.id,
            status=_final_status(
                processed_count=len(interview_ids),
                failed_count=failed_count,
                total_count=len(interviews),
            ),
            processed_count=len(interview_ids),
            created_count=created_count,
            updated_count=updated_count,
            failed_count=failed_count,
            interview_ids=tuple(interview_ids),
            summary_ids=tuple(summary_ids),
            evidence_path=str(audit_path),
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
        interviews = [
            interview
            for interview in self.repositories.interviews.list()
            if interview.transcript_status in (TranscriptStatus.READY, TranscriptStatus.PARTIAL)
            and interview.status != InterviewStatus.SUMMARY_READY
            and (
                not self.interview_ids
                or interview.id in self.interview_ids
            )
        ]
        if self.limit is not None:
            return interviews[: self.limit]
        return interviews


def _final_status(*, processed_count: int, failed_count: int, total_count: int) -> str:
    if failed_count and processed_count == 0:
        return "failed"
    if failed_count:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[SummaryRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} summary errors"


__all__ = [
    "SummaryRunError",
    "SummaryRunResult",
    "SummaryRunner",
]
