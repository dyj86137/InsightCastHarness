"""Markdown 日报生成与本地推送阶段 runner。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.agent.brief_writing_agent import BriefWritingAgent
from insightcast.briefs import (
    MarkdownBriefGenerator,
    MarkdownBriefInputItem,
    write_markdown_brief,
)
from insightcast.domain.enums import BriefSection, BriefStatus, InterviewStatus, PushChannel
from insightcast.domain.models import (
    DailyBrief,
    DailyBriefItem,
    DomainModel,
    Interview,
    InterviewSummary,
    RunRecord,
    dedupe_non_empty,
    utc_now,
)
from insightcast.storage.repositories import InsightCastRepositories


MUST_READ_LIMIT = 3
WORTH_WATCHING_LIMIT = 5


class MarkdownBriefRunError(DomainModel):
    """准备或生成 Markdown 日报时记录的错误。"""

    interview_id: Optional[str] = None
    title: Optional[str] = None
    error_type: str
    message: str


class MarkdownBriefRunResult(DomainModel):
    """一次 Markdown 日报运行的结果。"""

    run_id: str
    status: str
    brief_id: Optional[str] = None
    markdown_path: Optional[str] = None
    item_count: int = 0
    must_read_count: int = 0
    worth_watching_count: int = 0
    archived_count: int = 0
    pushed_count: int = 0
    pushed_interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[MarkdownBriefRunError, ...] = Field(default_factory=tuple)


class MarkdownBriefRunner:
    """构建 DailyBrief，调用 LLM 生成 Markdown，并写入本地文件。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        generator: MarkdownBriefGenerator,
        *,
        output_dir: Optional[Path] = None,
        brief_date: Optional[date] = None,
        limit: Optional[int] = None,
        markdown_writer: Optional[Callable[[str, DailyBrief, Optional[Path]], Path]] = None,
        async_markdown_writer: Optional[
            Callable[[str, DailyBrief, Optional[Path]], Awaitable[Path]]
        ] = None,
        brief_agent: Optional[BriefWritingAgent] = None,
    ) -> None:
        self.repositories = repositories
        self.generator = generator
        self.output_dir = output_dir
        self.brief_date = brief_date
        self.limit = limit
        self.markdown_writer = markdown_writer or _write_markdown_brief
        self.async_markdown_writer = async_markdown_writer
        self.brief_agent = brief_agent

    def run_once(self) -> MarkdownBriefRunResult:
        """同步执行一次 Markdown 日报生成。"""

        return asyncio.run(self.run_once_async())

    async def run_once_async(self) -> MarkdownBriefRunResult:
        """异步执行一次 Markdown 日报生成。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "markdown_brief"},
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
            discovered_count=result.item_count,
            accepted_count=result.pushed_count,
            pushed_count=result.pushed_count,
            metadata={
                "stage": "markdown_brief",
                "brief_id": result.brief_id,
                "markdown_path": result.markdown_path,
                "item_count": result.item_count,
                "must_read_count": result.must_read_count,
                "worth_watching_count": result.worth_watching_count,
                "archived_count": result.archived_count,
                "pushed_count": result.pushed_count,
                "pushed_interview_ids": list(result.pushed_interview_ids),
                "interview_ids": list(result.interview_ids),
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    async def _run(self, run: RunRecord) -> MarkdownBriefRunResult:
        brief_date = self.brief_date or utc_now().date()
        interviews = self._load_interviews()
        source_items, errors = self._build_source_items(interviews)
        if not source_items:
            return MarkdownBriefRunResult(
                run_id=run.id,
                status="failed" if errors else "finished",
                errors=tuple(errors),
            )

        brief = build_daily_brief(brief_date, source_items)
        markdown_path = await self._write_markdown(brief, source_items, run, errors)
        saved = self._save_brief(
            brief.clone(
                status=BriefStatus.SENT,
                markdown_path=str(markdown_path),
                sent_at=utc_now(),
            )
        )
        pushed_ids, archived_ids = self._update_interview_statuses(source_items)

        return MarkdownBriefRunResult(
            run_id=run.id,
            status=_final_status(errors=tuple(errors)),
            brief_id=saved.id,
            markdown_path=str(markdown_path),
            item_count=len(source_items),
            must_read_count=_count_section(brief, BriefSection.MUST_READ),
            worth_watching_count=_count_section(brief, BriefSection.WORTH_WATCHING),
            archived_count=_count_section(brief, BriefSection.ARCHIVED),
            pushed_count=len(pushed_ids),
            pushed_interview_ids=pushed_ids,
            interview_ids=tuple(dedupe_non_empty(list(pushed_ids) + list(archived_ids))),
            errors=tuple(errors),
        )

    async def _write_markdown(
        self,
        brief: DailyBrief,
        source_items: Sequence[MarkdownBriefInputItem],
        run: RunRecord,
        errors: List[MarkdownBriefRunError],
    ) -> Path:
        if self.brief_agent is not None:
            try:
                agent_result = await self.brief_agent.write_brief_async(
                    brief=brief,
                    items=source_items,
                    output_dir=self.output_dir,
                    run_id=run.id,
                )
                return Path(agent_result.markdown_path)
            except Exception as exc:
                errors.append(
                    MarkdownBriefRunError(
                        error_type="BriefWritingAgentFailed",
                        message=str(exc),
                    )
                )

        markdown = await self.generator.generate_async(brief, source_items)
        if self.async_markdown_writer is not None:
            return await self.async_markdown_writer(
                markdown,
                brief,
                self.output_dir,
            )
        return self.markdown_writer(
            markdown,
            brief,
            self.output_dir,
        )

    def _load_interviews(self) -> Sequence[Interview]:
        interviews = [
            interview
            for interview in self.repositories.interviews.list()
            if interview.status == InterviewStatus.PUSH_READY and interview.pushed_at is None
        ]
        interviews.sort(
            key=lambda interview: (
                interview.importance_score,
                interview.published_at or interview.created_at,
            ),
            reverse=True,
        )
        if self.limit is not None:
            return interviews[: self.limit]
        return interviews

    def _build_source_items(
        self,
        interviews: Sequence[Interview],
    ) -> Tuple[Tuple[MarkdownBriefInputItem, ...], List[MarkdownBriefRunError]]:
        items: List[MarkdownBriefInputItem] = []
        errors: List[MarkdownBriefRunError] = []
        brief_items = build_daily_brief_items(interviews, self.repositories)
        brief_items_by_interview_id = {item.interview_id: item for item in brief_items}

        for interview in interviews:
            summary = self.repositories.interview_summaries.find_by_interview_id(
                interview.id
            )
            if summary is None:
                errors.append(
                    MarkdownBriefRunError(
                        interview_id=interview.id,
                        title=interview.title,
                        error_type="MissingSummary",
                        message=f"访谈缺少总结：{interview.id}",
                    )
                )
                self.repositories.interviews.update(
                    interview.id,
                    status=InterviewStatus.FAILED,
                )
                continue
            items.append(
                MarkdownBriefInputItem(
                    brief_item=brief_items_by_interview_id[interview.id],
                    interview=interview,
                    summary=summary,
                )
            )
        return tuple(items), errors

    def _save_brief(self, brief: DailyBrief) -> DailyBrief:
        existing = self.repositories.daily_briefs.find_by_date(brief.brief_date)
        if existing is not None:
            brief = brief.clone(id=existing.id, created_at=existing.created_at)
            return self.repositories.daily_briefs.save(brief)
        return self.repositories.daily_briefs.create(brief)

    def _update_interview_statuses(
        self,
        source_items: Sequence[MarkdownBriefInputItem],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        pushed_ids: List[str] = []
        archived_ids: List[str] = []
        now = utc_now()
        for source_item in source_items:
            interview = source_item.interview
            section = source_item.brief_item.section
            if section == BriefSection.ARCHIVED:
                self.repositories.interviews.update(
                    interview.id,
                    status=InterviewStatus.ARCHIVED,
                )
                archived_ids.append(interview.id)
                continue
            self.repositories.interviews.update(
                interview.id,
                status=InterviewStatus.PUSHED,
                pushed_at=now,
            )
            pushed_ids.append(interview.id)
        return tuple(pushed_ids), tuple(archived_ids)


def build_daily_brief(
    brief_date: date,
    source_items: Sequence[MarkdownBriefInputItem],
) -> DailyBrief:
    """基于已准备的源条目创建内存中的 DailyBrief。"""

    return DailyBrief(
        brief_date=brief_date,
        title=f"InsightCast Daily Brief - {brief_date.isoformat()}",
        status=BriefStatus.DRAFT,
        items=tuple(item.brief_item for item in source_items),
        push_channels=(PushChannel.MARKDOWN,),
        metadata={
            "generator": "markdown_llm",
            "must_read_limit": MUST_READ_LIMIT,
            "worth_watching_limit": WORTH_WATCHING_LIMIT,
        },
    )


def _write_markdown_brief(
    markdown: str,
    brief: DailyBrief,
    output_dir: Optional[Path],
) -> Path:
    return write_markdown_brief(
        markdown,
        brief=brief,
        output_dir=output_dir,
    )


def build_daily_brief_items(
    interviews: Sequence[Interview],
    repositories: InsightCastRepositories,
) -> Tuple[DailyBriefItem, ...]:
    """为已排序访谈构建分板块的 DailyBriefItem 记录。"""

    section_ranks: Dict[BriefSection, int] = {
        BriefSection.MUST_READ: 0,
        BriefSection.WORTH_WATCHING: 0,
        BriefSection.ARCHIVED: 0,
    }
    items: List[DailyBriefItem] = []
    for index, interview in enumerate(interviews):
        section = section_for_index(index)
        section_ranks[section] += 1
        summary = repositories.interview_summaries.find_by_interview_id(interview.id)
        items.append(
            DailyBriefItem(
                interview_id=interview.id,
                summary_id=summary.id if summary is not None else None,
                section=section,
                rank=section_ranks[section],
                title=interview.title,
                url=interview.url,
                person_names=interview.person_names,
                industries=interview.industries,
                score=interview.importance_score,
                reason=brief_item_reason(interview, summary),
                metadata={
                    "source_name": interview.source_name,
                    "transcript_status": interview.transcript_status.value,
                    "novelty_score": interview.novelty_score,
                    "relevance_score": interview.relevance_score,
                },
            )
        )
    return tuple(items)


def section_for_index(index: int) -> BriefSection:
    """将从 0 开始的排序位置映射到日报板块。"""

    if index < MUST_READ_LIMIT:
        return BriefSection.MUST_READ
    if index < MUST_READ_LIMIT + WORTH_WATCHING_LIMIT:
        return BriefSection.WORTH_WATCHING
    return BriefSection.ARCHIVED


def brief_item_reason(
    interview: Interview,
    summary: Optional[InterviewSummary],
) -> str:
    """生成展示给 Markdown LLM 的简短入选理由。"""

    if summary is not None:
        if summary.potential_opportunities:
            return summary.potential_opportunities[0]
        if summary.novelty_assessment:
            return summary.novelty_assessment
    return (
        f"importance={interview.importance_score:.2f}, "
        f"novelty={interview.novelty_score:.2f}, "
        f"relevance={interview.relevance_score:.2f}"
    )


def _count_section(brief: DailyBrief, section: BriefSection) -> int:
    return len([item for item in brief.items if item.section == section])


def _final_status(*, errors: Tuple[MarkdownBriefRunError, ...]) -> str:
    if errors:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[MarkdownBriefRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} 个 Markdown 日报错误"


__all__ = [
    "MUST_READ_LIMIT",
    "WORTH_WATCHING_LIMIT",
    "MarkdownBriefRunError",
    "MarkdownBriefRunResult",
    "MarkdownBriefRunner",
    "brief_item_reason",
    "build_daily_brief",
    "build_daily_brief_items",
    "section_for_index",
]
