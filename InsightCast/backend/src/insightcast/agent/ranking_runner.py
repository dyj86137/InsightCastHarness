"""访谈重要性排序阶段的业务 Runner。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field

from insightcast.domain.enums import InterviewStatus, TranscriptStatus
from insightcast.domain.models import (
    DomainModel,
    Interview,
    InterviewSummary,
    Person,
    RunRecord,
    Source,
    UserInterest,
    score_0_1,
    utc_now,
)
from insightcast.storage.repositories import InsightCastRepositories


SCORE_WEIGHTS = {
    "person_importance": 0.25,
    "source_authority": 0.20,
    "recency": 0.15,
    "duration": 0.10,
    "transcript": 0.10,
    "novelty": 0.10,
    "relevance": 0.10,
}


class InterviewScore(DomainModel):
    """单条访谈的重要性评分结果。"""

    interview_id: str
    importance_score: float
    novelty_score: float
    relevance_score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        score_0_1(self.importance_score, field_name="score.importance_score")
        score_0_1(self.novelty_score, field_name="score.novelty_score")
        score_0_1(self.relevance_score, field_name="score.relevance_score")


class RankingRunError(DomainModel):
    """单条访谈评分失败时记录的错误。"""

    interview_id: str
    title: str
    error_type: str
    message: str


class RankingRunResult(DomainModel):
    """一次重要性排序运行的结果。"""

    run_id: str
    status: str
    processed_count: int = 0
    failed_count: int = 0
    ranked_interview_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[RankingRunError, ...] = Field(default_factory=tuple)


class InterviewRankingRunner:
    """读取摘要已生成的访谈，计算重要性分数并写回 Interview。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        fail_fast: bool = False,
        limit: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.repositories = repositories
        self.fail_fast = fail_fast
        self.limit = limit
        self.now = now or utc_now()

    def run_once(self) -> RankingRunResult:
        """执行一次重要性排序任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "ranking"},
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
            accepted_count=result.processed_count,
            pushed_count=0,
            metadata={
                "stage": "ranking",
                "processed_count": result.processed_count,
                "ranked_interview_ids": list(result.ranked_interview_ids),
                "failed_count": result.failed_count,
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    def _run(self, run: RunRecord) -> RankingRunResult:
        interviews = self._load_interviews()
        people = self.repositories.people.list()
        sources = self.repositories.sources.list()
        interests = self.repositories.user_interests.list()
        processed: List[Tuple[str, float]] = []
        errors: List[RankingRunError] = []

        for interview in interviews:
            try:
                summary = self.repositories.interview_summaries.find_by_interview_id(
                    interview.id
                )
                if summary is None:
                    raise ValueError(f"interview has no summary: {interview.id}")
                score = score_interview(
                    interview,
                    summary,
                    people=people,
                    sources=sources,
                    interests=interests,
                    now=self.now,
                )
                metadata = dict(interview.metadata)
                metadata["score_breakdown"] = score.breakdown
                metadata["ranked_at"] = self.now.isoformat()
                self.repositories.interviews.update(
                    interview.id,
                    importance_score=score.importance_score,
                    novelty_score=score.novelty_score,
                    relevance_score=score.relevance_score,
                    status=InterviewStatus.PUSH_READY,
                    metadata=metadata,
                )
                processed.append((interview.id, score.importance_score))
            except Exception as exc:
                errors.append(
                    RankingRunError(
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
                if self.fail_fast:
                    raise

        ranked = tuple(
            interview_id
            for interview_id, _score in sorted(
                processed,
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return RankingRunResult(
            run_id=run.id,
            status=_final_status(
                processed_count=len(processed),
                failed_count=len(errors),
                total_count=len(interviews),
            ),
            processed_count=len(processed),
            failed_count=len(errors),
            ranked_interview_ids=ranked,
            errors=tuple(errors),
        )

    def _load_interviews(self) -> Sequence[Interview]:
        interviews = [
            interview
            for interview in self.repositories.interviews.list()
            if interview.status == InterviewStatus.SUMMARY_READY
        ]
        if self.limit is not None:
            return interviews[: self.limit]
        return interviews


def score_interview(
    interview: Interview,
    summary: InterviewSummary,
    *,
    people: Sequence[Person],
    sources: Sequence[Source],
    interests: Sequence[UserInterest],
    now: datetime,
) -> InterviewScore:
    """计算单条访谈的重要性、观点新意和相关性分数。"""

    breakdown = {
        "person_importance": person_importance_score(interview, people),
        "source_authority": source_authority_score(interview, sources),
        "recency": recency_score(interview, now),
        "duration": duration_score(interview),
        "transcript": transcript_score(interview),
        "novelty": novelty_score(summary),
        "relevance": relevance_score(interview, summary, interests),
    }
    importance = weighted_score(breakdown, SCORE_WEIGHTS)
    return InterviewScore(
        interview_id=interview.id,
        importance_score=importance,
        novelty_score=breakdown["novelty"],
        relevance_score=breakdown["relevance"],
        breakdown=breakdown,
    )


def person_importance_score(interview: Interview, people: Sequence[Person]) -> float:
    """按匹配到的人物重要性取最高分。"""

    people_by_id = {person.id: person for person in people}
    names = {
        normalize_text(value)
        for value in list(interview.person_names) + list(interview.person_ids)
    }
    scores: List[float] = []
    for person_id in interview.person_ids:
        person = people_by_id.get(person_id)
        if person is not None:
            scores.append(person.importance)
    for person in people:
        candidate_names = {normalize_text(person.id)}
        candidate_names.update(normalize_text(value) for value in person.search_names)
        if names.intersection(candidate_names):
            scores.append(person.importance)
    return max(scores) if scores else 0.5


def source_authority_score(interview: Interview, sources: Sequence[Source]) -> float:
    """按来源权威性评分。"""

    for source in sources:
        if source.id == interview.source_id:
            return source.authority
    return 0.5


def recency_score(interview: Interview, now: datetime) -> float:
    """按发布时间距离当前时间评分。"""

    if interview.published_at is None:
        return 0.4
    age_days = max(0, (now - interview.published_at.replace(tzinfo=None)).days)
    if age_days <= 3:
        return 1.0
    if age_days <= 7:
        return 0.8
    if age_days <= 30:
        return 0.5
    if age_days <= 90:
        return 0.3
    return 0.1


def duration_score(interview: Interview) -> float:
    """按内容时长判断是否更像长访谈。"""

    if interview.duration_seconds is None:
        return 0.4
    if interview.duration_seconds >= 1800:
        return 1.0
    if interview.duration_seconds >= 600:
        return 0.7
    if interview.duration_seconds > 0:
        return 0.3
    return 0.2


def transcript_score(interview: Interview) -> float:
    """按 transcript 可用程度评分。"""

    if interview.transcript_status == TranscriptStatus.READY:
        return 1.0
    if interview.transcript_status == TranscriptStatus.PARTIAL:
        return 0.6
    if interview.transcript_status == TranscriptStatus.PENDING:
        return 0.3
    if interview.transcript_status == TranscriptStatus.FAILED:
        return 0.0
    return 0.2


def novelty_score(summary: InterviewSummary) -> float:
    """根据观点变化描述和洞察数量估算新意。"""

    text = normalize_text(summary.novelty_assessment or "")
    positive_markers = (
        "变化",
        "转向",
        "首次",
        "新增",
        "不同",
        "shift",
        "change",
        "new",
    )
    weak_markers = (
        "暂无",
        "无明显",
        "无法判断",
        "不可判断",
        "no clear",
        "unknown",
    )
    if any(marker in text for marker in positive_markers):
        return 0.85
    if any(marker in text for marker in weak_markers):
        return 0.3
    if summary.insights:
        return 0.6
    return 0.4


def relevance_score(
    interview: Interview,
    summary: InterviewSummary,
    interests: Sequence[UserInterest],
) -> float:
    """按用户兴趣匹配程度评分。"""

    if not interests:
        return 0.5

    scores = [interest_relevance_score(interview, summary, interest) for interest in interests]
    return max(scores) if scores else 0.5


def interest_relevance_score(
    interview: Interview,
    summary: InterviewSummary,
    interest: UserInterest,
) -> float:
    """计算单个用户兴趣与访谈的匹配分。"""

    score = 0.0
    if interest.industries and set(interview.industries).intersection(interest.industries):
        score += 0.3

    interest_people = {normalize_text(value) for value in interest.people}
    interview_people = {normalize_text(value) for value in interview.person_ids}
    interview_people.update(normalize_text(value) for value in interview.person_names)
    if interest_people and interview_people.intersection(interest_people):
        score += 0.3

    interest_companies = {normalize_text(value) for value in interest.companies}
    mentioned_companies = {normalize_text(value) for value in summary.mentioned_companies}
    if interest_companies and mentioned_companies.intersection(interest_companies):
        score += 0.2

    keyword_text = normalize_text(
        " ".join(
            [
                interview.title,
                summary.summary,
                " ".join(summary.key_points),
                " ".join(summary.potential_opportunities),
                " ".join(summary.industry_judgements),
            ]
        )
    )
    keyword_hits = [
        keyword
        for keyword in interest.keywords
        if normalize_text(keyword) and normalize_text(keyword) in keyword_text
    ]
    if keyword_hits:
        score += min(0.2, 0.05 * len(keyword_hits))

    return min(1.0, score if score > 0 else 0.4)


def weighted_score(
    values: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """按权重汇总各维度评分。"""

    total = 0.0
    for key, weight in weights.items():
        total += values.get(key, 0.0) * weight
    return round(max(0.0, min(1.0, total)), 4)


def normalize_text(value: str) -> str:
    """归一化文本以便匹配。"""

    return " ".join(str(value).strip().lower().split())


def _final_status(*, processed_count: int, failed_count: int, total_count: int) -> str:
    if failed_count and processed_count == 0:
        return "failed"
    if failed_count:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[RankingRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} ranking errors"


__all__ = [
    "InterviewRankingRunner",
    "InterviewScore",
    "RankingRunError",
    "RankingRunResult",
    "SCORE_WEIGHTS",
    "duration_score",
    "interest_relevance_score",
    "normalize_text",
    "novelty_score",
    "person_importance_score",
    "recency_score",
    "relevance_score",
    "score_interview",
    "source_authority_score",
    "transcript_score",
    "weighted_score",
]
