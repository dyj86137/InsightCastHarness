"""InsightCast V1 领域模型。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from uuid import uuid4

from pydantic import Field, HttpUrl, field_validator, model_validator

from infra.serialization import SerializableModel

from insightcast.domain.enums import (
    BriefSection,
    BriefStatus,
    CandidateStatus,
    ContentFormat,
    FeedbackType,
    Industry,
    InterviewStatus,
    InterviewType,
    PushChannel,
    SourceType,
    TranscriptSource,
    TranscriptStatus,
)


def utc_now() -> datetime:
    """返回当前 UTC 时间。"""

    return datetime.utcnow()


def new_id(prefix: str) -> str:
    """创建带稳定前缀的 ID。"""

    return f"{prefix}_{uuid4().hex}"


class DomainModel(SerializableModel):
    """InsightCast 不可变领域模型的基类。"""

    model_config = SerializableModel.config(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        use_enum_values=False,
        arbitrary_types_allowed=True,
    )

    def clone(self, **updates: Any):
        """返回应用更新后的校验副本。"""

        return super().clone(**updates)


class IndustryProfile(DomainModel):
    """需要追踪的行业定义。"""

    id: str = Field(default_factory=lambda: new_id("industry"))
    name: str
    slug: Industry = Industry.OTHER
    description: Optional[str] = None
    keywords: Tuple[str, ...] = Field(default_factory=tuple)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        return non_empty_string(value, field_name="industry.name")

    @field_validator("keywords")
    def validate_keywords(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="industry.keyword")


class Person(DomainModel):
    """InsightCast 追踪的商业领袖或专家。"""

    id: str = Field(default_factory=lambda: new_id("person"))
    name: str
    display_name: Optional[str] = None
    aliases: Tuple[str, ...] = Field(default_factory=tuple)
    companies: Tuple[str, ...] = Field(default_factory=tuple)
    title: Optional[str] = None
    industries: Tuple[Industry, ...] = Field(default_factory=tuple)
    importance: float = 0.5
    enabled: bool = True
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        return non_empty_string(value, field_name="person.name")

    @field_validator("aliases", "companies")
    def validate_text_tuple_fields(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="person.text_tuple")

    @field_validator("importance")
    def validate_importance(cls, value: float) -> float:
        return score_0_1(value, field_name="person.importance")

    @property
    def search_names(self) -> Tuple[str, ...]:
        """返回生成搜索查询时应该使用的全部名称。"""

        values = [self.name]
        values.extend(self.aliases)
        return tuple(dedupe_non_empty(values))


class Source(DomainModel):
    """可以产出候选访谈内容的来源。"""

    id: str = Field(default_factory=lambda: new_id("source"))
    name: str
    type: SourceType
    url: Optional[HttpUrl] = None
    platform_id: Optional[str] = None
    authority: float = 0.5
    enabled: bool = True
    languages: Tuple[str, ...] = Field(default_factory=tuple)
    industries: Tuple[Industry, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        return non_empty_string(value, field_name="source.name")

    @field_validator("platform_id")
    def validate_platform_id(cls, value: Optional[str]) -> Optional[str]:
        return optional_non_empty_string(value, field_name="source.platform_id")

    @field_validator("languages")
    def validate_languages(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="source.language")

    @field_validator("authority")
    def validate_authority(cls, value: float) -> float:
        return score_0_1(value, field_name="source.authority")


class UserInterest(DomainModel):
    """用于排序和过滤访谈的用户偏好。"""

    id: str = Field(default_factory=lambda: new_id("interest"))
    industries: Tuple[Industry, ...] = Field(default_factory=tuple)
    people: Tuple[str, ...] = Field(default_factory=tuple)
    companies: Tuple[str, ...] = Field(default_factory=tuple)
    keywords: Tuple[str, ...] = Field(default_factory=tuple)
    push_channels: Tuple[PushChannel, ...] = (PushChannel.MARKDOWN,)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("people", "companies", "keywords")
    def validate_text_tuple_fields(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="interest.text_tuple")


class SearchQuery(DomainModel):
    """面向人物和来源生成的具体搜索查询。"""

    id: str = Field(default_factory=lambda: new_id("query"))
    text: str
    person_id: Optional[str] = None
    source_id: Optional[str] = None
    source_type: Optional[SourceType] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    def validate_text(cls, value: str) -> str:
        return non_empty_string(value, field_name="query.text")


class CandidateItem(DomainModel):
    """从内容来源发现的原始候选项。"""

    id: str = Field(default_factory=lambda: new_id("candidate"))
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    source_type: SourceType = SourceType.OTHER
    platform_item_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: HttpUrl
    canonical_url: Optional[HttpUrl] = None
    format: ContentFormat = ContentFormat.UNKNOWN
    published_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    detected_person_ids: Tuple[str, ...] = Field(default_factory=tuple)
    detected_person_names: Tuple[str, ...] = Field(default_factory=tuple)
    query_id: Optional[str] = None
    query_text: Optional[str] = None
    status: CandidateStatus = CandidateStatus.DISCOVERED
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    def validate_title(cls, value: str) -> str:
        return non_empty_string(value, field_name="candidate.title")

    @field_validator("duration_seconds")
    def validate_duration(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("candidate.duration_seconds cannot be negative")
        return value

    @property
    def effective_url(self) -> str:
        """优先返回 canonical_url，否则返回 url。"""

        return str(self.canonical_url or self.url)


class InterviewDecision(DomainModel):
    """LLM 或规则判断候选项是否属于可收录内容的结果。"""

    candidate_id: str
    is_qualifying_content: bool
    target_person_present: bool
    content_type: InterviewType = InterviewType.UNKNOWN
    is_short_clip_or_commentary: bool = False
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_decision_fields(cls, value: Any) -> Any:
        """读取旧版分类结果时迁移已删除的来源判断字段。"""

        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if "is_qualifying_content" not in payload:
            payload["is_qualifying_content"] = bool(
                payload.get("is_interview", False)
            )
        payload.pop("is_interview", None)
        payload.pop("is_original_or_authoritative_source", None)
        return payload

    @field_validator("candidate_id")
    def validate_candidate_id(cls, value: str) -> str:
        return non_empty_string(value, field_name="decision.candidate_id")

    @field_validator("confidence")
    def validate_confidence(cls, value: float) -> float:
        return score_0_1(value, field_name="decision.confidence")


class Transcript(DomainModel):
    """规范化访谈可用的转录文本或来源文本。"""

    id: str = Field(default_factory=lambda: new_id("transcript"))
    interview_id: str
    status: TranscriptStatus = TranscriptStatus.UNAVAILABLE
    source: TranscriptSource = TranscriptSource.UNKNOWN
    text: Optional[str] = None
    language: Optional[str] = None
    segments: Tuple[str, ...] = Field(default_factory=tuple)
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("interview_id")
    def validate_interview_id(cls, value: str) -> str:
        return non_empty_string(value, field_name="transcript.interview_id")

    @field_validator("segments")
    def validate_segments(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="transcript.segment")

    @model_validator(mode="after")
    def validate_text_for_ready_status(self) -> "Transcript":
        if self.status == TranscriptStatus.READY and not self.text and not self.segments:
            raise ValueError("ready transcript requires text or segments")
        return self


class Interview(DomainModel):
    """去重后的规范化访谈记录。"""

    id: str = Field(default_factory=lambda: new_id("interview"))
    title: str
    url: HttpUrl
    canonical_url: Optional[HttpUrl] = None
    type: InterviewType = InterviewType.UNKNOWN
    format: ContentFormat = ContentFormat.UNKNOWN
    status: InterviewStatus = InterviewStatus.NEW
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)
    person_ids: Tuple[str, ...] = Field(default_factory=tuple)
    person_names: Tuple[str, ...] = Field(default_factory=tuple)
    industries: Tuple[Industry, ...] = Field(default_factory=tuple)
    published_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    transcript_status: TranscriptStatus = TranscriptStatus.UNAVAILABLE
    importance_score: float = 0.0
    novelty_score: float = 0.0
    relevance_score: float = 0.0
    pushed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    def validate_title(cls, value: str) -> str:
        return non_empty_string(value, field_name="interview.title")

    @field_validator("importance_score", "novelty_score", "relevance_score")
    def validate_scores(cls, value: float) -> float:
        return score_0_1(value, field_name="interview.score")

    @property
    def effective_url(self) -> str:
        """优先返回 canonical_url，否则返回 url。"""

        return str(self.canonical_url or self.url)


class ExtractedInsight(DomainModel):
    """从访谈中抽取出的单条结构化洞察。"""

    statement: str
    category: str = "general"
    confidence: float = 0.5
    evidence: Optional[str] = None
    mentioned_companies: Tuple[str, ...] = Field(default_factory=tuple)
    mentioned_products: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("statement")
    def validate_statement(cls, value: str) -> str:
        return non_empty_string(value, field_name="insight.statement")

    @field_validator("category")
    def validate_category(cls, value: str) -> str:
        return non_empty_string(value, field_name="insight.category")

    @field_validator("confidence")
    def validate_confidence(cls, value: float) -> float:
        return score_0_1(value, field_name="insight.confidence")


class InterviewSummary(DomainModel):
    """面向商业情报场景的访谈摘要。"""

    id: str = Field(default_factory=lambda: new_id("summary"))
    interview_id: str
    summary: str
    key_points: Tuple[str, ...] = Field(default_factory=tuple)
    potential_opportunities: Tuple[str, ...] = Field(default_factory=tuple)
    industry_judgements: Tuple[str, ...] = Field(default_factory=tuple)
    mentioned_companies: Tuple[str, ...] = Field(default_factory=tuple)
    mentioned_products: Tuple[str, ...] = Field(default_factory=tuple)
    audience: Tuple[str, ...] = Field(default_factory=tuple)
    novelty_assessment: Optional[str] = None
    insights: Tuple[ExtractedInsight, ...] = Field(default_factory=tuple)
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_summary_fields(cls, value: Any) -> Any:
        """读取旧 JSON/SQLite 摘要时迁移已移除的字段。"""

        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        payload.setdefault("summary", payload.get("five_sentence_summary", ""))
        payload.setdefault(
            "potential_opportunities",
            payload.get("strategic_signals", ()),
        )
        payload.pop("five_sentence_summary", None)
        payload.pop("strategic_signals", None)
        payload.pop("potential_impact", None)
        payload.pop("why_it_matters", None)
        payload.pop("notable_quotes", None)
        return payload

    @field_validator("interview_id")
    def validate_interview_id(cls, value: str) -> str:
        return non_empty_string(value, field_name="summary.interview_id")

    @field_validator("summary")
    def validate_summary(cls, value: str) -> str:
        return non_empty_string(value, field_name="summary.summary")

    @field_validator(
        "key_points",
        "potential_opportunities",
        "industry_judgements",
        "mentioned_companies",
        "mentioned_products",
        "audience",
    )
    def validate_text_tuple_fields(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return validate_text_tuple(values, field_name="summary.text_tuple")


class DailyBriefItem(DomainModel):
    """日报中的单条访谈条目。"""

    interview_id: str
    summary_id: Optional[str] = None
    section: BriefSection
    rank: int
    title: str
    url: HttpUrl
    person_names: Tuple[str, ...] = Field(default_factory=tuple)
    industries: Tuple[Industry, ...] = Field(default_factory=tuple)
    score: float = 0.0
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("interview_id", "title")
    def validate_required_text(cls, value: str) -> str:
        return non_empty_string(value, field_name="brief_item.required_text")

    @field_validator("rank")
    def validate_rank(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("brief_item.rank must be positive")
        return value

    @field_validator("score")
    def validate_score(cls, value: float) -> float:
        return score_0_1(value, field_name="brief_item.score")


class DailyBrief(DomainModel):
    """生成后的每日情报简报。"""

    id: str = Field(default_factory=lambda: new_id("brief"))
    brief_date: date
    title: str
    status: BriefStatus = BriefStatus.DRAFT
    items: Tuple[DailyBriefItem, ...] = Field(default_factory=tuple)
    markdown_path: Optional[str] = None
    sent_at: Optional[datetime] = None
    push_channels: Tuple[PushChannel, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    def validate_title(cls, value: str) -> str:
        return non_empty_string(value, field_name="brief.title")


class UserFeedback(DomainModel):
    """用于改进排序和过滤的用户反馈。"""

    id: str = Field(default_factory=lambda: new_id("feedback"))
    type: FeedbackType
    interview_id: Optional[str] = None
    candidate_id: Optional[str] = None
    person_id: Optional[str] = None
    source_id: Optional[str] = None
    note: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_target(self) -> "UserFeedback":
        if not any(
            (
                self.interview_id,
                self.candidate_id,
                self.person_id,
                self.source_id,
            )
        ):
            raise ValueError("feedback requires at least one target id")
        return self


class RunRecord(DomainModel):
    """一次 InsightCast 流水线运行记录。"""

    id: str = Field(default_factory=lambda: new_id("run"))
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None
    status: str = "running"
    discovered_count: int = 0
    accepted_count: int = 0
    pushed_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    def validate_status(cls, value: str) -> str:
        return non_empty_string(value, field_name="run.status")

    @field_validator("discovered_count", "accepted_count", "pushed_count")
    def validate_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("run counts cannot be negative")
        return value


def non_empty_string(value: str, *, field_name: str) -> str:
    """归一化并校验必填的非空字符串。"""

    if value is None:
        raise ValueError(f"{field_name} cannot be empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def validate_text_tuple(values: Iterable[str], *, field_name: str) -> Tuple[str, ...]:
    """逐项校验字符串 tuple，并保持 tuple 输出。"""

    return tuple(non_empty_string(value, field_name=field_name) for value in values)


def optional_non_empty_string(value: Optional[str], *, field_name: str) -> Optional[str]:
    """归一化可选字符串，并拒绝空白值。"""

    if value is None:
        return None
    return non_empty_string(value, field_name=field_name)


def score_0_1(value: float, *, field_name: str) -> float:
    """校验 0 到 1 之间的分数类数值。"""

    if value < 0 or value > 1:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return value


def dedupe_non_empty(values: Iterable[str]) -> Tuple[str, ...]:
    """在保留顺序的同时，对非空字符串去重。"""

    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return tuple(result)


__all__ = [
    "CandidateItem",
    "DailyBrief",
    "DailyBriefItem",
    "DomainModel",
    "ExtractedInsight",
    "IndustryProfile",
    "Interview",
    "InterviewDecision",
    "InterviewSummary",
    "Person",
    "RunRecord",
    "SearchQuery",
    "Source",
    "Transcript",
    "UserFeedback",
    "UserInterest",
    "dedupe_non_empty",
    "new_id",
    "non_empty_string",
    "optional_non_empty_string",
    "score_0_1",
    "utc_now",
]
