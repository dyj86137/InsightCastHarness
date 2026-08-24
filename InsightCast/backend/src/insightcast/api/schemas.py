"""FastAPI request and response models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from insightcast.domain.enums import (
    BriefStatus,
    CandidateStatus,
    Industry,
    InterviewStatus,
    SourceType,
)


class RunCreateRequest(BaseModel):
    """Parameters accepted when starting a pipeline run."""

    resume_run_id: Optional[str] = None
    transcript_fetch_limit: Optional[int] = Field(default=None, gt=0)
    fail_fast: bool = False


class IndustryCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    slug: Industry = Industry.OTHER
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndustryUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    slug: Optional[Industry] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    enabled: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class IndustryResponse(BaseModel):
    id: str
    name: str
    slug: Industry
    description: Optional[str] = None
    keywords: List[str]
    enabled: bool
    metadata: Dict[str, Any]


class IndustryListResponse(BaseModel):
    items: List[IndustryResponse]
    total: int
    offset: int
    limit: int


class PersonCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    display_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    industries: List[Industry] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PersonUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    display_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    companies: Optional[List[str]] = None
    title: Optional[str] = None
    industries: Optional[List[Industry]] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enabled: Optional[bool] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PersonResponse(BaseModel):
    id: str
    name: str
    display_name: Optional[str] = None
    aliases: List[str]
    companies: List[str]
    title: Optional[str] = None
    industries: List[Industry]
    importance: float
    enabled: bool
    notes: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class PersonListResponse(BaseModel):
    items: List[PersonResponse]
    total: int
    offset: int
    limit: int


class SourceCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    type: SourceType = SourceType.OTHER
    url: Optional[HttpUrl] = None
    platform_id: Optional[str] = None
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True
    languages: List[str] = Field(default_factory=list)
    industries: List[Industry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    type: Optional[SourceType] = None
    url: Optional[HttpUrl] = None
    platform_id: Optional[str] = None
    authority: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enabled: Optional[bool] = None
    languages: Optional[List[str]] = None
    industries: Optional[List[Industry]] = None
    metadata: Optional[Dict[str, Any]] = None


class SourceResponse(BaseModel):
    id: str
    name: str
    type: SourceType
    url: Optional[HttpUrl] = None
    platform_id: Optional[str] = None
    authority: float
    enabled: bool
    languages: List[str]
    industries: List[Industry]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SourceListResponse(BaseModel):
    items: List[SourceResponse]
    total: int
    offset: int
    limit: int


class CandidateReviewRequest(BaseModel):
    action: Literal["accept", "reject", "duplicate", "archive"]
    reason: Optional[str] = None
    duplicate_of_candidate_id: Optional[str] = None


class CandidateListResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int


class CandidateDetailResponse(BaseModel):
    data: Dict[str, Any]


class InterviewDetailResponse(BaseModel):
    data: Dict[str, Any]


class BriefListResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int


class RunAcceptedResponse(BaseModel):
    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    stage: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    discovered_count: int = 0
    accepted_count: int = 0
    pushed_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    items: List[RunStatusResponse]
    total: int
    offset: int
    limit: int


class DashboardResponse(BaseModel):
    dashboard_date: date
    discovered_count: int
    pushed_count: int
    pending_review_count: int
    recent_runs: List[RunStatusResponse]


class TraceEventResponse(BaseModel):
    offset: int
    event: Dict[str, Any]


class TraceEventsResponse(BaseModel):
    run_id: str
    items: List[TraceEventResponse]
    next_offset: int


class BriefResponse(BaseModel):
    brief_date: date
    data: Dict[str, Any]


class InterviewListResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int


class HealthResponse(BaseModel):
    status: str
    storage_backend: str
    database_path: str
    database_exists: bool
