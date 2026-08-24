"""Application service used by the FastAPI routes."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from insightcast.domain.enums import CandidateStatus
from insightcast.domain.models import (
    IndustryProfile,
    Person,
    Source,
    new_id,
    utc_now,
)
from insightcast.run.daily_run import run_daily_pipeline
from insightcast.storage.repositories import InsightCastRepositories

from insightcast.api.schemas import (
    BriefResponse,
    BriefListResponse,
    CandidateDetailResponse,
    CandidateListResponse,
    CandidateReviewRequest,
    DashboardResponse,
    HealthResponse,
    IndustryCreateRequest,
    IndustryListResponse,
    IndustryUpdateRequest,
    InterviewDetailResponse,
    InterviewListResponse,
    PersonCreateRequest,
    PersonListResponse,
    PersonUpdateRequest,
    RunCreateRequest,
    RunListResponse,
    RunStatusResponse,
    SourceCreateRequest,
    SourceListResponse,
    SourceUpdateRequest,
    TraceEventResponse,
    TraceEventsResponse,
)
from insightcast.api.settings import ApiSettings


RealtimePublisher = Callable[[Iterable[Mapping[str, Any]]], None]


@dataclass
class _JobState:
    run_id: str
    status: str = "queued"
    error: Optional[str] = None


class PipelineService:
    """Starts pipelines in a bounded local worker and serves read models."""

    def __init__(
        self,
        settings: Optional[ApiSettings] = None,
        *,
        realtime_publisher: Optional[RealtimePublisher] = None,
    ) -> None:
        self.settings = settings or ApiSettings.from_env()
        self._executor = ThreadPoolExecutor(max_workers=self.settings.max_workers)
        self._jobs: Dict[str, _JobState] = {}
        self._lock = threading.Lock()
        self._realtime_publisher = realtime_publisher

    def set_realtime_publisher(
        self,
        publisher: Optional[RealtimePublisher],
    ) -> None:
        """Attach a process-local realtime publisher after ASGI startup wiring."""

        self._realtime_publisher = publisher

    def start_run(self, request: RunCreateRequest) -> str:
        run_id = new_id("run")
        with self._lock:
            self._jobs[run_id] = _JobState(run_id=run_id)
        self._executor.submit(self._execute_run, run_id, request)
        return run_id

    def list_people(
        self,
        *,
        enabled: Optional[bool],
        offset: int,
        limit: int,
    ) -> PersonListResponse:
        people = self._repositories().people.list()
        if enabled is not None:
            people = [person for person in people if person.enabled is enabled]
        people.sort(key=lambda person: person.updated_at, reverse=True)
        total = len(people)
        page = people[offset : offset + limit]
        return PersonListResponse(
            items=[person.to_dict() for person in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        person = self._repositories().people.get(person_id)
        return person.to_dict() if person is not None else None

    def create_person(self, request: PersonCreateRequest) -> Dict[str, Any]:
        person = Person(**_request_data(request))
        return self._repositories().people.create(person).to_dict()

    def update_person(
        self,
        person_id: str,
        request: PersonUpdateRequest,
    ) -> Optional[Dict[str, Any]]:
        repositories = self._repositories()
        person = repositories.people.get(person_id)
        if person is None:
            return None
        updated = person.clone(**_normalise_updates(_request_data(request, exclude_unset=True)))
        return repositories.people.save(updated).to_dict()

    def delete_person(self, person_id: str) -> bool:
        return self._repositories().people.delete(person_id)

    def list_industries(
        self,
        *,
        enabled: Optional[bool],
        offset: int,
        limit: int,
    ) -> IndustryListResponse:
        industries = self._repositories().industry_profiles.list()
        if enabled is not None:
            industries = [industry for industry in industries if industry.enabled is enabled]
        industries.sort(key=lambda industry: industry.name.lower())
        total = len(industries)
        page = industries[offset : offset + limit]
        return IndustryListResponse(
            items=[industry.to_dict() for industry in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_industry(self, industry_id: str) -> Optional[Dict[str, Any]]:
        industry = self._repositories().industry_profiles.get(industry_id)
        return industry.to_dict() if industry is not None else None

    def create_industry(self, request: IndustryCreateRequest) -> Dict[str, Any]:
        industry = IndustryProfile(**_request_data(request))
        return self._repositories().industry_profiles.create(industry).to_dict()

    def update_industry(
        self,
        industry_id: str,
        request: IndustryUpdateRequest,
    ) -> Optional[Dict[str, Any]]:
        repositories = self._repositories()
        industry = repositories.industry_profiles.get(industry_id)
        if industry is None:
            return None
        updated = industry.clone(
            **_normalise_updates(_request_data(request, exclude_unset=True))
        )
        return repositories.industry_profiles.save(updated).to_dict()

    def delete_industry(self, industry_id: str) -> bool:
        return self._repositories().industry_profiles.delete(industry_id)

    def list_sources(
        self,
        *,
        enabled: Optional[bool],
        offset: int,
        limit: int,
    ) -> SourceListResponse:
        sources = self._repositories().sources.list()
        if enabled is not None:
            sources = [source for source in sources if source.enabled is enabled]
        sources.sort(key=lambda source: source.updated_at, reverse=True)
        total = len(sources)
        page = sources[offset : offset + limit]
        return SourceListResponse(
            items=[source.to_dict() for source in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        source = self._repositories().sources.get(source_id)
        return source.to_dict() if source is not None else None

    def create_source(self, request: SourceCreateRequest) -> Dict[str, Any]:
        source = Source(**_request_data(request))
        return self._repositories().sources.create(source).to_dict()

    def update_source(
        self,
        source_id: str,
        request: SourceUpdateRequest,
    ) -> Optional[Dict[str, Any]]:
        repositories = self._repositories()
        source = repositories.sources.get(source_id)
        if source is None:
            return None
        updated = source.clone(
            **_normalise_updates(_request_data(request, exclude_unset=True))
        )
        return repositories.sources.save(updated).to_dict()

    def delete_source(self, source_id: str) -> bool:
        return self._repositories().sources.delete(source_id)

    def list_candidates(
        self,
        *,
        status: Optional[str],
        source_id: Optional[str],
        person_id: Optional[str],
        offset: int,
        limit: int,
    ) -> CandidateListResponse:
        repositories = self._repositories()
        candidates = repositories.candidates.list()
        if status:
            candidates = [item for item in candidates if item.status.value == status]
        if source_id:
            candidates = [item for item in candidates if item.source_id == source_id]
        if person_id:
            candidates = [
                item
                for item in candidates
                if person_id in item.detected_person_ids
            ]
        candidates.sort(key=lambda item: item.updated_at, reverse=True)
        total = len(candidates)
        page = candidates[offset : offset + limit]
        return CandidateListResponse(
            items=[self._candidate_payload(item, repositories) for item in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_candidate(self, candidate_id: str) -> Optional[CandidateDetailResponse]:
        repositories = self._repositories()
        candidate = repositories.candidates.get(candidate_id)
        if candidate is None:
            return None
        return CandidateDetailResponse(
            data=self._candidate_payload(candidate, repositories)
        )

    def review_candidate(
        self,
        candidate_id: str,
        request: CandidateReviewRequest,
    ) -> Optional[CandidateDetailResponse]:
        repositories = self._repositories()
        candidate = repositories.candidates.get(candidate_id)
        if candidate is None:
            return None
        if request.action == "accept" and repositories.interview_decisions.get(candidate_id) is None:
            raise ValueError(
                "candidate cannot be accepted without an InterviewDecision"
            )
        if request.action == "duplicate":
            duplicate_id = request.duplicate_of_candidate_id
            if not duplicate_id:
                raise ValueError(
                    "duplicate_of_candidate_id is required for duplicate review"
                )
            if duplicate_id == candidate_id:
                raise ValueError("candidate cannot be a duplicate of itself")
            if repositories.candidates.get(duplicate_id) is None:
                raise ValueError("duplicate target candidate not found")

        status = {
            "accept": CandidateStatus.ACCEPTED,
            "reject": CandidateStatus.REJECTED,
            "duplicate": CandidateStatus.DUPLICATE,
            "archive": CandidateStatus.ARCHIVED,
        }[request.action]
        raw_metadata = dict(candidate.raw_metadata)
        raw_metadata["manual_review"] = {
            "action": request.action,
            "reason": request.reason,
            "duplicate_of_candidate_id": request.duplicate_of_candidate_id,
            "reviewed_at": utc_now().isoformat(),
        }
        if request.action == "duplicate":
            raw_metadata["insightcast_duplicate_of"] = request.duplicate_of_candidate_id
        updated = candidate.clone(
            status=status,
            raw_metadata=raw_metadata,
            updated_at=utc_now(),
        )
        saved = repositories.candidates.save(updated)
        return CandidateDetailResponse(
            data=self._candidate_payload(saved, repositories)
        )

    def list_briefs(
        self,
        *,
        status: Optional[str],
        brief_date: Optional[date],
        from_date: Optional[date],
        to_date: Optional[date],
        offset: int,
        limit: int,
    ) -> BriefListResponse:
        briefs = self._repositories().daily_briefs.list()
        if status:
            briefs = [item for item in briefs if item.status.value == status]
        if brief_date is not None:
            briefs = [item for item in briefs if item.brief_date == brief_date]
        if from_date is not None:
            briefs = [item for item in briefs if item.brief_date >= from_date]
        if to_date is not None:
            briefs = [item for item in briefs if item.brief_date <= to_date]
        briefs.sort(key=lambda item: item.brief_date, reverse=True)
        total = len(briefs)
        page = briefs[offset : offset + limit]
        return BriefListResponse(
            items=[self._brief_payload(item) for item in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_brief(self, brief_key: str) -> Optional[BriefResponse]:
        repositories = self._repositories()
        brief = repositories.daily_briefs.get(brief_key)
        if brief is None:
            try:
                brief = repositories.daily_briefs.find_by_date(
                    date.fromisoformat(brief_key)
                )
            except ValueError:
                brief = None
        if brief is None:
            return None

        data = self._brief_payload(brief)
        related_interviews = []
        for item in brief.items:
            interview = repositories.interviews.get(item.interview_id)
            if interview is not None:
                related_interviews.append(
                    self._interview_payload(interview, repositories)
                )
        data["interviews"] = related_interviews
        return BriefResponse(brief_date=brief.brief_date, data=data)

    def _execute_run(self, run_id: str, request: RunCreateRequest) -> None:
        self._set_job_status(run_id, "running")
        try:
            result = run_daily_pipeline(
                config_path=self.settings.config_path,
                storage_backend=self.settings.storage_backend,
                sqlite_path=self.settings.sqlite_path,
                brief_output_dir=self.settings.brief_output_dir,
                trace_dir=self.settings.trace_dir,
                checkpoint_dir=self.settings.checkpoint_dir,
                error_log_dir=self.settings.error_log_dir,
                resume_run_id=request.resume_run_id,
                run_id=run_id,
                transcript_fetch_limit=request.transcript_fetch_limit,
                fail_fast=request.fail_fast,
            )
        except Exception as exc:
            self._set_job_status(run_id, "failed", str(exc))
            return
        self._publish_completed_updates(run_id, result)
        self._set_job_status(run_id, "completed")

    def _publish_completed_updates(self, run_id: str, result: Any) -> None:
        publisher = self._realtime_publisher
        pushed_ids = tuple(getattr(result, "pushed_interview_ids", ()) or ())
        if publisher is None or not pushed_ids:
            return

        repositories = self._repositories()
        brief = (
            repositories.daily_briefs.get(result.brief_id)
            if getattr(result, "brief_id", None)
            else None
        )
        brief_items = {
            item.interview_id: item
            for item in (brief.items if brief is not None else ())
        }
        updates: List[Dict[str, Any]] = []
        for interview_id in pushed_ids:
            interview = repositories.interviews.get(interview_id)
            if interview is None:
                continue
            summary = repositories.interview_summaries.find_by_interview_id(
                interview.id
            )
            brief_item = brief_items.get(interview.id)
            updates.append(
                {
                    "run_id": run_id,
                    "brief_id": getattr(result, "brief_id", None),
                    "brief_date": (
                        brief.brief_date.isoformat() if brief is not None else None
                    ),
                    "interview_id": interview.id,
                    "title": interview.title,
                    "url": interview.effective_url,
                    "section": (
                        brief_item.section.value if brief_item is not None else None
                    ),
                    "rank": brief_item.rank if brief_item is not None else None,
                    "reason": brief_item.reason if brief_item is not None else None,
                    "source_name": interview.source_name,
                    "person_names": list(interview.person_names),
                    "industries": [
                        industry.value for industry in interview.industries
                    ],
                    "summary": summary.summary if summary is not None else None,
                    "key_points": list(summary.key_points) if summary else [],
                    "mentioned_companies": (
                        list(summary.mentioned_companies) if summary else []
                    ),
                    "mentioned_products": (
                        list(summary.mentioned_products) if summary else []
                    ),
                }
            )
        if updates:
            publisher(updates)

    def _set_job_status(
        self,
        run_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is not None:
                job.status = status
                job.error = error

    def get_run(self, run_id: str) -> Optional[RunStatusResponse]:
        repositories = self._repositories()
        record = repositories.run_records.get(run_id)
        if record is not None:
            return _run_status_response(record)

        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            return RunStatusResponse(
                run_id=job.run_id,
                status=job.status,
                error=job.error,
            )

    def list_runs(
        self,
        *,
        status: Optional[str],
        offset: int,
        limit: int,
    ) -> RunListResponse:
        repositories = self._repositories()
        records = repositories.run_records.list()
        responses = [_run_status_response(record) for record in records]
        known_ids = {item.run_id for item in responses}
        with self._lock:
            jobs = list(self._jobs.values())
        responses.extend(
            RunStatusResponse(
                run_id=job.run_id,
                status=job.status,
                error=job.error,
            )
            for job in jobs
            if job.run_id not in known_ids
        )
        if status:
            responses = [item for item in responses if item.status == status]
        responses.sort(
            key=lambda item: item.started_at or datetime.min,
            reverse=True,
        )
        total = len(responses)
        return RunListResponse(
            items=responses[offset : offset + limit],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_dashboard(self, dashboard_date: date) -> DashboardResponse:
        repositories = self._repositories()
        candidates = repositories.candidates.list()
        interviews = repositories.interviews.list()
        discovered_count = sum(
            candidate.created_at.date() == dashboard_date
            for candidate in candidates
        )
        pushed_count = sum(
            interview.pushed_at is not None
            and interview.pushed_at.date() == dashboard_date
            for interview in interviews
        )
        pending_review_count = sum(
            candidate.status
            in {CandidateStatus.DISCOVERED, CandidateStatus.CLASSIFIED}
            for candidate in candidates
        )
        return DashboardResponse(
            dashboard_date=dashboard_date,
            discovered_count=discovered_count,
            pushed_count=pushed_count,
            pending_review_count=pending_review_count,
            recent_runs=self.list_runs(status=None, offset=0, limit=5).items,
        )

    def get_events(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> TraceEventsResponse:
        path = self.settings.trace_dir / f"{run_id}.jsonl"
        items: List[TraceEventResponse] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if index < offset:
                        continue
                    if len(items) >= limit:
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        items.append(TraceEventResponse(offset=index, event=event))
        return TraceEventsResponse(
            run_id=run_id,
            items=items,
            next_offset=(items[-1].offset + 1 if items else offset),
        )

    def list_interviews(
        self,
        *,
        status: Optional[str],
        source_id: Optional[str],
        person_id: Optional[str],
        offset: int,
        limit: int,
    ) -> InterviewListResponse:
        repositories = self._repositories()
        interviews = repositories.interviews.list()
        if status:
            interviews = [item for item in interviews if item.status.value == status]
        if source_id:
            interviews = [item for item in interviews if item.source_id == source_id]
        if person_id:
            interviews = [item for item in interviews if person_id in item.person_ids]
        interviews.sort(
            key=lambda item: item.updated_at,
            reverse=True,
        )
        total = len(interviews)
        page = interviews[offset : offset + limit]
        return InterviewListResponse(
            items=[self._interview_list_payload(item) for item in page],
            total=total,
            offset=offset,
            limit=limit,
        )

    def get_interview(self, interview_id: str) -> Optional[InterviewDetailResponse]:
        repositories = self._repositories()
        interview = repositories.interviews.get(interview_id)
        if interview is None:
            return None
        return InterviewDetailResponse(
            data=self._interview_payload(interview, repositories)
        )

    def _candidate_payload(
        self,
        candidate: Any,
        repositories: InsightCastRepositories,
    ) -> Dict[str, Any]:
        data = candidate.to_dict()
        decision = repositories.interview_decisions.get(candidate.id)
        data["interview_decision"] = decision.to_dict() if decision else None
        return data

    def _interview_list_payload(self, interview: Any) -> Dict[str, Any]:
        data = interview.to_dict()
        data["candidate_count"] = len(interview.candidate_ids)
        data["person_count"] = len(interview.person_ids)
        return data

    def _interview_payload(
        self,
        interview: Any,
        repositories: InsightCastRepositories,
    ) -> Dict[str, Any]:
        data = interview.to_dict()
        summary = repositories.interview_summaries.find_by_interview_id(interview.id)
        transcript = repositories.transcripts.find_by_interview_id(interview.id)
        data["summary"] = summary.to_dict() if summary else None
        data["transcript"] = _transcript_metadata(transcript)
        data["candidates"] = [
            self._candidate_payload(candidate, repositories)
            for candidate_id in interview.candidate_ids
            if (candidate := repositories.candidates.get(candidate_id)) is not None
        ]
        return data

    def _brief_payload(self, brief: Any) -> Dict[str, Any]:
        return brief.to_dict()

    def health(self) -> HealthResponse:
        path = self.settings.sqlite_path
        return HealthResponse(
            status="ok",
            storage_backend=self.settings.storage_backend,
            database_path=str(path),
            database_exists=path.exists(),
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _repositories(self) -> InsightCastRepositories:
        return InsightCastRepositories(
            backend=self.settings.storage_backend,
            sqlite_path=self.settings.sqlite_path,
        )


def _run_status_response(record: Any) -> RunStatusResponse:
    metadata = dict(record.metadata)
    return RunStatusResponse(
        run_id=record.id,
        status=record.status,
        stage=_current_stage(metadata),
        started_at=record.started_at,
        finished_at=record.finished_at,
        discovered_count=record.discovered_count,
        accepted_count=record.accepted_count,
        pushed_count=record.pushed_count,
        error=record.error,
        metadata=metadata,
    )


def _current_stage(metadata: Dict[str, Any]) -> Optional[str]:
    stages = metadata.get("stages")
    if isinstance(stages, list):
        for stage in reversed(stages):
            if not isinstance(stage, dict):
                continue
            if stage.get("status") not in {"finished", "skipped"}:
                return stage.get("name")
        if stages and isinstance(stages[-1], dict):
            return stages[-1].get("name")
    value = metadata.get("stage")
    return str(value) if value else None


def _request_data(request: Any, *, exclude_unset: bool = False) -> Dict[str, Any]:
    model_dump = getattr(request, "model_dump", None)
    if model_dump is not None:
        return model_dump(exclude_unset=exclude_unset)
    return request.dict(exclude_unset=exclude_unset)


def _normalise_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    for field in (
        "aliases",
        "companies",
        "industries",
        "keywords",
        "languages",
    ):
        if field in updates and updates[field] is None:
            updates[field] = []
    if "metadata" in updates and updates["metadata"] is None:
        updates["metadata"] = {}
    return updates


def _transcript_metadata(transcript: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Return transcript availability without exposing the transcript body."""

    if transcript is None:
        return None
    data = transcript.to_dict()
    text = data.pop("text", None)
    segments = data.pop("segments", None)
    data["has_text"] = bool(text)
    data["segment_count"] = len(segments or [])
    data["text_chars"] = len(text or "")
    return data
