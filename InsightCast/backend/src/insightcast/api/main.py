"""FastAPI entrypoint for the InsightCast backend."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar


def _bootstrap_paths() -> None:
    """Support running Uvicorn directly from the source checkout."""

    current = Path(__file__).resolve()
    backend_src = current.parents[2]
    workspace_root = current.parents[6]
    harness_src = workspace_root / "myHarness" / "myHarness-V2" / "src"
    for path in (backend_src, harness_src):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_paths()

import socketio
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

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
    IndustryResponse,
    IndustryUpdateRequest,
    InterviewDetailResponse,
    InterviewListResponse,
    PersonCreateRequest,
    PersonListResponse,
    PersonResponse,
    PersonUpdateRequest,
    RunAcceptedResponse,
    RunCreateRequest,
    RunListResponse,
    RunStatusResponse,
    SourceCreateRequest,
    SourceListResponse,
    SourceResponse,
    SourceUpdateRequest,
    TraceEventsResponse,
)
from insightcast.api.realtime import RealtimeHub
from insightcast.api.service import PipelineService
from insightcast.api.settings import ApiSettings


MutationResult = TypeVar("MutationResult")


def _run_mutation(operation: Callable[[], MutationResult]) -> MutationResult:
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(
    service: Optional[PipelineService] = None,
    *,
    realtime_hub: Optional[RealtimeHub] = None,
) -> FastAPI:
    pipeline_service = service or PipelineService(ApiSettings.from_env())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if realtime_hub is not None:
            await realtime_hub.start()
        try:
            yield
        finally:
            if realtime_hub is not None:
                await realtime_hub.close()
            pipeline_service.shutdown()

    app = FastAPI(
        title="InsightCast API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.pipeline_service = pipeline_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(pipeline_service.settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return pipeline_service.health()

    @app.post(
        "/api/runs",
        response_model=RunAcceptedResponse,
        status_code=202,
        tags=["runs"],
    )
    async def create_run(request: RunCreateRequest) -> RunAcceptedResponse:
        run_id = pipeline_service.start_run(request)
        return RunAcceptedResponse(run_id=run_id, status="queued")

    @app.get(
        "/api/runs",
        response_model=RunListResponse,
        tags=["runs"],
    )
    async def list_runs(
        status: Optional[str] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> RunListResponse:
        return pipeline_service.list_runs(
            status=status,
            offset=offset,
            limit=limit,
        )

    @app.get(
        "/api/dashboard",
        response_model=DashboardResponse,
        tags=["dashboard"],
    )
    async def dashboard(
        dashboard_date: Optional[date] = Query(default=None),
    ) -> DashboardResponse:
        return pipeline_service.get_dashboard(dashboard_date or date.today())

    @app.get(
        "/api/persons",
        response_model=PersonListResponse,
        tags=["persons"],
    )
    async def list_persons(
        enabled: Optional[bool] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> PersonListResponse:
        return pipeline_service.list_people(
            enabled=enabled,
            offset=offset,
            limit=limit,
        )

    @app.post(
        "/api/persons",
        response_model=PersonResponse,
        status_code=201,
        tags=["persons"],
    )
    async def create_person(request: PersonCreateRequest) -> PersonResponse:
        return _run_mutation(
            lambda: pipeline_service.create_person(request)
        )

    @app.get(
        "/api/persons/{person_id}",
        response_model=PersonResponse,
        tags=["persons"],
    )
    async def get_person(person_id: str) -> PersonResponse:
        result = pipeline_service.get_person(person_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Person not found")
        return result

    @app.patch(
        "/api/persons/{person_id}",
        response_model=PersonResponse,
        tags=["persons"],
    )
    async def update_person(
        person_id: str,
        request: PersonUpdateRequest,
    ) -> PersonResponse:
        result = _run_mutation(
            lambda: pipeline_service.update_person(person_id, request)
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Person not found")
        return result

    @app.delete(
        "/api/persons/{person_id}",
        status_code=204,
        tags=["persons"],
    )
    async def delete_person(person_id: str) -> Response:
        if not pipeline_service.delete_person(person_id):
            raise HTTPException(status_code=404, detail="Person not found")
        return Response(status_code=204)

    @app.get(
        "/api/industries",
        response_model=IndustryListResponse,
        tags=["industries"],
    )
    async def list_industries(
        enabled: Optional[bool] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> IndustryListResponse:
        return pipeline_service.list_industries(
            enabled=enabled,
            offset=offset,
            limit=limit,
        )

    @app.post(
        "/api/industries",
        response_model=IndustryResponse,
        status_code=201,
        tags=["industries"],
    )
    async def create_industry(request: IndustryCreateRequest) -> IndustryResponse:
        return _run_mutation(
            lambda: pipeline_service.create_industry(request)
        )

    @app.get(
        "/api/industries/{industry_id}",
        response_model=IndustryResponse,
        tags=["industries"],
    )
    async def get_industry(industry_id: str) -> IndustryResponse:
        result = pipeline_service.get_industry(industry_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Industry not found")
        return result

    @app.patch(
        "/api/industries/{industry_id}",
        response_model=IndustryResponse,
        tags=["industries"],
    )
    async def update_industry(
        industry_id: str,
        request: IndustryUpdateRequest,
    ) -> IndustryResponse:
        result = _run_mutation(
            lambda: pipeline_service.update_industry(industry_id, request)
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Industry not found")
        return result

    @app.delete(
        "/api/industries/{industry_id}",
        status_code=204,
        tags=["industries"],
    )
    async def delete_industry(industry_id: str) -> Response:
        if not pipeline_service.delete_industry(industry_id):
            raise HTTPException(status_code=404, detail="Industry not found")
        return Response(status_code=204)

    @app.get(
        "/api/sources",
        response_model=SourceListResponse,
        tags=["sources"],
    )
    async def list_sources(
        enabled: Optional[bool] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> SourceListResponse:
        return pipeline_service.list_sources(
            enabled=enabled,
            offset=offset,
            limit=limit,
        )

    @app.post(
        "/api/sources",
        response_model=SourceResponse,
        status_code=201,
        tags=["sources"],
    )
    async def create_source(request: SourceCreateRequest) -> SourceResponse:
        return _run_mutation(
            lambda: pipeline_service.create_source(request)
        )

    @app.get(
        "/api/sources/{source_id}",
        response_model=SourceResponse,
        tags=["sources"],
    )
    async def get_source(source_id: str) -> SourceResponse:
        result = pipeline_service.get_source(source_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return result

    @app.patch(
        "/api/sources/{source_id}",
        response_model=SourceResponse,
        tags=["sources"],
    )
    async def update_source(
        source_id: str,
        request: SourceUpdateRequest,
    ) -> SourceResponse:
        result = _run_mutation(
            lambda: pipeline_service.update_source(source_id, request)
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return result

    @app.delete(
        "/api/sources/{source_id}",
        status_code=204,
        tags=["sources"],
    )
    async def delete_source(source_id: str) -> Response:
        if not pipeline_service.delete_source(source_id):
            raise HTTPException(status_code=404, detail="Source not found")
        return Response(status_code=204)

    @app.get(
        "/api/runs/{run_id}",
        response_model=RunStatusResponse,
        tags=["runs"],
    )
    async def get_run(run_id: str) -> RunStatusResponse:
        result = pipeline_service.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return result

    @app.get(
        "/api/runs/{run_id}/events",
        response_model=TraceEventsResponse,
        tags=["runs"],
    )
    async def get_run_events(
        run_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> TraceEventsResponse:
        if pipeline_service.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return pipeline_service.get_events(run_id, offset=offset, limit=limit)

    @app.get(
        "/api/candidates",
        response_model=CandidateListResponse,
        tags=["candidates"],
    )
    async def list_candidates(
        status: Optional[str] = Query(default=None),
        source_id: Optional[str] = Query(default=None),
        person_id: Optional[str] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> CandidateListResponse:
        return pipeline_service.list_candidates(
            status=status,
            source_id=source_id,
            person_id=person_id,
            offset=offset,
            limit=limit,
        )

    @app.get(
        "/api/candidates/{candidate_id}",
        response_model=CandidateDetailResponse,
        tags=["candidates"],
    )
    async def get_candidate(candidate_id: str) -> CandidateDetailResponse:
        result = pipeline_service.get_candidate(candidate_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return result

    @app.post(
        "/api/candidates/{candidate_id}/review",
        response_model=CandidateDetailResponse,
        tags=["candidates"],
    )
    async def review_candidate(
        candidate_id: str,
        request: CandidateReviewRequest,
    ) -> CandidateDetailResponse:
        result = _run_mutation(
            lambda: pipeline_service.review_candidate(candidate_id, request)
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return result

    @app.get(
        "/api/briefs",
        response_model=BriefListResponse,
        tags=["briefs"],
    )
    async def list_briefs(
        status: Optional[str] = Query(default=None),
        brief_date: Optional[date] = Query(default=None),
        from_date: Optional[date] = Query(default=None),
        to_date: Optional[date] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> BriefListResponse:
        return pipeline_service.list_briefs(
            status=status,
            brief_date=brief_date,
            from_date=from_date,
            to_date=to_date,
            offset=offset,
            limit=limit,
        )

    @app.get(
        "/api/briefs/{brief_key}",
        response_model=BriefResponse,
        tags=["briefs"],
    )
    async def get_brief(brief_key: str) -> BriefResponse:
        result = pipeline_service.get_brief(brief_key)
        if result is None:
            raise HTTPException(status_code=404, detail="Brief not found")
        return result

    @app.get(
        "/api/interviews",
        response_model=InterviewListResponse,
        tags=["interviews"],
    )
    async def list_interviews(
        status: Optional[str] = Query(default=None),
        source_id: Optional[str] = Query(default=None),
        person_id: Optional[str] = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> InterviewListResponse:
        return pipeline_service.list_interviews(
            status=status,
            source_id=source_id,
            person_id=person_id,
            offset=offset,
            limit=limit,
        )

    @app.get(
        "/api/interviews/{interview_id}",
        response_model=InterviewDetailResponse,
        tags=["interviews"],
    )
    async def get_interview(interview_id: str) -> InterviewDetailResponse:
        result = pipeline_service.get_interview(interview_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Interview not found")
        return result

    return app


def create_socketio_app(service: Optional[PipelineService] = None) -> Any:
    """Compose Socket.IO and FastAPI while retaining ``create_app`` for tests."""

    pipeline_service = service or PipelineService(ApiSettings.from_env())
    realtime_hub = RealtimeHub(
        cors_allowed_origins=pipeline_service.settings.cors_origins
    )
    pipeline_service.set_realtime_publisher(realtime_hub.publish_updates)
    fastapi_app = create_app(pipeline_service, realtime_hub=realtime_hub)
    return socketio.ASGIApp(
        realtime_hub.sio,
        other_asgi_app=fastapi_app,
        socketio_path="socket.io",
    )


app = create_socketio_app()
