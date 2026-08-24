from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from fastapi.testclient import TestClient

from insightcast.api.main import create_app
from insightcast.api.service import PipelineService
from insightcast.api.settings import ApiSettings
from insightcast.domain.enums import (
    BriefSection,
    BriefStatus,
    CandidateStatus,
    ContentFormat,
    Industry,
    InterviewStatus,
    InterviewType,
    SourceType,
    TranscriptSource,
    TranscriptStatus,
)
from insightcast.domain.models import (
    CandidateItem,
    DailyBrief,
    DailyBriefItem,
    Interview,
    InterviewDecision,
    InterviewSummary,
    RunRecord,
    Transcript,
    utc_now,
)
from insightcast.storage.repositories import InsightCastRepositories


class ApiTests(unittest.TestCase):
    def test_health_and_read_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = ApiSettings(
                backend_root=root,
                config_path=root / "config.json",
                storage_backend="sqlite",
                sqlite_path=root / "insightcast.sqlite3",
                brief_output_dir=root / "briefs",
                trace_dir=root / "traces",
                checkpoint_dir=root / "checkpoints",
                error_log_dir=root / "error_logs",
                cors_origins=(
                    "http://localhost:3000",
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                ),
            )
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "ok")
                self.assertTrue(health.json()["database_exists"] is False)

                preflight = client.options(
                    "/health",
                    headers={
                        "Origin": "http://127.0.0.1:5173",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(
                    preflight.headers["access-control-allow-origin"],
                    "http://127.0.0.1:5173",
                )

                interviews = client.get("/api/interviews?limit=10")
                self.assertEqual(interviews.status_code, 200)
                self.assertEqual(interviews.json()["items"], [])

                missing = client.get("/api/runs/run_missing")
                self.assertEqual(missing.status_code, 404)

    def test_person_crud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                created = client.post(
                    "/api/persons",
                    json={
                        "name": "Jensen Huang",
                        "aliases": ["Jen-Hsun Huang"],
                        "companies": ["NVIDIA"],
                        "importance": 0.9,
                        "industries": ["ai"],
                    },
                )
                self.assertEqual(created.status_code, 201)
                person = created.json()
                self.assertTrue(person["id"].startswith("person_"))
                person_id = person["id"]

                detail = client.get(f"/api/persons/{person_id}")
                self.assertEqual(detail.status_code, 200)
                self.assertEqual(detail.json()["name"], "Jensen Huang")

                updated = client.patch(
                    f"/api/persons/{person_id}",
                    json={"importance": 1.0, "enabled": False},
                )
                self.assertEqual(updated.status_code, 200)
                self.assertEqual(updated.json()["importance"], 1.0)
                self.assertFalse(updated.json()["enabled"])

                listing = client.get("/api/persons?enabled=false&limit=10")
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json()["total"], 1)
                self.assertEqual(listing.json()["items"][0]["id"], person_id)

                deleted = client.delete(f"/api/persons/{person_id}")
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(client.get(f"/api/persons/{person_id}").status_code, 404)

    def test_industry_and_source_crud(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _test_settings(Path(tmp))
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                industry_response = client.post(
                    "/api/industries",
                    json={
                        "name": "Artificial Intelligence",
                        "slug": "ai",
                        "keywords": ["AI", "machine learning"],
                    },
                )
                self.assertEqual(industry_response.status_code, 201)
                industry_id = industry_response.json()["id"]

                industry_update = client.patch(
                    f"/api/industries/{industry_id}",
                    json={"description": "AI industry profile"},
                )
                self.assertEqual(industry_update.status_code, 200)
                self.assertEqual(
                    industry_update.json()["description"],
                    "AI industry profile",
                )
                self.assertEqual(
                    client.get("/api/industries?limit=10").json()["total"],
                    1,
                )

                source_response = client.post(
                    "/api/sources",
                    json={
                        "name": "NVIDIA YouTube",
                        "type": "youtube",
                        "url": "https://www.youtube.com/@NVIDIA",
                        "authority": 0.95,
                        "industries": ["ai"],
                    },
                )
                self.assertEqual(source_response.status_code, 201)
                source_id = source_response.json()["id"]

                source_update = client.patch(
                    f"/api/sources/{source_id}",
                    json={"enabled": False},
                )
                self.assertEqual(source_update.status_code, 200)
                self.assertFalse(source_update.json()["enabled"])

                self.assertEqual(
                    client.delete(f"/api/industries/{industry_id}").status_code,
                    204,
                )
                self.assertEqual(
                    client.delete(f"/api/sources/{source_id}").status_code,
                    204,
                )

    def test_candidate_list_detail_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _test_settings(root)
            repositories = InsightCastRepositories(
                backend="sqlite",
                sqlite_path=settings.sqlite_path,
            )
            candidate = CandidateItem(
                id="candidate_api",
                title="Jensen Huang interview",
                url="https://www.youtube.com/watch?v=api-test",
                source_type=SourceType.YOUTUBE,
                format=ContentFormat.VIDEO,
                detected_person_ids=("person_jensen",),
                detected_person_names=("Jensen Huang",),
                status=CandidateStatus.CLASSIFIED,
            )
            repositories.candidates.create(candidate)
            repositories.interview_decisions.create(
                InterviewDecision(
                    candidate_id=candidate.id,
                    is_qualifying_content=True,
                    target_person_present=True,
                    content_type=InterviewType.INTERVIEW,
                    confidence=0.9,
                    reason="authoritative interview",
                )
            )
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                listing = client.get("/api/candidates?status=classified")
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json()["total"], 1)
                self.assertEqual(
                    listing.json()["items"][0]["interview_decision"]["confidence"],
                    0.9,
                )

                detail = client.get("/api/candidates/candidate_api")
                self.assertEqual(detail.status_code, 200)
                self.assertEqual(detail.json()["data"]["id"], "candidate_api")

                reviewed = client.post(
                    "/api/candidates/candidate_api/review",
                    json={"action": "accept", "reason": "manual confirmation"},
                )
                self.assertEqual(reviewed.status_code, 200)
                self.assertEqual(
                    reviewed.json()["data"]["status"],
                    CandidateStatus.ACCEPTED.value,
                )
                self.assertEqual(
                    reviewed.json()["data"]["raw_metadata"]["manual_review"]["action"],
                    "accept",
                )

                duplicate = client.post(
                    "/api/candidates/candidate_api/review",
                    json={"action": "duplicate"},
                )
                self.assertEqual(duplicate.status_code, 422)

    def test_interview_and_brief_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _test_settings(root)
            repositories = InsightCastRepositories(
                backend="sqlite",
                sqlite_path=settings.sqlite_path,
            )
            interview = Interview(
                id="interview_api",
                title="Jensen Huang interview",
                url="https://www.youtube.com/watch?v=interview-api",
                status=InterviewStatus.SUMMARY_READY,
                type=InterviewType.INTERVIEW,
                person_ids=("person_jensen",),
                person_names=("Jensen Huang",),
                transcript_status=TranscriptStatus.READY,
            )
            repositories.interviews.create(interview)
            repositories.interview_summaries.create(
                InterviewSummary(
                    id="summary_api",
                    interview_id=interview.id,
                    summary="A concise summary.",
                    key_points=("Point one",),
                    potential_opportunities=("Opportunity one",),
                )
            )
            repositories.transcripts.create(
                Transcript(
                    id="transcript_api",
                    interview_id=interview.id,
                    status=TranscriptStatus.READY,
                    source=TranscriptSource.AUDIO_TRANSCRIPTION,
                    text="private transcript body",
                )
            )
            brief = DailyBrief(
                id="brief_api",
                brief_date=date(2026, 7, 25),
                title="Daily brief",
                status=BriefStatus.READY,
                items=(
                    DailyBriefItem(
                        interview_id=interview.id,
                        summary_id="summary_api",
                        section=BriefSection.MUST_READ,
                        rank=1,
                        title=interview.title,
                        url=interview.url,
                    ),
                ),
            )
            repositories.daily_briefs.create(brief)
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                interview_response = client.get("/api/interviews/interview_api")
                self.assertEqual(interview_response.status_code, 200)
                interview_data = interview_response.json()["data"]
                self.assertEqual(
                    interview_data["summary"]["summary"],
                    "A concise summary.",
                )
                self.assertNotIn("text", interview_data["transcript"])
                self.assertEqual(interview_data["transcript"]["text_chars"], 23)

                brief_by_id = client.get("/api/briefs/brief_api")
                self.assertEqual(brief_by_id.status_code, 200)
                self.assertEqual(len(brief_by_id.json()["data"]["interviews"]), 1)

                brief_by_date = client.get("/api/briefs/2026-07-25")
                self.assertEqual(brief_by_date.status_code, 200)
                self.assertEqual(
                    brief_by_date.json()["data"]["id"],
                    "brief_api",
                )

                brief_list = client.get("/api/briefs?status=ready")
                self.assertEqual(brief_list.status_code, 200)
                self.assertEqual(brief_list.json()["total"], 1)

    def test_dashboard_and_run_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _test_settings(root)
            repositories = InsightCastRepositories(
                backend="sqlite",
                sqlite_path=settings.sqlite_path,
            )
            now = utc_now()
            for index, status in enumerate(
                (
                    CandidateStatus.DISCOVERED,
                    CandidateStatus.CLASSIFIED,
                    CandidateStatus.ACCEPTED,
                )
            ):
                repositories.candidates.create(
                    CandidateItem(
                        id=f"candidate_dashboard_{index}",
                        title=f"Dashboard candidate {index}",
                        url=f"https://example.com/dashboard/{index}",
                        status=status,
                        created_at=now,
                        updated_at=now,
                    )
                )
            repositories.interviews.create(
                Interview(
                    id="interview_dashboard",
                    title="Dashboard interview",
                    url="https://example.com/interview/dashboard",
                    status=InterviewStatus.PUSHED,
                    pushed_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )
            repositories.run_records.create(
                RunRecord(
                    id="run_dashboard",
                    started_at=now,
                    finished_at=now,
                    status="finished",
                    discovered_count=3,
                    accepted_count=1,
                    pushed_count=1,
                )
            )
            service = PipelineService(settings)
            app = create_app(service)
            with TestClient(app) as client:
                runs = client.get("/api/runs?status=finished")
                self.assertEqual(runs.status_code, 200)
                self.assertEqual(runs.json()["total"], 1)
                self.assertEqual(runs.json()["items"][0]["run_id"], "run_dashboard")

                dashboard = client.get(
                    f"/api/dashboard?dashboard_date={now.date().isoformat()}"
                )
                self.assertEqual(dashboard.status_code, 200)
                self.assertEqual(dashboard.json()["discovered_count"], 3)
                self.assertEqual(dashboard.json()["pushed_count"], 1)
                self.assertEqual(dashboard.json()["pending_review_count"], 2)
                self.assertEqual(
                    dashboard.json()["recent_runs"][0]["run_id"],
                    "run_dashboard",
                )


def _test_settings(root: Path) -> ApiSettings:
    return ApiSettings(
        backend_root=root,
        config_path=root / "config.json",
        storage_backend="sqlite",
        sqlite_path=root / "insightcast.sqlite3",
        brief_output_dir=root / "briefs",
        trace_dir=root / "traces",
        checkpoint_dir=root / "checkpoints",
        error_log_dir=root / "error_logs",
        cors_origins=(
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ),
    )


if __name__ == "__main__":
    unittest.main()
