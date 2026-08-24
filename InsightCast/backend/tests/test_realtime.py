from __future__ import annotations

import asyncio
import socket
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import socketio
import uvicorn


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.api.realtime import RealtimeHub, keyword_room, normalize_keywords
from insightcast.api.main import create_socketio_app
from insightcast.api.service import PipelineService
from insightcast.api.settings import ApiSettings
from insightcast.domain.enums import BriefSection, BriefStatus, Industry, InterviewStatus
from insightcast.domain.models import (
    DailyBrief,
    DailyBriefItem,
    Interview,
    InterviewSummary,
)
from insightcast.storage.repositories import InsightCastRepositories


class RealtimeHubTests(unittest.TestCase):
    def test_normalize_keywords_deduplicates_and_bounds_input(self) -> None:
        values = ["  AI  Agent ", "ai agent", "具身智能", 3, "", "x" * 81]

        self.assertEqual(normalize_keywords(values), ["ai agent", "具身智能"])

    def test_subscription_replaces_keyword_rooms_and_acknowledges(self) -> None:
        async def run() -> None:
            hub = RealtimeHub(cors_allowed_origins=("http://localhost:5173",))
            hub._subscriptions["sid-1"] = {"old"}
            hub.sio.enter_room = AsyncMock()
            hub.sio.leave_room = AsyncMock()
            hub.sio.emit = AsyncMock()

            handler = hub.sio.handlers["/"]["subscribe_keywords"]
            result = await handler(
                "sid-1",
                {"keywords": ["AI", " 具身智能 ", "ai"]},
            )

            self.assertEqual(result, {"keywords": ["ai", "具身智能"]})
            hub.sio.leave_room.assert_awaited_once_with("sid-1", keyword_room("old"))
            self.assertEqual(hub.sio.enter_room.await_count, 2)
            hub.sio.emit.assert_awaited_once_with(
                "subscription_updated",
                {"keywords": ["ai", "具身智能"]},
                to="sid-1",
            )

        asyncio.run(run())

    def test_update_is_isolated_and_deduplicated_per_subscriber(self) -> None:
        async def run() -> None:
            hub = RealtimeHub(cors_allowed_origins=("http://localhost:5173",))
            hub._subscriptions = {
                "sid-ai": {"ai", "nvidia"},
                "sid-energy": {"energy"},
            }
            hub.sio.emit = AsyncMock()
            update = {
                "title": "NVIDIA publishes a new AI inference platform",
                "summary": "Lower latency for enterprise workloads.",
            }

            await hub._emit_updates([update])

            hub.sio.emit.assert_awaited_once_with(
                "industry_update",
                update,
                to=[keyword_room("ai"), keyword_room("nvidia")],
            )

        asyncio.run(run())

    def test_socketio_asgi_round_trip_and_websocket_upgrade(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                service = PipelineService(_test_settings(Path(tmp)))
                app = create_socketio_app(service)
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                port = listener.getsockname()[1]
                server = uvicorn.Server(
                    uvicorn.Config(app, log_level="error", lifespan="on")
                )
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                client = socketio.AsyncClient(reconnection=False)
                try:
                    for _ in range(100):
                        if server.started:
                            break
                        await asyncio.sleep(0.01)
                    self.assertTrue(server.started)

                    await client.connect(f"http://127.0.0.1:{port}")
                    acknowledgement = await client.call(
                        "subscribe_keywords",
                        {"keywords": ["AI", "NVIDIA"]},
                        timeout=2,
                    )
                    self.assertEqual(
                        acknowledgement,
                        {"keywords": ["ai", "nvidia"]},
                    )
                    self.assertEqual(client.transport(), "websocket")
                finally:
                    if client.connected:
                        await client.disconnect()
                    server.should_exit = True
                    await server_task

        asyncio.run(run())


class PipelineRealtimePayloadTests(unittest.TestCase):
    def test_pipeline_publishes_completed_brief_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _test_settings(root)
            repositories = InsightCastRepositories(
                backend="sqlite",
                sqlite_path=settings.sqlite_path,
            )
            interview = repositories.interviews.create(
                Interview(
                    id="interview_realtime",
                    title="AI infrastructure update",
                    url="https://example.com/ai-update",
                    status=InterviewStatus.PUSHED,
                    source_name="Industry Channel",
                    person_names=("Ada",),
                    industries=(Industry.AI,),
                )
            )
            repositories.interview_summaries.create(
                InterviewSummary(
                    id="summary_realtime",
                    interview_id=interview.id,
                    summary="A supported summary.",
                    key_points=("Inference costs declined.",),
                    mentioned_companies=("Example AI",),
                )
            )
            brief = repositories.daily_briefs.create(
                DailyBrief(
                    id="brief_realtime",
                    brief_date=date(2026, 8, 21),
                    title="Realtime brief",
                    status=BriefStatus.SENT,
                    items=(
                        DailyBriefItem(
                            interview_id=interview.id,
                            summary_id="summary_realtime",
                            section=BriefSection.MUST_READ,
                            rank=1,
                            title=interview.title,
                            url=interview.url,
                            industries=interview.industries,
                        ),
                    ),
                )
            )
            published = []
            service = PipelineService(
                settings,
                realtime_publisher=lambda updates: published.extend(updates),
            )
            try:
                service._publish_completed_updates(
                    "run_realtime",
                    SimpleNamespace(
                        brief_id=brief.id,
                        pushed_interview_ids=(interview.id,),
                    ),
                )
            finally:
                service.shutdown()

            self.assertEqual(len(published), 1)
            self.assertEqual(published[0]["interview_id"], interview.id)
            self.assertEqual(published[0]["summary"], "A supported summary.")
            self.assertEqual(published[0]["industries"], ["ai"])
            self.assertEqual(published[0]["section"], "must_read")


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
        cors_origins=("http://localhost:5173",),
    )


if __name__ == "__main__":
    unittest.main()
