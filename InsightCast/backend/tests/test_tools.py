from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from core.event import RunEventType
from tracing.jsonl import JsonlTraceStore
from tools.mcp import InMemoryMCPClient, MCPToolProvider, MCPToolSpec

from insightcast.domain.models import DailyBrief
from insightcast.tools import (
    FETCH_JSON_TOOL_NAME,
    LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME,
    WRITE_MARKDOWN_BRIEF_TOOL_NAME,
    LemonadeTranscribeAudioTool,
    ToolJsonFetcher,
    ToolMarkdownBriefWriter,
    create_insightcast_tool_executor,
    tool_trace_scope,
)


class InsightCastToolTests(unittest.TestCase):
    def test_tool_adapters_fetch_json_write_markdown_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fake_fetch_json(
                url: str,
                params: Mapping[str, Any],
                headers: Mapping[str, str],
            ) -> Mapping[str, Any]:
                calls.append((url, dict(params), dict(headers)))
                return {"ok": True, "items": [{"id": params["q"]}]}

            executor = create_insightcast_tool_executor(fetch_json=fake_fetch_json)
            run_id = "run_tool_adapters"

            with tool_trace_scope(
                run_id=run_id,
                trace_dir=root / "traces",
                stage="discovery",
                step=1,
                cwd=root,
            ):
                payload = ToolJsonFetcher(executor)(
                    "https://example.test/search",
                    {
                        "q": "nvidia",
                        "keyword": "Jensen Huang 访谈",
                        "key": "secret-api-key",
                        "api_key": "another-secret",
                    },
                    {"X-Test": "1"},
                )

            self.assertEqual(payload["items"][0]["id"], "nvidia")
            self.assertEqual(calls[0][0], "https://example.test/search")

            brief = DailyBrief(
                brief_date=date(2026, 7, 12),
                title="InsightCast Daily Brief - 2026-07-12",
            )
            with tool_trace_scope(
                run_id=run_id,
                trace_dir=root / "traces",
                stage="markdown_brief",
                step=7,
                cwd=root,
            ):
                markdown_path = ToolMarkdownBriefWriter(executor)(
                    "# InsightCast Daily Brief",
                    brief,
                    root / "briefs",
                )

            self.assertEqual(markdown_path.name, "2026-07-12-insightcast-brief.md")
            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                "# InsightCast Daily Brief\n",
            )

            events = load_trace_events(root / "traces", run_id)
            completed = [
                event
                for event in events
                if event.event_type == RunEventType.TOOL_CALL_COMPLETED
            ]
            self.assertEqual(
                [event.payload.get("tool_name") for event in completed],
                [FETCH_JSON_TOOL_NAME, WRITE_MARKDOWN_BRIEF_TOOL_NAME],
            )
            self.assertEqual(completed[0].payload.get("stage"), "discovery")
            self.assertEqual(completed[1].payload.get("stage"), "markdown_brief")
            result_metadata = completed[0].payload.get("result_metadata")
            self.assertEqual(
                result_metadata["request_url"],
                "https://example.test/search?q=nvidia&keyword=Jensen+Huang+%E8%AE%BF%E8%B0%88&key=%5Bredacted%5D&api_key=%5Bredacted%5D",
            )
            self.assertNotIn("top_level_keys", result_metadata)
            self.assertNotIn("data_keys", completed[0].payload)
            self.assertNotIn("secret-api-key", result_metadata["request_url"])
            self.assertNotIn("another-secret", result_metadata["request_url"])

    def test_lemonade_wrapper_normalizes_mcp_transcription_result(self) -> None:
        client = InMemoryMCPClient(
            server_name="lemonade",
            tools=(
                MCPToolSpec(
                    name="lemonade_transcribe_audio",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "audio_path": {"type": "string"},
                            "model": {"type": "string"},
                            "response_format": {"type": "string"},
                        },
                        "required": ["audio_path"],
                    },
                ),
            ),
            handlers={
                "lemonade_transcribe_audio": lambda arguments: {
                    "text": f"transcribed {arguments['audio_path']}",
                    "language": "zh",
                }
            },
        )
        executor = create_insightcast_tool_executor()
        for tool in MCPToolProvider(client=client).discover():
            executor.register(tool)
        executor.register(LemonadeTranscribeAudioTool(executor))

        async def run_tool():
            from core.tool import ToolCall
            from tools.base import ToolExecutionContext

            return await executor.execute(
                ToolCall.create(
                    name=LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME,
                    arguments={"audio_path": "D:/tmp/audio.m4a"},
                ),
                ToolExecutionContext(run_id="run_lemonade", step=1),
            )

        result = asyncio.run(run_tool())

        self.assertFalse(result.is_error)
        self.assertEqual(result.data["text"], "transcribed D:/tmp/audio.m4a")
        self.assertEqual(result.data["language"], "zh")


def load_trace_events(trace_dir: Path, run_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        store = JsonlTraceStore(trace_dir)
        return loop.run_until_complete(store.list_events(run_id))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


if __name__ == "__main__":
    unittest.main()
