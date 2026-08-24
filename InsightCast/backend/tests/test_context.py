from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from core.event import RunEventType
from tracing.jsonl import JsonlTraceStore

from insightcast.context import (
    TranscriptContextProvider,
    context_trace_scope,
    create_summary_context_provider_from_env,
)
from insightcast.domain.enums import (
    ContentFormat,
    InterviewStatus,
    InterviewType,
    TranscriptSource,
    TranscriptStatus,
)
from insightcast.domain.models import Interview, Transcript


class TranscriptContextTests(unittest.TestCase):
    def test_summary_context_provider_reads_tokenizer_and_budget_env(self) -> None:
        provider = create_summary_context_provider_from_env(
            {
                "INSIGHTCAST_TOKENIZER_PROVIDER": "heuristic",
                "INSIGHTCAST_TOKENIZER_CHARS_PER_TOKEN": "2",
                "INSIGHTCAST_SUMMARY_CONTEXT_MAX_INPUT_TOKENS": "100",
                "INSIGHTCAST_SUMMARY_CONTEXT_RESERVED_OUTPUT_TOKENS": "10",
                "INSIGHTCAST_SUMMARY_CONTEXT_SAFETY_MARGIN_TOKENS": "10",
                "INSIGHTCAST_SUMMARY_CONTEXT_TRANSCRIPT_MAX_TOKENS": "60",
                "INSIGHTCAST_SUMMARY_CONTEXT_TRANSCRIPT_BUDGET_RATIO": "0.5",
                "INSIGHTCAST_SUMMARY_CONTEXT_CHUNK_TOKEN_LIMIT": "7",
            }
        )

        self.assertEqual(provider.estimator.name, "heuristic")
        self.assertEqual(provider.estimator.chars_per_token, 2)
        self.assertEqual(provider.chunk_token_limit, 7)
        self.assertEqual(provider._transcript_token_budget(), 40)

    def test_long_transcript_is_chunked_compressed_and_traced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interview = make_interview()
            transcript = make_long_transcript(interview.id)
            provider = TranscriptContextProvider(
                max_transcript_tokens=120,
                chunk_token_limit=30,
            )

            with context_trace_scope(
                run_id="run_context_test",
                trace_dir=root / "traces",
                stage="summary",
                step=5,
            ):
                context = asyncio.run(
                    provider.build_transcript_context(interview, transcript)
                )

            self.assertTrue(context.was_compressed)
            self.assertGreater(context.chunk_count, len(context.selected_chunks))
            self.assertGreater(len(context.dropped_chunk_indexes), 0)
            self.assertLessEqual(context.estimated_tokens, context.token_budget)
            self.assertIn("[上下文已压缩]", context.text)
            self.assertIn("被丢弃块的抽取式摘要", context.text)

            events = load_trace_events(root / "traces", "run_context_test")
            self.assertTrue(
                any(
                    event.event_type == RunEventType.CONTEXT_BUILT
                    and event.payload.get("stage") == "summary"
                    and event.payload.get("context_name") == "summary_transcript"
                    and event.payload.get("was_compressed") is True
                    for event in events
                )
            )


def make_interview() -> Interview:
    return Interview(
        id="interview_context",
        title="Jensen Huang long interview on AI infrastructure",
        url="https://example.com/interview_context",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.TRANSCRIPT_READY,
        person_names=("Jensen Huang",),
        transcript_status=TranscriptStatus.READY,
    )


def make_long_transcript(interview_id: str) -> Transcript:
    paragraphs = [
        (
            f"段落 {index}: Jensen Huang discusses AI infrastructure, GPU supply, "
            "enterprise inference demand, data center buildout, and software ecosystem. "
            "This part contains detailed business signals for investors."
        )
        for index in range(1, 45)
    ]
    return Transcript(
        id="transcript_context",
        interview_id=interview_id,
        status=TranscriptStatus.READY,
        source=TranscriptSource.OFFICIAL_CAPTION,
        text="\n\n".join(paragraphs),
    )


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
