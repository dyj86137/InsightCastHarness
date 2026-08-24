from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import List


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from core.event import RunEvent, RunEventType
from core.message import Message
from llm import ChatRequest, FakeLLM

from insightcast.agent.llm_trace import (
    TRACE_DEBUG_ENV,
    TRACE_MAX_CONTENT_CHARS_ENV,
    LLMTraceContext,
    TracedLLM,
    llm_trace_scope,
    safe_request_payload,
)


class LLMTraceTests(unittest.TestCase):
    def test_default_request_payload_only_records_request_id_and_messages(self) -> None:
        request = sample_request(
            metadata={
                "task": "insightcast_interview_classification",
                "candidate_id": "candidate_1",
                "estimated_input_tokens": 42,
                "prompt_pipeline": {"large": "debug-only"},
                "context_pruner": {"large": "debug-only"},
                "prune_decisions": [{"large": "debug-only"}],
                "content_budget": {"large": "debug-only"},
            }
        )

        payload = safe_request_payload(
            request,
            LLMTraceContext(
                run_id="run_1",
                trace_dir=None,
                stage="classification",
                step=2,
                env={TRACE_MAX_CONTENT_CHARS_ENV: "100"},
            ),
        )

        self.assertEqual(set(payload), {"request_id", "messages"})
        self.assertEqual(payload["request_id"], request.id)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["content"], "Classify this candidate.")

    def test_trace_events_record_input_output_and_duration(self) -> None:
        recorder = RecordingEventRecorder()
        llm = TracedLLM(FakeLLM(default_response="structured output"))

        response = run_async(
            call_traced_llm(
                llm,
                sample_request(),
                recorder,
                env={TRACE_MAX_CONTENT_CHARS_ENV: "100"},
            )
        )

        self.assertEqual(response.text, "structured output")
        self.assertEqual(
            [event.event_type for event in recorder.events],
            [RunEventType.LLM_REQUESTED, RunEventType.LLM_RESPONDED],
        )
        requested = recorder.events[0].payload
        responded = recorder.events[1].payload
        self.assertEqual(requested["messages"][1]["content"], "Classify this candidate.")
        self.assertEqual(
            set(responded),
            {"provider", "model", "request_id", "response_id", "finish_reason", "duration_ms", "output_text"},
        )
        self.assertEqual(responded["output_text"], "structured output")
        self.assertGreaterEqual(responded["duration_ms"], 0)

    def test_trace_events_record_failure_payload(self) -> None:
        recorder = RecordingEventRecorder()
        llm = TracedLLM(FakeLLM(responses=[RuntimeError("provider unavailable")]))

        with self.assertRaises(RuntimeError):
            run_async(
                call_traced_llm(
                    llm,
                    sample_request(),
                    recorder,
                    env={TRACE_MAX_CONTENT_CHARS_ENV: "100"},
                )
            )

        self.assertEqual(
            [event.event_type for event in recorder.events],
            [RunEventType.LLM_REQUESTED, RunEventType.LLM_FAILED],
        )
        failed = recorder.events[1]
        self.assertEqual(failed.payload["messages"][1]["content"], "Classify this candidate.")
        self.assertGreaterEqual(failed.payload["duration_ms"], 0)
        self.assertEqual(failed.error["type"], "RuntimeError")

    def test_long_content_is_truncated_with_hash(self) -> None:
        request = sample_request(user_content="x" * 30)

        payload = safe_request_payload(
            request,
            LLMTraceContext(
                run_id="run_1",
                trace_dir=None,
                stage="summary",
                step=5,
                env={TRACE_MAX_CONTENT_CHARS_ENV: "10"},
            ),
        )

        user_message = payload["messages"][1]
        self.assertEqual(user_message["content"], "x" * 10)
        self.assertEqual(user_message["content_chars"], 30)
        self.assertTrue(user_message["truncated"])
        self.assertEqual(len(user_message["content_hash"]), 64)

    def test_debug_mode_keeps_full_request_metadata(self) -> None:
        request = sample_request(
            metadata={
                "task": "insightcast_interview_summary",
                "prompt_pipeline": {"kept": True},
            }
        )

        payload = safe_request_payload(
            request,
            LLMTraceContext(
                run_id="run_1",
                trace_dir=None,
                stage="summary",
                step=5,
                env={TRACE_DEBUG_ENV: "1"},
            ),
        )

        self.assertEqual(payload["debug_metadata"]["prompt_pipeline"], {"kept": True})


def sample_request(
    *,
    user_content: str = "Classify this candidate.",
    metadata: dict | None = None,
) -> ChatRequest:
    return ChatRequest.create(
        messages=(
            Message.system("You are a classifier."),
            Message.user(user_content),
        ),
        model="fake-model",
        temperature=0,
        max_tokens=100,
        timeout_seconds=30,
        metadata=metadata
        or {
            "task": "insightcast_interview_classification",
            "candidate_id": "candidate_1",
        },
    )


async def call_traced_llm(
    llm: TracedLLM,
    request: ChatRequest,
    recorder: "RecordingEventRecorder",
    *,
    env: dict[str, str],
):
    with llm_trace_scope(
        run_id="run_1",
        trace_dir=None,
        stage="classification",
        step=2,
        event_recorder=recorder,
        env=env,
    ):
        return await llm.chat(request)


def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class RecordingEventRecorder:
    def __init__(self) -> None:
        self.events: List[RunEvent] = []

    async def record_event(self, event: RunEvent) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
