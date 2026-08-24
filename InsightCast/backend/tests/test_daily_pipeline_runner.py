from __future__ import annotations

import hashlib
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM

from core.event import RunEventType
from runtime.checkpoint import FileCheckpointStore
from runtime.hooks import BaseHook, HookContext, HookEvent, HookManager
from tracing.jsonl import JsonlTraceStore

from insightcast.agent import LLMInterviewClassifier, LLMInterviewSummarizer
from insightcast.agent.pipeline_runner import DailyPipelineRunner
from insightcast.briefs import LLMMarkdownBriefGenerator
from insightcast.domain.enums import InterviewStatus
from insightcast.run.daily_run import format_result, run_daily_pipeline
from insightcast.storage.repositories import InsightCastRepositories
from insightcast.tools import FETCH_JSON_TOOL_NAME, WRITE_MARKDOWN_BRIEF_TOOL_NAME


class DailyPipelineRunnerTests(unittest.TestCase):
    def test_run_daily_pipeline_executes_all_stages_and_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(pipeline_config()), encoding="utf-8")
            sqlite_path = root / "insightcast.sqlite3"
            recording_hook = RecordingHook()
            hook_manager = HookManager([recording_hook])

            result = run_daily_pipeline(
                config_path=config_path,
                storage_backend="sqlite",
                sqlite_path=sqlite_path,
                brief_output_dir=root / "briefs",
                trace_dir=root / "traces",
                checkpoint_dir=root / "checkpoints",
                fetch_json=bilibili_fetch_json,
                classifier=LLMInterviewClassifier(
                    FakeLLM(default_response=classification_response())
                ),
                summarizer=LLMInterviewSummarizer(
                    FakeLLM(default_response=summary_response())
                ),
                brief_generator=LLMMarkdownBriefGenerator(
                    FakeLLM(responses=["# InsightCast Daily Brief\n\n## Must Read\n\n- NVIDIA"])
                ),
                discovery_agent_enabled=False,
                brief_agent_enabled=False,
                env={"INSIGHTCAST_TRANSCRIPT_PLATFORM_CAPTIONS": "0"},
                hook_manager=hook_manager,
            )

            self.assertEqual(result.status, "finished")
            self.assertGreater(result.discovered_count, 0)
            self.assertEqual(result.accepted_count, result.discovered_count)
            self.assertEqual(result.pushed_count, 8)
            self.assertTrue(Path(result.markdown_path).exists())
            self.assertTrue(Path(result.error_log_path).exists())
            self.assertEqual(Path(result.error_log_path).read_text(encoding="utf-8"), "")
            self.assertEqual(
                [stage.name for stage in result.stages],
                [
                    "discovery",
                    "classification",
                    "interview_promotion",
                    "transcript_fetch",
                    "summary",
                    "ranking",
                    "markdown_brief",
                ],
            )
            self.assertTrue(all(stage.status == "finished" for stage in result.stages))

            repositories = InsightCastRepositories(
                root_dir=sqlite_path,
                backend="sqlite",
            )
            self.assertEqual(
                repositories.run_records.require(result.run_id).metadata["stage"],
                "pipeline",
            )
            pushed = repositories.interviews.list_by_status(InterviewStatus.PUSHED)
            archived = repositories.interviews.list_by_status(InterviewStatus.ARCHIVED)
            self.assertEqual(len(pushed), 8)
            self.assertEqual(len(archived), result.accepted_count - 8)
            self.assertIn("Daily pipeline finished", format_result(result))
            self.assertIn("markdown:", format_result(result))
            self.assertIn("error_log:", format_result(result))

            trace_events = load_trace_events(root / "traces", result.run_id)
            self.assertEqual(trace_events[0].event_type, RunEventType.RUN_STARTED)
            self.assertEqual(trace_events[-1].event_type, RunEventType.CHECKPOINT_CREATED)
            self.assertTrue(
                any(event.event_type == RunEventType.RUN_COMPLETED for event in trace_events)
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.LLM_REQUESTED
                    and event.payload.get("stage") == "classification"
                    for event in trace_events
                )
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.LLM_RESPONDED
                    and event.payload.get("stage") == "markdown_brief"
                    for event in trace_events
                )
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.STEP_COMPLETED
                    and event.payload.get("stage") == "markdown_brief"
                    for event in trace_events
                )
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.TOOL_CALL_STARTED
                    and event.payload.get("stage") == "discovery"
                    and event.payload.get("tool_name") == FETCH_JSON_TOOL_NAME
                    for event in trace_events
                )
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.TOOL_CALL_COMPLETED
                    and event.payload.get("stage") == "markdown_brief"
                    and event.payload.get("tool_name") == WRITE_MARKDOWN_BRIEF_TOOL_NAME
                    for event in trace_events
                )
            )
            self.assertTrue(
                any(
                    event.event_type == RunEventType.CONTEXT_BUILT
                    and event.payload.get("stage") == "summary"
                    and event.payload.get("context_name") == "summary_transcript"
                    for event in trace_events
                )
            )
            summary_stage = next(stage for stage in result.stages if stage.name == "summary")
            self.assertTrue(Path(summary_stage.metrics["evidence_path"]).exists())
            checkpoints = load_checkpoints(root / "checkpoints", result.run_id)
            self.assertGreaterEqual(len(checkpoints), 8)
            self.assertEqual(checkpoints[-1].state.status.value, "completed")
            self.assertEqual(
                checkpoints[-1].state.variables["completed_stages"][-1],
                "markdown_brief",
            )
            self.assertIn((HookEvent.RUN_STARTED, None), recording_hook.events)
            self.assertIn((HookEvent.STEP_STARTED, "discovery"), recording_hook.events)
            self.assertIn((HookEvent.STEP_COMPLETED, "markdown_brief"), recording_hook.events)
            self.assertIn((HookEvent.BEFORE_LLM_CALL, "classification"), recording_hook.events)
            self.assertIn((HookEvent.AFTER_LLM_CALL, "markdown_brief"), recording_hook.events)
            self.assertIn((HookEvent.BEFORE_TOOL_CALL, "discovery"), recording_hook.events)
            self.assertIn((HookEvent.AFTER_TOOL_CALL, "markdown_brief"), recording_hook.events)
            self.assertIn((HookEvent.CONTEXT_BUILT, "summary"), recording_hook.events)
            self.assertIn((HookEvent.CHECKPOINT_CREATED, None), recording_hook.events)
            self.assertIn((HookEvent.RUN_COMPLETED, None), recording_hook.events)

    def test_resume_from_checkpoint_skips_completed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(pipeline_config()), encoding="utf-8")
            sqlite_path = root / "insightcast.sqlite3"

            with self.assertRaises(RuntimeError):
                run_daily_pipeline(
                    config_path=config_path,
                    storage_backend="sqlite",
                    sqlite_path=sqlite_path,
                    brief_output_dir=root / "briefs",
                    trace_dir=root / "traces",
                    checkpoint_dir=root / "checkpoints",
                    fetch_json=bilibili_fetch_json,
                    classifier=LLMInterviewClassifier(
                        FakeLLM(default_response=classification_response())
                    ),
                    summarizer=LLMInterviewSummarizer(
                        FakeLLM(default_response=summary_response())
                    ),
                    brief_generator=LLMMarkdownBriefGenerator(
                        FakeLLM(responses=[RuntimeError("brief generation failed")])
                    ),
                    discovery_agent_enabled=False,
                    brief_agent_enabled=False,
                    env={"INSIGHTCAST_TRANSCRIPT_PLATFORM_CAPTIONS": "0"},
                )

            repositories = InsightCastRepositories(
                root_dir=sqlite_path,
                backend="sqlite",
            )
            failed_pipeline_run = [
                record
                for record in repositories.run_records.list()
                if record.metadata.get("stage") == "pipeline"
            ][0]
            self.assertEqual(failed_pipeline_run.status, "failed")
            error_log_path = Path(failed_pipeline_run.metadata["error_log_path"])
            self.assertTrue(error_log_path.exists())
            error_log_entries = load_error_log_entries(error_log_path)
            self.assertTrue(
                any(
                    log_extra(entry).get("event") == "stage_exception"
                    and log_extra(entry).get("stage") == "markdown_brief"
                    for entry in error_log_entries
                )
            )
            self.assertTrue(
                any(log_extra(entry).get("event") == "run_exception" for entry in error_log_entries)
            )
            self.assertTrue(
                all(
                    str(entry.get("logger", "")).startswith(
                        "myharness.insightcast.pipeline.error."
                    )
                    for entry in error_log_entries
                )
            )

            resumed = run_daily_pipeline(
                config_path=config_path,
                storage_backend="sqlite",
                sqlite_path=sqlite_path,
                brief_output_dir=root / "briefs",
                trace_dir=root / "traces",
                checkpoint_dir=root / "checkpoints",
                resume_run_id=failed_pipeline_run.id,
                fetch_json=failing_fetch_json,
                classifier=LLMInterviewClassifier(
                    FakeLLM(responses=[RuntimeError("classifier should be skipped")])
                ),
                summarizer=LLMInterviewSummarizer(
                    FakeLLM(responses=[RuntimeError("summarizer should be skipped")])
                ),
                brief_generator=LLMMarkdownBriefGenerator(
                    FakeLLM(responses=["# Resumed Brief\n\n## Must Read\n\n- NVIDIA"])
                ),
                discovery_agent_enabled=False,
                brief_agent_enabled=False,
                env={"INSIGHTCAST_TRANSCRIPT_PLATFORM_CAPTIONS": "0"},
            )

            self.assertEqual(resumed.status, "finished")
            self.assertEqual(
                [stage.status for stage in resumed.stages[:-1]],
                ["skipped", "skipped", "skipped", "skipped", "skipped", "skipped"],
            )
            self.assertEqual(resumed.stages[-1].name, "markdown_brief")
            self.assertEqual(resumed.stages[-1].status, "finished")
            self.assertEqual(resumed.pushed_count, 8)
            self.assertTrue(Path(resumed.markdown_path).exists())
            failed_trace_events = load_trace_events(root / "traces", failed_pipeline_run.id)
            self.assertTrue(
                any(event.event_type == RunEventType.LLM_FAILED for event in failed_trace_events)
            )

    def test_transcript_fetch_limit_is_applied_by_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(pipeline_config()), encoding="utf-8")

            result = run_daily_pipeline(
                config_path=config_path,
                storage_backend="sqlite",
                sqlite_path=root / "insightcast.sqlite3",
                brief_output_dir=root / "briefs",
                trace_dir=root / "traces",
                checkpoint_dir=root / "checkpoints",
                fetch_json=bilibili_fetch_json,
                classifier=LLMInterviewClassifier(
                    FakeLLM(default_response=classification_response())
                ),
                summarizer=LLMInterviewSummarizer(
                    FakeLLM(default_response=summary_response())
                ),
                brief_generator=LLMMarkdownBriefGenerator(
                    FakeLLM(responses=["# Limited Brief\n\n## Must Read\n\n- NVIDIA"])
                ),
                discovery_agent_enabled=False,
                brief_agent_enabled=False,
                env={"INSIGHTCAST_TRANSCRIPT_PLATFORM_CAPTIONS": "0"},
                transcript_fetch_limit=1,
            )

            transcript_stage = next(
                stage for stage in result.stages if stage.name == "transcript_fetch"
            )
            self.assertEqual(transcript_stage.metrics["processed_count"], 1)

    def test_pipeline_runner_disables_total_run_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            runner = DailyPipelineRunner(
                repositories,
                classifier=LLMInterviewClassifier(FakeLLM(default_response=classification_response())),
                summarizer=LLMInterviewSummarizer(FakeLLM(default_response=summary_response())),
                brief_generator=LLMMarkdownBriefGenerator(FakeLLM(default_response="# Brief")),
            )

            self.assertEqual(runner.config.runner.run_timeout_seconds, 0)


def pipeline_config() -> Mapping[str, Any]:
    return {
        "industries": [
            {
                "id": "industry_ai",
                "name": "Artificial Intelligence",
                "slug": "ai",
                "keywords": ["AI", "GPU"],
            }
        ],
        "people": [
            {
                "id": "person_jensen",
                "name": "Jensen Huang",
                "aliases": ["黄仁勋"],
                "companies": ["NVIDIA"],
                "title": "CEO",
                "industries": ["ai"],
                "importance": 0.95,
            }
        ],
        "sources": [
            {
                "id": "source_bilibili",
                "name": "Bilibili Search",
                "type": "bilibili",
                "authority": 0.6,
                "languages": ["zh"],
                "industries": ["ai"],
                "metadata": {"max_results": 1, "order": "pubdate"},
            }
        ],
        "interests": [
            {
                "id": "interest_default",
                "industries": ["ai"],
                "people": ["person_jensen"],
                "companies": ["NVIDIA"],
                "keywords": ["AI infrastructure"],
                "push_channels": ["markdown"],
            }
        ],
    }


def bilibili_fetch_json(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    keyword = str(params["keyword"])
    digest = hashlib.sha1(keyword.encode("utf-8")).hexdigest()[:10]
    return {
        "code": 0,
        "data": {
            "result": [
                {
                    "bvid": f"BV{digest}",
                    "title": f"{keyword} 深度访谈",
                    "description": "NVIDIA CEO discusses AI infrastructure and enterprise demand.",
                    "duration": "45:00",
                    "pubdate": 1783579200,
                }
            ]
        },
    }


def failing_fetch_json(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    raise RuntimeError("discovery should be skipped")


def load_trace_events(trace_dir: Path, run_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        store = JsonlTraceStore(trace_dir)
        return loop.run_until_complete(store.list_events(run_id))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def load_checkpoints(checkpoint_dir: Path, run_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        store = FileCheckpointStore(checkpoint_dir)
        return loop.run_until_complete(store.list_for_run(run_id))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def load_error_log_entries(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def log_extra(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = entry.get("extra") or {}
    return extra if isinstance(extra, dict) else {}


class RecordingHook(BaseHook):
    name = "recording_hook"

    def __init__(self) -> None:
        self.events: list[tuple[HookEvent, str | None]] = []

    async def on_event(self, event: HookEvent, context: HookContext) -> None:
        stage = context.payload.get("stage")
        self.events.append((event, str(stage) if stage is not None else None))


def classification_response() -> str:
    return """
    {
      "is_qualifying_content": true,
      "target_person_present": true,
      "content_type": "interview",
      "is_short_clip_or_commentary": false,
      "confidence": 0.9,
      "reason": "The candidate is a long-form interview with the target person."
    }
    """


def summary_response() -> str:
    return """
    {
      "summary": "Jensen Huang 继续强调 AI 基础设施需求。企业推理负载可能扩大。数据中心仍是 NVIDIA 的战略重点。访谈同时说明企业部署需要平衡算力成本、供应链弹性、软件生态和应用落地节奏，这为 AI 基础设施服务提供了持续观察的方向。该材料适合关注 AI 投资和企业技术部署的人阅读。当前 transcript 只包含标题、简介和来源信息，证据不足，不能据此确认更多具体产品、合同或时间表。相关判断仅用于概括可见材料，后续仍需完整访谈文本验证。",
      "key_points": ["GPU 需求仍然强劲"],
      "potential_opportunities": ["NVIDIA 继续强调数据中心扩张"],
      "industry_judgements": ["AI 基础设施仍处于高景气阶段"],
      "mentioned_companies": ["NVIDIA"],
      "mentioned_products": ["GPU"],
      "evidence_claims": [
        {
          "field": "summary",
          "claim": "Jensen Huang 继续强调 AI 基础设施需求。",
          "model_confidence": 0.8,
          "evidence": [{"text": "Description: NVIDIA CEO discusses AI infrastructure and enterprise demand.", "segment_id": null}]
        },
        {
          "field": "potential_opportunities",
          "claim": "NVIDIA 继续强调数据中心扩张",
          "model_confidence": 0.75,
          "evidence": [{"text": "Description: NVIDIA CEO discusses AI infrastructure and enterprise demand.", "segment_id": null}]
        }
      ],
      "audience": ["AI 投资研究员"],
      "novelty_assessment": "变化：本次更强调企业推理需求。",
      "insights": []
    }
    """


if __name__ == "__main__":
    unittest.main()
