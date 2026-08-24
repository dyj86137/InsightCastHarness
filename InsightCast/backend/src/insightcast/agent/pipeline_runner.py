"""InsightCast V1 日常 pipeline 的端到端 runner。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from pydantic import Field

from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.result import RunResult
from core.state import AgentState
from infra.config import HarnessConfig, RunnerConfig, TraceConfig
from runtime.checkpoint import (
    CheckpointManager,
    CheckpointReason,
    FileCheckpointStore,
)
from runtime.hooks import HookManager
from runtime.runner import Runner, RunnerContext
from tracing.base import TraceStore
from tracing.jsonl import JsonlTraceStore

from insightcast.agent.classification_runner import ClassificationRunner
from insightcast.agent.brief_writing_agent import BriefWritingAgent
from insightcast.agent.discovery_research_agent import DiscoveryResearchAgent
from insightcast.agent.discovery_runner import DiscoveryRunner
from insightcast.agent.interview_classifier import InterviewClassifier
from insightcast.agent.interview_promotion_runner import InterviewPromotionRunner
from insightcast.agent.interview_summarizer import InterviewSummarizer
from insightcast.agent.error_log import PipelineErrorLogHook
from insightcast.agent.llm_trace import llm_trace_scope
from insightcast.agent.markdown_brief_runner import MarkdownBriefRunner
from insightcast.agent.ranking_runner import InterviewRankingRunner
from insightcast.agent.summary_runner import SummaryRunner
from insightcast.agent.transcript_fetch_runner import TranscriptFetchRunner
from insightcast.briefs import MarkdownBriefGenerator
from insightcast.context import context_trace_scope
from insightcast.domain.models import DomainModel, RunRecord, new_id
from insightcast.harness_events import HarnessEventRecorder, harness_event_scope
from insightcast.sources.discovery import (
    DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS,
    JsonFetcher,
)
from insightcast.storage.repositories import InsightCastRepositories
from insightcast.transcript import (
    DEFAULT_PREFERRED_CAPTION_LANGUAGES,
    create_default_transcript_provider,
)
from insightcast.tools import (
    LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME,
    ToolJsonFetcher,
    ToolMarkdownBriefWriter,
    ToolAudioDownloader,
    ToolAudioTranscriber,
    ToolPlatformCaptionFetcher,
    create_insightcast_tool_executor,
    tool_trace_scope,
)


StageResultT = TypeVar("StageResultT", bound=DomainModel)


PIPELINE_STAGE_NAMES = (
    "discovery",
    "classification",
    "interview_promotion",
    "transcript_fetch",
    "summary",
    "ranking",
    "markdown_brief",
)


class PipelineStageResult(DomainModel):
    """单个 pipeline 阶段的标准化结果。"""

    name: str
    status: str
    run_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DailyPipelineRunResult(DomainModel):
    """一次端到端日常 pipeline 运行的结果。"""

    run_id: str
    status: str
    stages: Tuple[PipelineStageResult, ...] = Field(default_factory=tuple)
    markdown_path: Optional[str] = None
    brief_id: Optional[str] = None
    error_log_path: Optional[str] = None
    discovered_count: int = 0
    accepted_count: int = 0
    pushed_count: int = 0
    pushed_interview_ids: Tuple[str, ...] = Field(default_factory=tuple)


class NoopTraceStore(TraceStore):
    """关闭 trace 输出时使用的 TraceStore。"""

    async def append(self, event: RunEvent) -> None:
        del event

    async def list_events(self, run_id: str) -> List[RunEvent]:
        del run_id
        return []


class InsightCastDailyPipelineLoop:
    """执行 InsightCast V1 日常 pipeline 各阶段的 myHarness loop。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        classifier: InterviewClassifier,
        summarizer: InterviewSummarizer,
        brief_generator: MarkdownBriefGenerator,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
        fail_fast: bool = False,
        discovery_agent: Optional[DiscoveryResearchAgent] = None,
        brief_output_dir: Optional[Path] = None,
        brief_date: Optional[date] = None,
        brief_agent: Optional[BriefWritingAgent] = None,
        trace_dir: Optional[Path] = None,
        trace_enabled: bool = True,
        restored_state: Optional[AgentState] = None,
        tools_enabled: bool = True,
        discovery_fetch_timeout_seconds: Optional[float] = None,
        transcript_fetch_limit: Optional[int] = None,
    ) -> None:
        self.repositories = repositories
        self.classifier = classifier
        self.summarizer = summarizer
        self.brief_generator = brief_generator
        self.env = env
        self.fetch_json_source = fetch_json
        self.fail_fast = fail_fast
        self.discovery_agent = discovery_agent
        self.brief_output_dir = brief_output_dir
        self.brief_date = brief_date
        self.brief_agent = brief_agent
        self.trace_dir = trace_dir
        self.trace_enabled = trace_enabled
        self.restored_state = restored_state
        self.tools_enabled = tools_enabled
        self.transcript_fetch_limit = transcript_fetch_limit
        self.discovery_fetch_timeout_seconds = (
            discovery_fetch_timeout_seconds
            if discovery_fetch_timeout_seconds is not None
            else _env_float(
                env,
                "INSIGHTCAST_DISCOVERY_FETCH_TIMEOUT_SECONDS",
                DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS,
            )
        )
        self.tool_executor = (
            create_insightcast_tool_executor(fetch_json=fetch_json, env=env)
            if tools_enabled
            else None
        )
        self.fetch_json = (
            ToolJsonFetcher(
                self.tool_executor,
                timeout_seconds=self.discovery_fetch_timeout_seconds,
            )
            if self.tool_executor is not None
            else fetch_json
        )
        self.markdown_writer = (
            ToolMarkdownBriefWriter(self.tool_executor)
            if self.tool_executor is not None
            else None
        )
        transcript_audio_dir = Path(self.repositories.root_dir) / "transcript_audio"
        audio_transcription_tool = _env_value(
            self.env,
            "INSIGHTCAST_AUDIO_TRANSCRIPTION_TOOL",
        ) or (
            LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME
            if _env_flag(self.env, "INSIGHTCAST_LEMONADE_MCP_ENABLED", False)
            else None
        )
        self.transcript_provider = create_default_transcript_provider(
            env=self.env,
            platform_caption_fetcher=(
                ToolPlatformCaptionFetcher(
                    self.tool_executor,
                    preferred_languages=_env_csv(
                        self.env,
                        "INSIGHTCAST_TRANSCRIPT_LANGUAGES",
                        DEFAULT_PREFERRED_CAPTION_LANGUAGES,
                    ),
                )
                if self.tool_executor is not None
                else None
            ),
            audio_downloader=(
                ToolAudioDownloader(
                    self.tool_executor,
                    output_dir=transcript_audio_dir,
                )
                if self.tool_executor is not None
                else None
            ),
            audio_transcriber=(
                ToolAudioTranscriber(
                    self.tool_executor,
                    tool_name=audio_transcription_tool,
                )
                if self.tool_executor is not None and audio_transcription_tool
                else None
            ),
            audio_output_dir=transcript_audio_dir,
        )

    async def run(self, state: AgentState, context: RunnerContext) -> AgentState:
        """运行全部 pipeline 阶段，并返回终态 AgentState。"""

        state = self._merge_restored_state(state)
        stages: List[PipelineStageResult] = []

        state = await self._run_stage(
            "discovery",
            1,
            state,
            stages,
            context,
            lambda: DiscoveryRunner(
                self.repositories,
                env=self.env,
                fetch_json=self.fetch_json,
                discovery_agent=self.discovery_agent,
                fail_fast=self.fail_fast,
            ).run_once(),
        )
        state = await self._run_stage(
            "classification",
            2,
            state,
            stages,
            context,
            lambda: ClassificationRunner(
                self.repositories,
                self.classifier,
                fail_fast=self.fail_fast,
            ).run_once(),
        )
        state = await self._run_stage(
            "interview_promotion",
            3,
            state,
            stages,
            context,
            lambda: InterviewPromotionRunner(
                self.repositories,
                fail_fast=self.fail_fast,
            ).run_once(),
        )
        state = await self._run_stage(
            "transcript_fetch",
            4,
            state,
            stages,
            context,
            lambda: TranscriptFetchRunner(
                self.repositories,
                provider=self.transcript_provider,
                fail_fast=self.fail_fast,
                limit=self.transcript_fetch_limit,
            ).run_once(),
        )
        state = await self._run_stage(
            "summary",
            5,
            state,
            stages,
            context,
            lambda: SummaryRunner(
                self.repositories,
                self.summarizer,
                fail_fast=self.fail_fast,
            ).run_once(),
        )
        state = await self._run_stage(
            "ranking",
            6,
            state,
            stages,
            context,
            lambda: InterviewRankingRunner(
                self.repositories,
                fail_fast=self.fail_fast,
            ).run_once(),
        )
        state = await self._run_stage(
            "markdown_brief",
            7,
            state,
            stages,
            context,
            lambda: MarkdownBriefRunner(
                self.repositories,
                self.brief_generator,
                output_dir=self.brief_output_dir,
                brief_date=self.brief_date,
                async_markdown_writer=(
                    self.markdown_writer.write_async
                    if self.markdown_writer is not None
                    else None
                ),
                brief_agent=self.brief_agent,
            ).run_once(),
        )

        final_status = _final_status(tuple(stages))
        pipeline_result = _pipeline_result_payload(
            run_id=state.run_id,
            final_status=final_status,
            stages=tuple(stages),
            state=state,
        )
        final_state = state._with_updates(
            step=len(PIPELINE_STAGE_NAMES),
            variables={**state.variables, "pipeline_result": pipeline_result},
        )
        return final_state.mark_completed(final_status)

    def _merge_restored_state(self, state: AgentState) -> AgentState:
        if self.restored_state is None:
            return state

        metadata = dict(state.metadata)
        metadata["resumed_from_run_id"] = self.restored_state.run_id
        metadata["resumed_from_step"] = self.restored_state.step
        return state._with_updates(
            variables=dict(self.restored_state.variables),
            metadata=metadata,
        )

    async def _run_stage(
        self,
        name: str,
        step: int,
        state: AgentState,
        stages: List[PipelineStageResult],
        context: RunnerContext,
        execute: Callable[[], StageResultT],
    ) -> AgentState:
        existing_metrics = _state_stage_results(state).get(name)
        if existing_metrics is not None:
            stages.append(
                PipelineStageResult(
                    name=name,
                    status="skipped",
                    run_id=existing_metrics.get("run_id"),
                    metrics=existing_metrics,
                )
            )
            return state._with_updates(step=step)

        started_state = state._with_updates(step=step)
        await context.record_event(
            RunEvent.step_started(
                run_id=state.run_id,
                session_id=state.session_id,
                step=step,
                payload={"stage": name},
            ),
            state=started_state,
        )

        try:
            result = await self._execute_stage(
                name,
                step,
                state.run_id,
                context,
                execute,
            )
        except Exception as exc:
            failed_state = _state_with_stage_failure(started_state, name, exc)
            await context.record_event(
                RunEvent.create(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    event_type=RunEventType.STEP_FAILED,
                    step=step,
                    level=RunEventLevel.ERROR,
                    message="Pipeline 阶段失败。",
                    payload={"stage": name},
                    error=error_to_payload(exc),
                ),
                state=failed_state,
            )
            await context.create_checkpoint(
                failed_state,
                reason=CheckpointReason.RUN_FAILED,
                metadata={"stage": name},
            )
            raise

        stage = _stage_result(name, result)
        stages.append(stage)
        completed_state = _state_after_stage(started_state, stage)
        await context.record_event(
            RunEvent.step_completed(
                run_id=state.run_id,
                session_id=state.session_id,
                step=step,
                payload={
                    "stage": name,
                    "status": stage.status,
                    "metrics": stage.metrics,
                },
            ),
            state=completed_state,
        )
        return completed_state

    async def _execute_stage(
        self,
        name: str,
        step: int,
        run_id: str,
        context: RunnerContext,
        execute: Callable[[], StageResultT],
    ) -> StageResultT:
        event_recorder = HarnessEventRecorder(
            runner_context=context,
            runner_loop=asyncio.get_running_loop(),
            trace_dir=self.trace_dir if self.trace_enabled else None,
            run_id=run_id,
            session_id=context.latest_state.session_id if context.latest_state else None,
            stage=name,
            step=step,
        )
        return await _run_sync_in_worker(
            self._execute_stage_sync,
            name,
            step,
            run_id,
            event_recorder,
            execute,
        )

    def _execute_stage_sync(
        self,
        name: str,
        step: int,
        run_id: str,
        event_recorder: HarnessEventRecorder,
        execute: Callable[[], StageResultT],
    ) -> StageResultT:
        trace_dir = self.trace_dir if self.trace_enabled else None
        with harness_event_scope(event_recorder):
            with llm_trace_scope(
                run_id=run_id,
                trace_dir=trace_dir,
                stage=name,
                step=step,
                event_recorder=event_recorder,
                env=self.env,
            ):
                with tool_trace_scope(
                    run_id=run_id,
                    trace_dir=trace_dir,
                    stage=name,
                    step=step,
                    cwd=Path(self.repositories.root_dir),
                    event_recorder=event_recorder,
                ):
                    with context_trace_scope(
                        run_id=run_id,
                        trace_dir=trace_dir,
                        stage=name,
                        step=step,
                        event_recorder=event_recorder,
                    ):
                        return execute()


class DailyPipelineRunner:
    """执行一次完整的 V1 日常 Agent pipeline。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        classifier: InterviewClassifier,
        summarizer: InterviewSummarizer,
        brief_generator: MarkdownBriefGenerator,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
        fail_fast: bool = False,
        discovery_agent: Optional[DiscoveryResearchAgent] = None,
        brief_output_dir: Optional[Path] = None,
        brief_date: Optional[date] = None,
        brief_agent: Optional[BriefWritingAgent] = None,
        trace_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        error_log_dir: Optional[Path] = None,
        hook_manager: Optional[HookManager] = None,
        trace_enabled: bool = True,
        checkpoint_enabled: bool = True,
        resume_run_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tools_enabled: bool = True,
        discovery_fetch_timeout_seconds: Optional[float] = None,
        transcript_fetch_limit: Optional[int] = None,
    ) -> None:
        self.repositories = repositories
        self.classifier = classifier
        self.summarizer = summarizer
        self.brief_generator = brief_generator
        self.env = env
        self.fetch_json_source = fetch_json
        self.fail_fast = fail_fast
        self.discovery_agent = discovery_agent
        self.brief_output_dir = brief_output_dir
        self.brief_date = brief_date
        self.brief_agent = brief_agent
        self.trace_enabled = trace_enabled
        self.checkpoint_enabled = checkpoint_enabled
        self.resume_run_id = resume_run_id
        self.run_id = run_id
        self.tools_enabled = tools_enabled
        self.transcript_fetch_limit = transcript_fetch_limit
        self.discovery_fetch_timeout_seconds = (
            discovery_fetch_timeout_seconds
            if discovery_fetch_timeout_seconds is not None
            else _env_float(
                env,
                "INSIGHTCAST_DISCOVERY_FETCH_TIMEOUT_SECONDS",
                DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS,
            )
        )
        self._current_run_id: Optional[str] = None
        self.error_log_dir = error_log_dir or default_error_log_dir(repositories)
        self.error_log_hook = PipelineErrorLogHook(self.error_log_dir)
        self.hooks = hook_manager or HookManager()
        self.hooks.register(self.error_log_hook)
        self.trace_dir = trace_dir or default_trace_dir(repositories)
        self.checkpoint_dir = checkpoint_dir or default_checkpoint_dir(repositories)
        self._event_loop = asyncio.new_event_loop()
        if self._event_loop is not None:
            asyncio.set_event_loop(self._event_loop)
        self.trace_store = (
            JsonlTraceStore(self.trace_dir)
            if trace_enabled
            else NoopTraceStore()
        )
        self.checkpoint_manager = CheckpointManager(
            FileCheckpointStore(
                self.checkpoint_dir,
                create_dir=checkpoint_enabled or resume_run_id is not None,
            ),
            enabled=checkpoint_enabled,
        )
        self.config = HarnessConfig(
            project_root=Path(self.repositories.root_dir),
            trace=TraceConfig(
                enabled=trace_enabled,
                directory=self.trace_dir,
            ),
            runner=RunnerConfig(
                max_steps=len(PIPELINE_STAGE_NAMES),
                run_timeout_seconds=0,
                checkpoint_enabled=checkpoint_enabled,
            ),
        )

    def run_once(self) -> DailyPipelineRunResult:
        """从发现到 Markdown 日报生成执行一次完整流程。"""

        run = self.repositories.run_records.create(
            RunRecord(
                id=self.run_id or new_id("run"),
                status="running",
                metadata={"stage": "pipeline"},
            )
        )
        self._current_run_id = run.id
        finished_record = False
        try:
            run_result = self._run(run)
            if run_result.failed:
                error_message = _run_result_error_message(run_result)
                self.repositories.run_records.finish(
                    run.id,
                    status="failed",
                    error=error_message,
                    metadata={
                        "stage": "pipeline",
                        "error_log_path": self._error_log_path(),
                    },
                )
                finished_record = True
                raise RuntimeError(error_message)

            result = _result_from_state(
                run_id=run.id,
                state=run_result.state,
                error_log_path=self._error_log_path(),
            )

            self.repositories.run_records.finish(
                run.id,
                status=result.status,
                error=_error_summary(result.stages),
                discovered_count=result.discovered_count,
                accepted_count=result.accepted_count,
                pushed_count=result.pushed_count,
                metadata={
                    "stage": "pipeline",
                    "brief_id": result.brief_id,
                    "markdown_path": result.markdown_path,
                    "error_log_path": result.error_log_path,
                    "stages": [stage.to_dict() for stage in result.stages],
                },
            )
            finished_record = True
            return result
        except Exception as exc:
            if not finished_record:
                self.repositories.run_records.finish(
                    run.id,
                    status="failed",
                    error=str(exc),
                    metadata={
                        "stage": "pipeline",
                        "error_log_path": self._error_log_path(),
                    },
                )
            raise
        finally:
            self._close_hooks(run.id)
            self._close_event_loop()

    def _run(self, run: RunRecord) -> RunResult:
        restored_state = self._restore_state()
        loop = InsightCastDailyPipelineLoop(
            self.repositories,
            classifier=self.classifier,
            summarizer=self.summarizer,
            brief_generator=self.brief_generator,
            env=self.env,
            fetch_json=self.fetch_json_source,
            fail_fast=self.fail_fast,
            discovery_agent=self.discovery_agent,
            brief_output_dir=self.brief_output_dir,
            brief_date=self.brief_date,
            brief_agent=self.brief_agent,
            trace_dir=self.trace_dir,
            trace_enabled=self.trace_enabled,
            restored_state=restored_state,
            tools_enabled=self.tools_enabled,
            discovery_fetch_timeout_seconds=self.discovery_fetch_timeout_seconds,
            transcript_fetch_limit=self.transcript_fetch_limit,
        )
        runner = Runner(
            loop,
            config=self.config,
            trace_store=self.trace_store,
            hooks=self.hooks,
            checkpoint_manager=self.checkpoint_manager,
        )
        return self._run_async(
            runner.run(
                "insightcast_v1_daily",
                run_id=run.id,
                metadata={
                    "pipeline": "insightcast_v1_daily",
                    "stage_count": len(PIPELINE_STAGE_NAMES),
                    "resume_run_id": self.resume_run_id,
                    "resumed_from_run_id": (
                        restored_state.run_id if restored_state is not None else None
                    ),
                    "resumed_from_step": (
                        restored_state.step if restored_state is not None else None
                    ),
                },
            )
        )

    def _restore_state(self) -> Optional[AgentState]:
        if not self.resume_run_id:
            return None
        return self._run_async(
            self.checkpoint_manager.latest_state_for_run(self.resume_run_id)
        )

    def _error_log_path(self) -> Optional[str]:
        if self._current_run_id is None:
            return None
        return str(self.error_log_hook.path_for_run(self._current_run_id))

    def _close_hooks(self, run_id: str) -> None:
        self.error_log_hook.close_run(run_id)

    def _run_async(self, coro):
        if self._event_loop is None:
            raise RuntimeError("Trace/checkpoint 事件循环未配置")
        return self._event_loop.run_until_complete(coro)

    def _close_event_loop(self) -> None:
        if self._event_loop is None:
            return
        if not self._event_loop.is_closed():
            self._event_loop.close()
        self._event_loop = None
        try:
            asyncio.set_event_loop(None)
        except RuntimeError:
            pass


def _stage_result(name: str, result: DomainModel) -> PipelineStageResult:
    payload = result.to_dict()
    return PipelineStageResult(
        name=name,
        status=str(payload.get("status") or "unknown"),
        run_id=payload.get("run_id"),
        metrics=payload,
    )


def _final_status(stages: Tuple[PipelineStageResult, ...]) -> str:
    statuses = {stage.status for stage in stages if stage.status != "skipped"}
    if "failed" in statuses:
        return "failed"
    if "partial" in statuses:
        return "partial"
    return "finished"


def _error_summary(stages: Tuple[PipelineStageResult, ...]) -> Optional[str]:
    failed = [stage.name for stage in stages if stage.status == "failed"]
    if failed:
        return f"失败阶段：{', '.join(failed)}"
    partial = [stage.name for stage in stages if stage.status == "partial"]
    if partial:
        return f"部分完成阶段：{', '.join(partial)}"
    return None


def _state_stage_results(state: AgentState) -> Dict[str, Dict[str, Any]]:
    value = state.variables.get("stage_results") or {}
    if not isinstance(value, dict):
        return {}
    return {str(key): dict(stage) for key, stage in value.items() if isinstance(stage, dict)}


def _state_after_stage(state: AgentState, stage: PipelineStageResult) -> AgentState:
    if stage.status == "skipped":
        return state
    stage_results = _state_stage_results(state)
    stage_results[stage.name] = stage.metrics
    return state._with_updates(
        variables={
            **state.variables,
            "stage_results": stage_results,
            "completed_stages": list(stage_results),
            "last_stage": stage.name,
        }
    )


def _state_with_stage_failure(
    state: AgentState,
    stage_name: str,
    exc: Exception,
) -> AgentState:
    failures = list(state.variables.get("stage_failures") or [])
    failures.append({"stage": stage_name, "error": error_to_payload(exc)})
    return state._with_updates(
        variables={
            **state.variables,
            "stage_failures": failures,
            "last_failed_stage": stage_name,
        }
    )


async def _run_sync_in_worker(
    function: Callable[..., StageResultT],
    *args: Any,
) -> StageResultT:
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(function, *args)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: function(*args))


def _pipeline_result_payload(
    *,
    run_id: str,
    final_status: str,
    stages: Tuple[PipelineStageResult, ...],
    state: AgentState,
) -> Dict[str, Any]:
    stage_results = _state_stage_results(state)
    markdown_stage = stage_results.get("markdown_brief", {})
    promotion_stage = stage_results.get("interview_promotion", {})
    discovery_stage = stage_results.get("discovery", {})
    return {
        "run_id": run_id,
        "status": final_status,
        "stages": [stage.to_dict() for stage in stages],
        "markdown_path": markdown_stage.get("markdown_path"),
        "brief_id": markdown_stage.get("brief_id"),
        "discovered_count": int(discovery_stage.get("discovered_count") or 0),
        "accepted_count": int(promotion_stage.get("created_count") or 0)
        + int(promotion_stage.get("merged_count") or 0),
        "pushed_count": int(markdown_stage.get("pushed_count") or 0),
        "pushed_interview_ids": list(
            markdown_stage.get("pushed_interview_ids") or ()
        ),
    }


def _result_from_state(
    *,
    run_id: str,
    state: AgentState,
    error_log_path: Optional[str],
) -> DailyPipelineRunResult:
    payload = state.variables.get("pipeline_result")
    if not isinstance(payload, Mapping):
        payload = _pipeline_result_payload(
            run_id=run_id,
            final_status=str(state.final_output or "finished"),
            stages=(),
            state=state,
        )

    stages = tuple(
        PipelineStageResult(**dict(stage))
        for stage in payload.get("stages", ())
        if isinstance(stage, Mapping)
    )
    return DailyPipelineRunResult(
        run_id=run_id,
        status=str(payload.get("status") or state.final_output or "finished"),
        stages=stages,
        markdown_path=_optional_str(payload.get("markdown_path")),
        brief_id=_optional_str(payload.get("brief_id")),
        error_log_path=error_log_path,
        discovered_count=int(payload.get("discovered_count") or 0),
        accepted_count=int(payload.get("accepted_count") or 0),
        pushed_count=int(payload.get("pushed_count") or 0),
        pushed_interview_ids=tuple(
            str(value)
            for value in payload.get("pushed_interview_ids") or ()
            if value
        ),
    )


def _run_result_error_message(result: RunResult) -> str:
    error = result.error or {}
    if isinstance(error, Mapping):
        message = error.get("message")
        if message:
            return str(message)
        error_type = error.get("type")
        if error_type:
            return str(error_type)
    return "Pipeline 运行失败。"


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _env_value(env: Optional[Mapping[str, str]], key: str) -> Optional[str]:
    source = env or {}
    value = source.get(key)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _env_csv(
    env: Optional[Mapping[str, str]],
    key: str,
    default: Tuple[str, ...],
) -> Tuple[str, ...]:
    source = env or {}
    value = source.get(key)
    if not value:
        return default
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or default


def _env_flag(env: Optional[Mapping[str, str]], key: str, default: bool) -> bool:
    source = env or {}
    value = source.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_float(env: Optional[Mapping[str, str]], key: str, default: float) -> float:
    source = env or {}
    value = source.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be a float.") from exc


def default_trace_dir(repositories: InsightCastRepositories) -> Path:
    return Path(repositories.root_dir) / "traces"


def default_checkpoint_dir(repositories: InsightCastRepositories) -> Path:
    return Path(repositories.root_dir) / "checkpoints"


def default_error_log_dir(repositories: InsightCastRepositories) -> Path:
    return Path(repositories.root_dir) / "error_logs"


__all__ = [
    "DailyPipelineRunResult",
    "DailyPipelineRunner",
    "PIPELINE_STAGE_NAMES",
    "PipelineStageResult",
    "default_checkpoint_dir",
    "default_error_log_dir",
    "default_trace_dir",
]
