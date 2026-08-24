"""Runner 运行时入口。"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Union

from core.event import RunEvent, RunEventType, error_to_payload
from core.message import Message
from core.result import RunResult
from core.state import AgentState, AgentStatus
from infra.config import HarnessConfig, RunnerConfig, load_config
from infra.exception import RunnerError
from runtime.checkpoint import Checkpoint, CheckpointManager, CheckpointReason
from runtime.hooks import HookContext, HookEvent, HookManager
from runtime.replay import ReplayManager, ReplayMode, ReplayPlan, ReplayResult
from tracing.base import TraceStore
from tracing.jsonl import JsonlTraceStore


RunnerInput = Union[str, Message, Iterable[Message]]


@dataclass
class RunnerContext:
    """传递给 Agent Loop 的运行时上下文。"""

    config: RunnerConfig
    trace_store: TraceStore
    hooks: HookManager
    checkpoint_manager: Optional[CheckpointManager] = None
    latest_state: Optional[AgentState] = field(default=None, init=False)

    async def record_event(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        """写入 Trace、触发 Hook，并按策略创建 Checkpoint。"""

        if state is not None:
            self.latest_state = state
        await self.trace_store.append(event)
        await self.emit_event_hook(event, state=state)
        await self.maybe_checkpoint(event, state=state)

    async def emit_event_hook(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        """根据 RunEvent 类型触发生命周期 Hook。"""

        hook_event = hook_event_from_run_event(event.event_type)
        if hook_event is None:
            return

        await self.hooks.emit(
            hook_event,
            build_hook_context(event, state=state),
        )

    async def maybe_checkpoint(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> Optional[Checkpoint]:
        """根据事件类型决定是否创建 Checkpoint。"""

        if self.checkpoint_manager is None or not self.config.checkpoint_enabled:
            return None
        if state is None:
            return None

        reason = checkpoint_reason_from_event(event.event_type)
        if reason is None:
            return None

        return await self.create_checkpoint(
            state,
            reason=reason,
            source_event_id=event.id,
            metadata={"source_event_type": event.event_type.value},
        )

    async def create_checkpoint(
        self,
        state: AgentState,
        *,
        reason: CheckpointReason,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Checkpoint]:
        """创建 Checkpoint，并写入 checkpoint_created 事件。"""

        if self.checkpoint_manager is None or not self.config.checkpoint_enabled:
            return None

        await self.hooks.emit(
            HookEvent.BEFORE_CHECKPOINT_SAVE,
            HookContext(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                state=state,
                payload=metadata or {},
            ),
        )

        try:
            checkpoint = await self.checkpoint_manager.create_if_enabled(
                state,
                reason=reason,
                source_event_id=source_event_id,
                metadata=metadata,
            )
        except Exception as exc:
            await self.hooks.emit(
                HookEvent.CHECKPOINT_FAILED,
                HookContext(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    step=state.step,
                    state=state,
                    payload=metadata or {},
                    error=error_to_payload(exc),
                ),
            )
            return None

        if checkpoint is None:
            return None

        checkpoint_event = checkpoint.to_event()
        await self.trace_store.append(checkpoint_event)
        await self.hooks.emit(
            HookEvent.CHECKPOINT_CREATED,
            build_hook_context(
                checkpoint_event,
                state=state,
                checkpoint_id=checkpoint.id,
            ),
        )
        return checkpoint


class Runner:
    """Agent 运行统一入口。

    Runner 负责生命周期、Trace、Hook、异常收敛和结果包装；
    具体 Agent 策略由传入的 loop 对象负责。
    """

    def __init__(
        self,
        loop: Any,
        *,
        config: Optional[HarnessConfig] = None,
        trace_store: Optional[TraceStore] = None,
        hooks: Optional[HookManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        replay_manager: Optional[ReplayManager] = None,
    ) -> None:
        self.loop = loop
        self.config = config or load_config()
        self.trace_store = trace_store or JsonlTraceStore.from_config(self.config.trace)
        self.hooks = hooks or HookManager()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            enabled=self.config.runner.checkpoint_enabled
        )
        self.replay_manager = replay_manager or ReplayManager(
            self.trace_store,
            checkpoint_store=self.checkpoint_manager.store,
        )
        self.context = RunnerContext(
            config=self.config.runner,
            trace_store=self.trace_store,
            hooks=self.hooks,
            checkpoint_manager=self.checkpoint_manager,
        )

    async def run(
        self,
        input_data: RunnerInput,
        *,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """创建 AgentState 并执行一次完整 run。"""

        started_at = time.perf_counter()
        state = create_initial_state(
            input_data,
            run_id=run_id,
            session_id=session_id,
            max_steps=self.config.runner.max_steps,
            metadata=metadata,
        )

        start_event = RunEvent.run_started(
            run_id=state.run_id,
            session_id=state.session_id,
            payload={
                "max_steps": state.max_steps,
                "input_message_count": len(state.messages),
            },
        )
        return await self._execute_state(state, started_at, start_event=start_event)

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """从指定 Checkpoint 恢复 AgentState，并继续执行 Loop。"""

        started_at = time.perf_counter()
        try:
            state = await self.checkpoint_manager.restore_state(checkpoint_id)
        except Exception as exc:
            await self.hooks.emit(
                HookEvent.RESUME_FAILED,
                HookContext(
                    run_id=checkpoint_id,
                    checkpoint_id=checkpoint_id,
                    payload=metadata or {},
                    error=error_to_payload(exc),
                ),
            )
            raise

        resumed_state = prepare_resumed_state(state, metadata=metadata)
        await self.hooks.emit(
            HookEvent.BEFORE_RESUME,
            HookContext(
                run_id=resumed_state.run_id,
                session_id=resumed_state.session_id,
                step=resumed_state.step,
                state=resumed_state,
                checkpoint_id=checkpoint_id,
                payload=metadata or {},
            ),
        )

        start_event = RunEvent.run_started(
            run_id=resumed_state.run_id,
            session_id=resumed_state.session_id,
            payload={
                "resumed": True,
                "checkpoint_id": checkpoint_id,
                "max_steps": resumed_state.max_steps,
            },
        )
        result = await self._execute_state(resumed_state, started_at, start_event=start_event)

        hook_event = HookEvent.AFTER_RESUME if result.succeeded else HookEvent.RESUME_FAILED
        await self.hooks.emit(
            hook_event,
            HookContext(
                run_id=result.run_id,
                session_id=result.session_id,
                step=result.state.step,
                state=result.state,
                checkpoint_id=checkpoint_id,
                payload={"status": result.status.value},
                error=result.error,
            ),
        )
        return result

    async def _execute_state(
        self,
        state: AgentState,
        started_at: float,
        *,
        start_event: Optional[RunEvent] = None,
    ) -> RunResult:
        """执行一个已经创建好的 AgentState，并包装为 RunResult。"""

        self.context.latest_state = state
        try:
            if start_event is not None:
                await self.context.record_event(start_event, state=state)

            final_state = await self._run_loop_with_timeout(state)
            final_state = ensure_terminal_state(final_state)
            return await self._finalize_state(final_state, started_at)
        except Exception as exc:
            failure_state = (self.context.latest_state or state).mark_failed(error_to_payload(exc))
            await self.context.record_event(
                RunEvent.run_failed(
                    run_id=failure_state.run_id,
                    session_id=failure_state.session_id,
                    error=exc,
                ),
                state=failure_state,
            )
            events = await self.trace_store.list_events(failure_state.run_id)
            return RunResult.failure(
                state=failure_state,
                error=exc,
                events=events,
                duration_ms=elapsed_ms(started_at),
            )

    async def _finalize_state(self, final_state: AgentState, started_at: float) -> RunResult:
        """根据终态 AgentState 记录终态事件并创建 RunResult。"""

        if final_state.status == AgentStatus.FAILED:
            loop_error = RunnerError(
                "Agent loop returned failed state.",
                details={"state_error": final_state.error},
            )
            await self.context.record_event(
                RunEvent.run_failed(
                    run_id=final_state.run_id,
                    session_id=final_state.session_id,
                    error=loop_error,
                ),
                state=final_state,
            )
            events = await self.trace_store.list_events(final_state.run_id)
            return RunResult.failure(
                state=final_state,
                error=loop_error,
                events=events,
                duration_ms=elapsed_ms(started_at),
            )

        if final_state.status == AgentStatus.CANCELLED:
            await self.context.record_event(
                RunEvent.create(
                    run_id=final_state.run_id,
                    session_id=final_state.session_id,
                    event_type=RunEventType.RUN_CANCELLED,
                    message="Run cancelled.",
                ),
                state=final_state,
            )
            events = await self.trace_store.list_events(final_state.run_id)
            return RunResult.cancelled(
                state=final_state,
                events=events,
                duration_ms=elapsed_ms(started_at),
            )

        await self.context.record_event(
            RunEvent.run_completed(
                run_id=final_state.run_id,
                session_id=final_state.session_id,
                payload={
                    "status": final_state.status.value,
                    "final_output": final_state.final_output,
                },
            ),
            state=final_state,
        )

        events = await self.trace_store.list_events(final_state.run_id)
        return RunResult.success(
            state=final_state,
            events=events,
            duration_ms=elapsed_ms(started_at),
        )

    async def _run_loop_with_timeout(self, state: AgentState) -> AgentState:
        """按 RunnerConfig.run_timeout_seconds 执行 Agent Loop。"""

        timeout = self.config.runner.run_timeout_seconds
        if timeout is None or timeout <= 0:
            return await self._run_loop(state)
        try:
            return await asyncio.wait_for(self._run_loop(state), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RunnerError(
                "Runner execution timed out.",
                details={"run_id": state.run_id, "timeout_seconds": timeout},
                cause=exc,
            ) from exc

    async def build_replay_plan(
        self,
        run_id: str,
        *,
        mode: ReplayMode = ReplayMode.ANALYZE,
        replay_run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        start_event_id: Optional[str] = None,
        end_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReplayPlan:
        """基于历史 Trace 构建 ReplayPlan。"""

        return await self.replay_manager.build_plan(
            run_id,
            mode=mode,
            replay_run_id=replay_run_id,
            checkpoint_id=checkpoint_id,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            metadata=metadata,
        )

    async def replay_run(
        self,
        run_id: str,
        *,
        mode: ReplayMode = ReplayMode.EVENT_LOG,
        replay_run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        start_event_id: Optional[str] = None,
        end_event_id: Optional[str] = None,
        target_trace_store: Optional[TraceStore] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReplayResult:
        """构建并执行某个 run 的 ReplayPlan。"""

        plan = await self.build_replay_plan(
            run_id,
            mode=mode,
            replay_run_id=replay_run_id,
            checkpoint_id=checkpoint_id,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            metadata=metadata,
        )
        return await self.replay_plan(plan, target_trace_store=target_trace_store)

    async def replay_plan(
        self,
        plan: ReplayPlan,
        *,
        target_trace_store: Optional[TraceStore] = None,
    ) -> ReplayResult:
        """执行已经构建好的 ReplayPlan，并触发 Replay Hook。"""

        await self.hooks.emit(
            HookEvent.BEFORE_REPLAY,
            HookContext(
                run_id=plan.source_run_id,
                replay_plan_id=plan.id,
                checkpoint_id=plan.checkpoint_id,
                payload={
                    "mode": plan.mode.value,
                    "event_count": plan.event_count,
                    "replay_run_id": plan.replay_run_id,
                },
            ),
        )

        result = await self.replay_manager.replay(
            plan,
            target_trace_store=target_trace_store,
        )
        hook_event = HookEvent.AFTER_REPLAY if result.succeeded else HookEvent.REPLAY_FAILED
        await self.hooks.emit(
            hook_event,
            HookContext(
                run_id=result.plan.source_run_id,
                replay_plan_id=result.plan.id,
                checkpoint_id=result.plan.checkpoint_id,
                payload={
                    "mode": result.plan.mode.value,
                    "status": result.status.value,
                    "replayed_event_count": len(result.replayed_events),
                    "replay_run_id": result.plan.replay_run_id,
                },
                error=result.error,
            ),
        )
        return result

    async def _run_loop(self, state: AgentState) -> AgentState:
        """调用 Agent Loop，并校验返回值。"""

        run_method = getattr(self.loop, "run", None)
        if run_method is None or not callable(run_method):
            raise RunnerError(
                "Runner loop must provide an async run(state, context) method.",
                details={"loop_type": self.loop.__class__.__name__},
            )

        result = run_method(state, self.context)
        if inspect.isawaitable(result):
            result = await result
        else:
            raise RunnerError(
                "Runner loop run method must be awaitable.",
                details={"loop_type": self.loop.__class__.__name__},
            )

        if not isinstance(result, AgentState):
            raise RunnerError(
                "Runner loop must return AgentState.",
                details={
                    "loop_type": self.loop.__class__.__name__,
                    "actual_type": result.__class__.__name__,
                },
            )
        if result.run_id != state.run_id:
            raise RunnerError(
                "Runner loop returned state from another run.",
                details={
                    "expected_run_id": state.run_id,
                    "actual_run_id": result.run_id,
                },
            )
        return result


def normalize_runner_input(input_data: RunnerInput) -> Iterable[Message]:
    """把 Runner 输入转换成 Message 序列。"""

    if isinstance(input_data, str):
        return (Message.user(input_data),)
    if isinstance(input_data, Message):
        return (input_data,)
    return tuple(input_data)


def create_initial_state(
    input_data: RunnerInput,
    *,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """根据 Runner 输入创建初始 AgentState。"""

    messages = tuple(normalize_runner_input(input_data))
    if run_id:
        return AgentState(
            run_id=run_id,
            session_id=session_id,
            messages=messages,
            max_steps=max_steps,
            metadata=metadata or {},
        ).mark_running()

    return AgentState.new(
        session_id=session_id,
        messages=messages,
        max_steps=max_steps,
        metadata=metadata or {},
    ).mark_running()


def prepare_resumed_state(
    state: AgentState,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """把 Checkpoint 中的状态转换为可继续运行的状态。"""

    merged_metadata = dict(state.metadata)
    merged_metadata.update(metadata or {})
    merged_metadata["resumed_from_checkpoint"] = True
    return state.clone(
        status=AgentStatus.RUNNING,
        error=None,
        final_output=None,
        metadata=merged_metadata,
    )


def ensure_terminal_state(state: AgentState) -> AgentState:
    """确保 Loop 返回的是终态 AgentState。"""

    if state.is_terminal:
        return state

    final_output = state.final_output
    if final_output is None and state.last_assistant_message is not None:
        final_output = state.last_assistant_message.content
    if final_output is None:
        final_output = ""

    return state.mark_completed(final_output)


def build_hook_context(
    event: RunEvent,
    *,
    state: Optional[AgentState] = None,
    checkpoint_id: Optional[str] = None,
    replay_plan_id: Optional[str] = None,
) -> HookContext:
    """根据 RunEvent 构建 HookContext。"""

    return HookContext(
        run_id=event.run_id,
        session_id=event.session_id,
        step=event.step,
        state=state,
        event=event,
        checkpoint_id=checkpoint_id,
        replay_plan_id=replay_plan_id,
        payload=event.payload,
        error=event.error,
    )


def hook_event_from_run_event(event_type: RunEventType) -> Optional[HookEvent]:
    """把 RunEventType 映射为 HookEvent。"""

    mapping = {
        RunEventType.RUN_STARTED: HookEvent.RUN_STARTED,
        RunEventType.RUN_COMPLETED: HookEvent.RUN_COMPLETED,
        RunEventType.RUN_FAILED: HookEvent.RUN_FAILED,
        RunEventType.STEP_STARTED: HookEvent.STEP_STARTED,
        RunEventType.STEP_COMPLETED: HookEvent.STEP_COMPLETED,
        RunEventType.STEP_FAILED: HookEvent.STEP_FAILED,
        RunEventType.LLM_REQUESTED: HookEvent.BEFORE_LLM_CALL,
        RunEventType.LLM_RESPONDED: HookEvent.AFTER_LLM_CALL,
        RunEventType.LLM_FAILED: HookEvent.LLM_CALL_FAILED,
        RunEventType.TOOL_CALL_STARTED: HookEvent.BEFORE_TOOL_CALL,
        RunEventType.TOOL_CALL_COMPLETED: HookEvent.AFTER_TOOL_CALL,
        RunEventType.TOOL_CALL_FAILED: HookEvent.TOOL_CALL_FAILED,
        RunEventType.CONTEXT_BUILT: HookEvent.CONTEXT_BUILT,
        RunEventType.CHECKPOINT_CREATED: HookEvent.CHECKPOINT_CREATED,
    }
    return mapping.get(event_type)


def checkpoint_reason_from_event(event_type: RunEventType) -> Optional[CheckpointReason]:
    """把运行事件映射为需要创建 Checkpoint 的原因。"""

    mapping = {
        RunEventType.RUN_STARTED: CheckpointReason.RUN_STARTED,
        RunEventType.STEP_COMPLETED: CheckpointReason.STEP_COMPLETED,
        RunEventType.RUN_FAILED: CheckpointReason.RUN_FAILED,
        RunEventType.RUN_COMPLETED: CheckpointReason.RUN_COMPLETED,
    }
    return mapping.get(event_type)


def elapsed_ms(started_at: float) -> float:
    """计算从 started_at 到当前的毫秒耗时。"""

    return (time.perf_counter() - started_at) * 1000


__all__ = [
    "Runner",
    "RunnerContext",
    "RunnerInput",
    "build_hook_context",
    "checkpoint_reason_from_event",
    "create_initial_state",
    "elapsed_ms",
    "ensure_terminal_state",
    "hook_event_from_run_event",
    "normalize_runner_input",
    "prepare_resumed_state",
]
