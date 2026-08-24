"""myHarness ToolExecutor 适配器和 trace 辅助工具。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.tool import ToolCall, ToolResult
from tools.base import BaseTool, ToolExecutionContext
from tools.executor import ToolExecutor
from insightcast.harness_events import HarnessEventRecorder


class ToolTraceContext:
    """当前 pipeline 中用于 myHarness 工具调用的 trace 上下文。"""

    def __init__(
        self,
        *,
        run_id: str,
        trace_dir: Optional[Path],
        stage: str,
        step: int,
        cwd: Optional[Path] = None,
        event_recorder: Optional[HarnessEventRecorder] = None,
    ) -> None:
        self.run_id = run_id
        self.trace_dir = trace_dir
        self.stage = stage
        self.step = step
        self.cwd = cwd or Path.cwd()
        self.event_recorder = event_recorder


_CURRENT_TOOL_TRACE_CONTEXT: ContextVar[Optional[ToolTraceContext]] = ContextVar(
    "insightcast_tool_trace_context",
    default=None,
)


class TracingToolExecutor:
    """会发出 myHarness 工具 trace 事件的 ToolExecutor 包装器。"""

    def __init__(self, inner: ToolExecutor) -> None:
        self.inner = inner

    @classmethod
    def from_tools(cls, tools: Iterable[BaseTool]) -> "TracingToolExecutor":
        return cls(ToolExecutor.from_tools(tools))

    def register(self, tool: BaseTool, *, override: bool = False) -> None:
        self.inner.register(tool, override=override)

    async def execute(
        self,
        tool_call: ToolCall,
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        trace_context = _CURRENT_TOOL_TRACE_CONTEXT.get()
        execution_context = _context_with_trace_defaults(context, trace_context)
        should_trace = _should_trace(trace_context, execution_context)

        if should_trace:
            await _append_tool_event(
                trace_context,
                _tool_event(
                    execution_context,
                    tool_call,
                    RunEventType.TOOL_CALL_STARTED,
                    message="工具调用已开始。",
                ),
            )

        try:
            result = await self.inner.execute(tool_call, execution_context)
        except Exception as exc:
            if should_trace:
                await _append_tool_event(
                    trace_context,
                    _tool_event(
                        execution_context,
                        tool_call,
                        RunEventType.TOOL_CALL_FAILED,
                        message="工具调用失败。",
                        level=RunEventLevel.ERROR,
                        error=error_to_payload(exc),
                    ),
                )
            raise

        if should_trace:
            if result.is_error:
                await _append_tool_event(
                    trace_context,
                    _tool_event(
                        execution_context,
                        tool_call,
                        RunEventType.TOOL_CALL_FAILED,
                        message="工具调用失败。",
                        level=RunEventLevel.ERROR,
                        result=result,
                        error=result.error,
                    ),
                )
            else:
                await _append_tool_event(
                    trace_context,
                    _tool_event(
                        execution_context,
                        tool_call,
                        RunEventType.TOOL_CALL_COMPLETED,
                        message="工具调用已完成。",
                        result=result,
                    ),
                )

        return result


@contextmanager
def tool_trace_scope(
    *,
    run_id: str,
    trace_dir: Optional[Path],
    stage: str,
    step: int,
    cwd: Optional[Path] = None,
    event_recorder: Optional[HarnessEventRecorder] = None,
) -> Iterator[None]:
    """为当前代码块中的 myHarness ToolExecutor 调用设置 trace 上下文。"""

    recorder = event_recorder or (
        HarnessEventRecorder(trace_dir=trace_dir) if trace_dir is not None else None
    )
    token = _CURRENT_TOOL_TRACE_CONTEXT.set(
        ToolTraceContext(
            run_id=run_id,
            trace_dir=trace_dir,
            stage=stage,
            step=step,
            cwd=cwd,
            event_recorder=recorder,
        )
    )
    try:
        yield
    finally:
        _CURRENT_TOOL_TRACE_CONTEXT.reset(token)


def current_tool_execution_context(
    *,
    allow_mutation: bool = False,
    timeout_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolExecutionContext:
    """基于当前 pipeline 工具 trace 上下文构建 ToolExecutionContext。"""

    trace_context = _CURRENT_TOOL_TRACE_CONTEXT.get()
    execution_metadata = dict(metadata or {})
    if trace_context is not None:
        execution_metadata.setdefault("stage", trace_context.stage)
        return ToolExecutionContext(
            run_id=trace_context.run_id,
            step=trace_context.step,
            cwd=trace_context.cwd,
            allow_mutation=allow_mutation,
            timeout_seconds=timeout_seconds,
            metadata=execution_metadata,
        )
    return ToolExecutionContext(
        allow_mutation=allow_mutation,
        timeout_seconds=timeout_seconds,
        metadata=execution_metadata,
    )


def run_tool_sync(coro):
    """在 InsightCast 同步 runner 代码中运行工具协程。"""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("Synchronous InsightCast tool adapter cannot run inside an event loop")


def _context_with_trace_defaults(
    context: Optional[ToolExecutionContext],
    trace_context: Optional[ToolTraceContext],
) -> Optional[ToolExecutionContext]:
    if trace_context is None:
        return context

    metadata = dict(context.metadata) if context is not None else {}
    metadata.setdefault("stage", trace_context.stage)
    if context is None:
        return ToolExecutionContext(
            run_id=trace_context.run_id,
            step=trace_context.step,
            cwd=trace_context.cwd,
            allow_mutation=False,
            metadata=metadata,
        )
    return context.clone(
        run_id=context.run_id or trace_context.run_id,
        step=context.step if context.step is not None else trace_context.step,
        cwd=context.cwd or trace_context.cwd,
        metadata=metadata,
    )


def _should_trace(
    trace_context: Optional[ToolTraceContext],
    execution_context: Optional[ToolExecutionContext],
) -> bool:
    return (
        trace_context is not None
        and trace_context.event_recorder is not None
        and trace_context.event_recorder.enabled
        and execution_context is not None
        and execution_context.run_id is not None
    )


def _tool_event(
    context: ToolExecutionContext,
    tool_call: ToolCall,
    event_type: RunEventType,
    *,
    message: str,
    level: RunEventLevel = RunEventLevel.INFO,
    result: Optional[ToolResult] = None,
    error: Optional[Dict[str, Any]] = None,
) -> RunEvent:
    payload = _tool_payload(context, tool_call)
    if result is not None:
        payload.update(_result_payload(result))
    return RunEvent.create(
        run_id=str(context.run_id),
        event_type=event_type,
        step=context.step,
        level=level,
        message=message,
        payload=payload,
        error=error,
    )


def _tool_payload(context: ToolExecutionContext, tool_call: ToolCall) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        "argument_keys": sorted(str(key) for key in tool_call.arguments),
    }
    stage = context.metadata.get("stage")
    if stage:
        payload["stage"] = stage
    if tool_call.metadata:
        payload["tool_metadata"] = dict(tool_call.metadata)
    return payload


def _result_payload(result: ToolResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": result.status.value,
        "duration_ms": result.duration_ms,
    }
    if result.output:
        payload["output"] = _trim_text(result.output)
    if result.metadata:
        payload["result_metadata"] = dict(result.metadata)
    return payload


async def _append_tool_event(
    trace_context: Optional[ToolTraceContext],
    event: RunEvent,
) -> None:
    if trace_context is None or trace_context.event_recorder is None:
        return
    await trace_context.event_recorder.record_event(event)


def _trim_text(value: str, *, limit: int = 300) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


__all__ = [
    "ToolTraceContext",
    "TracingToolExecutor",
    "current_tool_execution_context",
    "run_tool_sync",
    "tool_trace_scope",
]
