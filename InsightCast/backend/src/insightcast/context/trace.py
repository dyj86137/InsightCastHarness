"""InsightCast 上下文构建的 trace 辅助工具。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from core.event import RunEvent, RunEventType
from insightcast.harness_events import HarnessEventRecorder


class ContextTraceContext:
    """当前 pipeline 中用于上下文构建事件的 trace 上下文。"""

    def __init__(
        self,
        *,
        run_id: str,
        trace_dir: Optional[Path],
        stage: str,
        step: int,
        event_recorder: Optional[HarnessEventRecorder] = None,
    ) -> None:
        self.run_id = run_id
        self.trace_dir = trace_dir
        self.stage = stage
        self.step = step
        self.event_recorder = event_recorder


_CURRENT_CONTEXT_TRACE_CONTEXT: ContextVar[Optional[ContextTraceContext]] = ContextVar(
    "insightcast_context_trace_context",
    default=None,
)


@contextmanager
def context_trace_scope(
    *,
    run_id: str,
    trace_dir: Optional[Path],
    stage: str,
    step: int,
    event_recorder: Optional[HarnessEventRecorder] = None,
) -> Iterator[None]:
    """为当前代码块中的上下文构建工作设置 trace 上下文。"""

    recorder = event_recorder or (
        HarnessEventRecorder(trace_dir=trace_dir) if trace_dir is not None else None
    )
    token = _CURRENT_CONTEXT_TRACE_CONTEXT.set(
        ContextTraceContext(
            run_id=run_id,
            trace_dir=trace_dir,
            stage=stage,
            step=step,
            event_recorder=recorder,
        )
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT_TRACE_CONTEXT.reset(token)


async def append_context_built_event(
    *,
    name: str,
    payload: Dict[str, Any],
) -> None:
    """当存在 trace 上下文时，追加 myHarness context_built 事件。"""

    trace_context = _CURRENT_CONTEXT_TRACE_CONTEXT.get()
    if trace_context is None or trace_context.event_recorder is None:
        return
    event_payload = {
        "stage": trace_context.stage,
        "context_name": name,
    }
    event_payload.update(payload)
    await trace_context.event_recorder.record_event(
        RunEvent.create(
            run_id=trace_context.run_id,
            event_type=RunEventType.CONTEXT_BUILT,
            step=trace_context.step,
            message="上下文已构建。",
            payload=event_payload,
        )
    )


__all__ = [
    "ContextTraceContext",
    "append_context_built_event",
    "context_trace_scope",
]
