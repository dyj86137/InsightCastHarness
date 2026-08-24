"""将 InsightCast 组件事件桥接回 myHarness RunnerContext。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

from core.event import RunEvent
from core.state import AgentState
from runtime.runner import RunnerContext
from tracing.jsonl import JsonlTraceStore


_CURRENT_HARNESS_EVENT_RECORDER: ContextVar[Optional["HarnessEventRecorder"]] = ContextVar(
    "insightcast_harness_event_recorder",
    default=None,
)


class HarnessEventRecorder:
    """通过 RunnerContext 记录组件事件，并在需要时回落到 trace。"""

    def __init__(
        self,
        *,
        runner_context: Optional[RunnerContext] = None,
        runner_loop: Optional[asyncio.AbstractEventLoop] = None,
        trace_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        stage: Optional[str] = None,
        step: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        self.runner_context = runner_context
        self.runner_loop = runner_loop
        self.trace_dir = trace_dir
        self.run_id = run_id
        self.session_id = session_id
        self.stage = stage
        self.step = step
        self.agent_name = agent_name

    @property
    def enabled(self) -> bool:
        return self.runner_context is not None or self.trace_dir is not None

    async def record_event(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        event = self._enrich_event(event)
        if self.runner_context is not None:
            await self._record_via_runner_context(event, state=state)
            return
        await self._record_via_trace_dir(event)

    def _enrich_event(self, event: RunEvent) -> RunEvent:
        payload = dict(event.payload)
        metadata = dict(event.metadata)
        if self.stage is not None:
            payload.setdefault("stage", self.stage)
        if self.agent_name is not None:
            metadata.setdefault("agent", self.agent_name)
        if payload == event.payload and metadata == event.metadata:
            return event
        return event.clone(payload=payload, metadata=metadata)

    async def _record_via_runner_context(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        if self.runner_loop is None:
            await self.runner_context.record_event(event, state=state)
            return

        if self.runner_loop.is_closed() or not self.runner_loop.is_running():
            await self._record_via_trace_dir(event)
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self.runner_loop:
            await self.runner_context.record_event(event, state=state)
            return

        coroutine = self.runner_context.record_event(event, state=state)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.runner_loop)
        except RuntimeError:
            coroutine.close()
            await self._record_via_trace_dir(event)
            return
        await asyncio.wrap_future(future)

    async def _record_via_trace_dir(self, event: RunEvent) -> None:
        if self.trace_dir is None:
            return
        await JsonlTraceStore(self.trace_dir).append(event)


class HarnessLoopContext:
    """由 HarnessEventRecorder 支撑的最小 myHarness LoopRunContext。"""

    def __init__(self, recorder: Optional[HarnessEventRecorder] = None) -> None:
        self.recorder = recorder

    async def record_event(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        if self.recorder is None:
            return
        await self.recorder.record_event(event, state=state)


@contextmanager
def harness_event_scope(
    recorder: Optional[HarnessEventRecorder],
) -> Iterator[None]:
    token = _CURRENT_HARNESS_EVENT_RECORDER.set(recorder)
    try:
        yield
    finally:
        _CURRENT_HARNESS_EVENT_RECORDER.reset(token)


def current_harness_event_recorder() -> Optional[HarnessEventRecorder]:
    return _CURRENT_HARNESS_EVENT_RECORDER.get()


__all__ = [
    "HarnessEventRecorder",
    "HarnessLoopContext",
    "current_harness_event_recorder",
    "harness_event_scope",
]
