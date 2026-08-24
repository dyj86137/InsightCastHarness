"""Trace 存储抽象协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterable, List, Optional, Tuple

from pydantic import Field

from core.event import RunEvent, RunEventType
from infra.exception import TraceError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


class RunTrace(SerializableModel):
    """单次 run 的完整事件集合。"""

    run_id: str
    events: Tuple[RunEvent, ...] = Field(default_factory=tuple)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Trace 中的事件归属一致。"""

        if not self.run_id:
            raise HarnessValidationError("RunTrace requires run_id.")

        invalid_event_ids = [
            event.id for event in self.events if event.run_id != self.run_id
        ]
        if invalid_event_ids:
            raise HarnessValidationError(
                "RunTrace contains events from another run.",
                details={
                    "run_id": self.run_id,
                    "invalid_event_ids": invalid_event_ids,
                },
            )

    @property
    def event_count(self) -> int:
        """返回事件数量。"""

        return len(self.events)

    @property
    def first_event(self) -> Optional[RunEvent]:
        """返回第一条事件。"""

        if not self.events:
            return None
        return self.events[0]

    @property
    def last_event(self) -> Optional[RunEvent]:
        """返回最后一条事件。"""

        if not self.events:
            return None
        return self.events[-1]

    @property
    def terminal_event(self) -> Optional[RunEvent]:
        """返回最后一条终态事件。"""

        for event in reversed(self.events):
            if event.is_terminal:
                return event
        return None

    @property
    def is_terminal(self) -> bool:
        """判断 Trace 是否已经包含 run 终态事件。"""

        return self.terminal_event is not None

    def filter_by_type(self, event_type: RunEventType) -> Tuple[RunEvent, ...]:
        """按事件类型过滤事件。"""

        return tuple(event for event in self.events if event.event_type == event_type)

    def append(self, event: RunEvent) -> "RunTrace":
        """返回追加事件后的新 Trace。"""

        if event.run_id != self.run_id:
            raise TraceError(
                "Cannot append event from another run.",
                details={
                    "trace_run_id": self.run_id,
                    "event_run_id": event.run_id,
                    "event_id": event.id,
                },
            )
        return self.__class__(run_id=self.run_id, events=self.events + (event,))


class TraceStore(ABC):
    """Trace 存储后端的统一抽象接口。"""

    @abstractmethod
    async def append(self, event: RunEvent) -> None:
        """追加一条事件。"""

    async def append_many(self, events: Iterable[RunEvent]) -> None:
        """批量追加事件。"""

        for event in events:
            await self.append(event)

    @abstractmethod
    async def list_events(self, run_id: str) -> List[RunEvent]:
        """列出某个 run 的所有事件。"""

    async def load_run(self, run_id: str) -> RunTrace:
        """加载某个 run 的完整 Trace。"""

        events = await self.list_events(run_id)
        return RunTrace(run_id=run_id, events=tuple(events))

    async def iter_events(self, run_id: str) -> AsyncIterator[RunEvent]:
        """迭代某个 run 的所有事件。"""

        for event in await self.list_events(run_id):
            yield event

    async def last_event(self, run_id: str) -> Optional[RunEvent]:
        """返回某个 run 的最后一条事件。"""

        events = await self.list_events(run_id)
        if not events:
            return None
        return events[-1]

    async def event_count(self, run_id: str) -> int:
        """返回某个 run 的事件数量。"""

        events = await self.list_events(run_id)
        return len(events)


__all__ = [
    "RunTrace",
    "TraceStore",
]
