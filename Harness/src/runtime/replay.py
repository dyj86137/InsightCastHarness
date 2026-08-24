"""运行失败 Replay 能力。

Replay 模块负责从 Trace 中读取历史事件，生成可审计的重放计划，并支持两类基础动作：
1. 事件日志重放：把历史 RunEvent 复制到一个新的 replay run，方便排查和对比。
2. Checkpoint 恢复：从最近或指定 Checkpoint 还原 AgentState，交给 Runner 后续续跑。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from pydantic import Field

from core.event import RunEvent, error_to_payload
from core.state import AgentState
from infra.exception import RunnerError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from runtime.checkpoint import Checkpoint, CheckpointStore
from tracing.base import TraceStore


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_replay_plan_id() -> str:
    return f"replay_plan_{uuid4().hex}"


def _new_replay_run_id(source_run_id: str) -> str:
    return f"replay_{source_run_id}_{uuid4().hex}"


class ReplayMode(str, Enum):
    """Replay 执行模式。"""

    ANALYZE = "analyze"
    EVENT_LOG = "event_log"
    CHECKPOINT_RESTORE = "checkpoint_restore"


class ReplayStatus(str, Enum):
    """Replay 执行结果状态。"""

    COMPLETED = "completed"
    FAILED = "failed"


class ReplayError(RunnerError):
    """Replay 计划构建或执行失败。"""


class ReplayPlan(SerializableModel):
    """一次 Replay 的静态计划。"""

    id: str = Field(default_factory=_new_replay_plan_id)
    source_run_id: str
    mode: ReplayMode = ReplayMode.ANALYZE
    events: Tuple[RunEvent, ...]
    replay_run_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    start_event_id: Optional[str] = None
    end_event_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 ReplayPlan 的事件归属和边界。"""

        if not self.source_run_id or not self.source_run_id.strip():
            raise HarnessValidationError("ReplayPlan requires source_run_id.")
        if not self.events:
            raise HarnessValidationError(
                "ReplayPlan requires at least one event.",
                details={"source_run_id": self.source_run_id},
            )

        invalid_events = [
            event.id for event in self.events if event.run_id != self.source_run_id
        ]
        if invalid_events:
            raise HarnessValidationError(
                "ReplayPlan contains events from another run.",
                details={
                    "source_run_id": self.source_run_id,
                    "invalid_event_ids": invalid_events,
                },
            )

        event_ids = {event.id for event in self.events}
        if self.start_event_id and self.start_event_id not in event_ids:
            raise HarnessValidationError(
                "ReplayPlan start_event_id is not included in selected events.",
                details={"start_event_id": self.start_event_id},
            )
        if self.end_event_id and self.end_event_id not in event_ids:
            raise HarnessValidationError(
                "ReplayPlan end_event_id is not included in selected events.",
                details={"end_event_id": self.end_event_id},
            )

    @property
    def event_count(self) -> int:
        """返回计划中的事件数量。"""

        return len(self.events)

    @property
    def first_event(self) -> RunEvent:
        """返回计划中的第一条事件。"""

        return self.events[0]

    @property
    def last_event(self) -> RunEvent:
        """返回计划中的最后一条事件。"""

        return self.events[-1]

    @property
    def terminal_event(self) -> Optional[RunEvent]:
        """返回计划中最后一条 run 终态事件。"""

        for event in reversed(self.events):
            if event.is_terminal:
                return event
        return None

    def with_replay_run_id(self, replay_run_id: str) -> "ReplayPlan":
        """返回绑定 replay_run_id 后的新计划。"""

        return self.clone(replay_run_id=replay_run_id)


class ReplayResult(SerializableModel):
    """Replay 执行后的标准结果。"""

    plan: ReplayPlan
    status: ReplayStatus
    replayed_events: Tuple[RunEvent, ...] = Field(default_factory=tuple)
    restored_state: Optional[AgentState] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 ReplayResult 的基础规则。"""

        if self.status == ReplayStatus.FAILED and not self.error:
            raise HarnessValidationError(
                "Failed ReplayResult requires error payload.",
                details={"plan_id": self.plan.id},
            )
        if self.status == ReplayStatus.COMPLETED and self.error:
            raise HarnessValidationError(
                "Completed ReplayResult cannot carry error payload.",
                details={"plan_id": self.plan.id},
            )
        for event in self.replayed_events:
            if self.plan.replay_run_id and event.run_id != self.plan.replay_run_id:
                raise HarnessValidationError(
                    "ReplayResult contains event from unexpected replay run.",
                    details={
                        "plan_id": self.plan.id,
                        "expected_run_id": self.plan.replay_run_id,
                        "actual_run_id": event.run_id,
                        "event_id": event.id,
                    },
                )

    @property
    def succeeded(self) -> bool:
        """判断 Replay 是否成功。"""

        return self.status == ReplayStatus.COMPLETED

    @classmethod
    def completed(
        cls,
        *,
        plan: ReplayPlan,
        replayed_events: Optional[Iterable[RunEvent]] = None,
        restored_state: Optional[AgentState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReplayResult":
        """创建成功结果。"""

        return cls(
            plan=plan,
            status=ReplayStatus.COMPLETED,
            replayed_events=tuple(replayed_events or ()),
            restored_state=restored_state,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        *,
        plan: ReplayPlan,
        error: BaseException,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReplayResult":
        """创建失败结果。"""

        return cls(
            plan=plan,
            status=ReplayStatus.FAILED,
            error=error_to_payload(error),
            metadata=metadata or {},
        )


class ReplayManager:
    """Replay 的统一编排入口。"""

    def __init__(
        self,
        trace_store: TraceStore,
        *,
        checkpoint_store: Optional[CheckpointStore] = None,
    ) -> None:
        self.trace_store = trace_store
        self.checkpoint_store = checkpoint_store

    async def build_plan(
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
        """从 TraceStore 读取历史事件并构建 ReplayPlan。"""

        events = await self.trace_store.list_events(run_id)
        if not events:
            raise ReplayError(
                "Cannot build replay plan without trace events.",
                details={"run_id": run_id},
            )

        selected_events = select_event_range(
            events,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
        )
        resolved_checkpoint_id = await self._resolve_checkpoint_id(run_id, checkpoint_id)

        return ReplayPlan(
            source_run_id=run_id,
            mode=mode,
            events=tuple(selected_events),
            replay_run_id=replay_run_id,
            checkpoint_id=resolved_checkpoint_id,
            start_event_id=start_event_id,
            end_event_id=end_event_id,
            metadata={
                "event_type_counts": count_events_by_type(selected_events),
                "selected_event_count": len(selected_events),
                **(metadata or {}),
            },
        )

    async def replay(
        self,
        plan: ReplayPlan,
        *,
        target_trace_store: Optional[TraceStore] = None,
    ) -> ReplayResult:
        """执行 ReplayPlan。

        ANALYZE 只返回计划摘要；EVENT_LOG 会把事件复制到 replay run；
        CHECKPOINT_RESTORE 会从 checkpoint 还原 AgentState。
        """

        try:
            if plan.mode == ReplayMode.ANALYZE:
                return ReplayResult.completed(
                    plan=plan,
                    metadata={"event_count": plan.event_count},
                )

            if plan.mode == ReplayMode.CHECKPOINT_RESTORE:
                restored_state = await self.restore_state_for_plan(plan)
                return ReplayResult.completed(
                    plan=plan,
                    restored_state=restored_state,
                    metadata={"checkpoint_id": plan.checkpoint_id},
                )

            if plan.mode == ReplayMode.EVENT_LOG:
                return await self._replay_event_log(plan, target_trace_store)

            raise ReplayError(
                "Unsupported replay mode.",
                details={"mode": plan.mode.value},
            )
        except Exception as exc:
            return ReplayResult.failed(plan=plan, error=exc)

    async def restore_state_for_plan(self, plan: ReplayPlan) -> AgentState:
        """按计划中的 checkpoint_id 还原 AgentState。"""

        checkpoint = await self._load_checkpoint_for_plan(plan)
        return checkpoint.state

    async def _replay_event_log(
        self,
        plan: ReplayPlan,
        target_trace_store: Optional[TraceStore],
    ) -> ReplayResult:
        """把历史事件复制到新的 replay run。"""

        replay_run_id = plan.replay_run_id or _new_replay_run_id(plan.source_run_id)
        executable_plan = plan.with_replay_run_id(replay_run_id)
        store = target_trace_store or self.trace_store
        replayed_events = tuple(clone_events_for_replay(executable_plan))
        await store.append_many(replayed_events)
        return ReplayResult.completed(
            plan=executable_plan,
            replayed_events=replayed_events,
            metadata={
                "source_run_id": executable_plan.source_run_id,
                "replay_run_id": replay_run_id,
                "replayed_event_count": len(replayed_events),
            },
        )

    async def _resolve_checkpoint_id(
        self,
        run_id: str,
        checkpoint_id: Optional[str],
    ) -> Optional[str]:
        """优先使用显式 checkpoint_id，否则尝试找该 run 最新 checkpoint。"""

        if checkpoint_id:
            return checkpoint_id
        if self.checkpoint_store is None:
            return None

        checkpoint = await self.checkpoint_store.latest_for_run(run_id)
        if checkpoint is None:
            return None
        return checkpoint.id

    async def _load_checkpoint_for_plan(self, plan: ReplayPlan) -> Checkpoint:
        """读取计划关联的 Checkpoint。"""

        if self.checkpoint_store is None:
            raise ReplayError(
                "CheckpointStore is required for checkpoint replay.",
                details={"plan_id": plan.id, "source_run_id": plan.source_run_id},
            )
        if not plan.checkpoint_id:
            raise ReplayError(
                "ReplayPlan does not reference a checkpoint.",
                details={"plan_id": plan.id, "source_run_id": plan.source_run_id},
            )
        checkpoint = await self.checkpoint_store.load(plan.checkpoint_id)
        if checkpoint.run_id != plan.source_run_id:
            raise ReplayError(
                "Checkpoint belongs to another run.",
                details={
                    "plan_id": plan.id,
                    "source_run_id": plan.source_run_id,
                    "checkpoint_id": checkpoint.id,
                    "checkpoint_run_id": checkpoint.run_id,
                },
            )
        return checkpoint


def select_event_range(
    events: Iterable[RunEvent],
    *,
    start_event_id: Optional[str] = None,
    end_event_id: Optional[str] = None,
) -> List[RunEvent]:
    """按事件 ID 截取需要 replay 的事件范围。"""

    event_list = list(events)
    if not event_list:
        raise ReplayError("Cannot select event range from empty events.")

    start_index = 0
    end_index = len(event_list) - 1

    if start_event_id:
        start_index = find_event_index(event_list, start_event_id)
    if end_event_id:
        end_index = find_event_index(event_list, end_event_id)
    if start_index > end_index:
        raise ReplayError(
            "Replay event range is invalid.",
            details={
                "start_event_id": start_event_id,
                "end_event_id": end_event_id,
            },
        )

    return event_list[start_index : end_index + 1]


def find_event_index(events: List[RunEvent], event_id: str) -> int:
    """查找事件在列表中的位置。"""

    for index, event in enumerate(events):
        if event.id == event_id:
            return index
    raise ReplayError(
        "Replay event id not found.",
        details={"event_id": event_id},
    )


def count_events_by_type(events: Iterable[RunEvent]) -> Dict[str, int]:
    """统计不同事件类型的数量。"""

    counts: Dict[str, int] = {}
    for event in events:
        key = event.event_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def clone_events_for_replay(plan: ReplayPlan) -> List[RunEvent]:
    """批量复制历史事件，并在 replay run 内重建父子关系。"""

    id_map: Dict[str, str] = {}
    replayed_events: List[RunEvent] = []
    for event in plan.events:
        cloned = clone_event_for_replay(
            event,
            plan,
            parent_event_id=id_map.get(event.parent_event_id or ""),
        )
        id_map[event.id] = cloned.id
        replayed_events.append(cloned)
    return replayed_events


def clone_event_for_replay(
    event: RunEvent,
    plan: ReplayPlan,
    *,
    parent_event_id: Optional[str] = None,
) -> RunEvent:
    """复制历史事件为新的 replay run 事件。"""

    if not plan.replay_run_id:
        raise ReplayError(
            "ReplayPlan requires replay_run_id before cloning events.",
            details={"plan_id": plan.id},
        )

    metadata = dict(event.metadata)
    metadata.update(
        {
            "is_replay": True,
            "replay_plan_id": plan.id,
            "source_run_id": plan.source_run_id,
            "source_event_id": event.id,
            "source_parent_event_id": event.parent_event_id,
        }
    )

    return RunEvent.create(
        run_id=plan.replay_run_id,
        session_id=event.session_id,
        event_type=event.event_type,
        step=event.step,
        parent_event_id=parent_event_id,
        level=event.level,
        message=event.message,
        payload=dict(event.payload),
        metadata=metadata,
        error=event.error,
    )


__all__ = [
    "ReplayError",
    "ReplayManager",
    "ReplayMode",
    "ReplayPlan",
    "ReplayResult",
    "ReplayStatus",
    "clone_events_for_replay",
    "clone_event_for_replay",
    "count_events_by_type",
    "find_event_index",
    "select_event_range",
]
