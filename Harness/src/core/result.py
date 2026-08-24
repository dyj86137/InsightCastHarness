"""核心运行结果协议。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

from pydantic import Field

from core.event import RunEvent, error_to_payload
from core.state import AgentState
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_result_id() -> str:
    return f"result_{uuid4().hex}"


class RunResultStatus(str, Enum):
    """单次 run 的最终结果状态。"""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunResult(SerializableModel):
    """Runner 返回给调用方的标准结果对象。"""

    id: str = Field(default_factory=_new_result_id)
    run_id: str
    status: RunResultStatus
    state: AgentState
    output: Optional[str] = None
    session_id: Optional[str] = None
    events: Tuple[RunEvent, ...] = Field(default_factory=tuple)
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验运行结果的基础领域规则。"""

        if not self.run_id:
            raise HarnessValidationError("RunResult requires run_id.")
        if self.state.run_id != self.run_id:
            raise HarnessValidationError(
                "RunResult state must belong to the same run.",
                details={
                    "result_run_id": self.run_id,
                    "state_run_id": self.state.run_id,
                },
            )
        if self.duration_ms is not None and self.duration_ms < 0:
            raise HarnessValidationError(
                "RunResult duration_ms cannot be negative.",
                details={"run_id": self.run_id, "duration_ms": self.duration_ms},
            )
        if self.status == RunResultStatus.SUCCESS and self.error:
            raise HarnessValidationError(
                "Successful RunResult cannot carry error.",
                details={"run_id": self.run_id},
            )
        if self.status == RunResultStatus.FAILED and not self.error:
            raise HarnessValidationError(
                "Failed RunResult requires error payload.",
                details={"run_id": self.run_id},
            )

    @property
    def succeeded(self) -> bool:
        """判断 run 是否成功。"""

        return self.status == RunResultStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """判断 run 是否失败。"""

        return self.status == RunResultStatus.FAILED

    @classmethod
    def success(
        cls,
        *,
        state: AgentState,
        events: Optional[Iterable[RunEvent]] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunResult":
        """创建成功结果。"""

        return cls(
            run_id=state.run_id,
            session_id=state.session_id,
            status=RunResultStatus.SUCCESS,
            state=state,
            output=state.final_output,
            events=tuple(events or ()),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        *,
        state: AgentState,
        error: BaseException,
        events: Optional[Iterable[RunEvent]] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunResult":
        """创建失败结果。"""

        error_payload = error_to_payload(error)
        failed_state = state.mark_failed(error_payload)
        return cls(
            run_id=failed_state.run_id,
            session_id=failed_state.session_id,
            status=RunResultStatus.FAILED,
            state=failed_state,
            output=failed_state.final_output,
            events=tuple(events or ()),
            error=error_payload,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def cancelled(
        cls,
        *,
        state: AgentState,
        events: Optional[Iterable[RunEvent]] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunResult":
        """创建取消结果。"""

        cancelled_state = state.mark_cancelled()
        return cls(
            run_id=cancelled_state.run_id,
            session_id=cancelled_state.session_id,
            status=RunResultStatus.CANCELLED,
            state=cancelled_state,
            output=cancelled_state.final_output,
            events=tuple(events or ()),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )


__all__ = [
    "RunResult",
    "RunResultStatus",
]
