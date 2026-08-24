"""核心运行事件协议。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import Field

from infra.exception import HarnessError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_event_id() -> str:
    return f"evt_{uuid4().hex}"


class RunEventType(str, Enum):
    """事件类型全集：统一 Harness 运行过程中的事件类型称呼。"""

    # 运行生命周期级
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # 推理步骤级
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"

    # 具体动作级：消息生成、上下文构建、LLM 请求 / 响应 / 失败、工具调用开始 / 完成 / 失败、检查点生成、告警 / 错误
    MESSAGE_CREATED = "message_created"
    CONTEXT_BUILT = "context_built"

    LLM_REQUESTED = "llm_requested"
    LLM_RESPONDED = "llm_responded"
    LLM_FAILED = "llm_failed"

    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"

    CHECKPOINT_CREATED = "checkpoint_created"
    WARNING = "warning"
    ERROR = "error"


class RunEventLevel(str, Enum):
    """事件严重级别（对齐通用日志级别），用于日志和 Trace 展示。"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RunEvent(SerializableModel):
    """Agent 单次运行中的标准事件。

    Runner、LLM、Tool、Context、Checkpoint 等模块只生产 RunEvent；
    事件如何落盘、展示、导出由 tracing 层负责。
    """

    id: str = Field(default_factory=_new_event_id)
    run_id: str
    event_type: RunEventType
    session_id: Optional[str] = None
    step: Optional[int] = None
    parent_event_id: Optional[str] = None
    level: RunEventLevel = RunEventLevel.INFO
    message: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验事件对象的基础领域规则。"""

        if not self.run_id:
            raise HarnessValidationError("RunEvent requires run_id.")
        if self.step is not None and self.step < 0:
            raise HarnessValidationError(
                "RunEvent step cannot be negative.",
                details={"event_id": self.id, "step": self.step},
            )
        if self.error and self.level != RunEventLevel.ERROR:
            raise HarnessValidationError(
                "RunEvent with error payload must use error level.",
                details={"event_id": self.id, "event_type": self.event_type.value},
            )

    @property
    def is_terminal(self) -> bool:
        """判断事件是否表示一次 run 已经结束。"""

        return self.event_type in {
            RunEventType.RUN_COMPLETED,
            RunEventType.RUN_FAILED,
            RunEventType.RUN_CANCELLED,
        }

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        event_type: RunEventType,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
        parent_event_id: Optional[str] = None,
        level: RunEventLevel = RunEventLevel.INFO,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建通用运行事件。"""

        return cls(
            run_id=run_id,
            session_id=session_id,
            event_type=event_type,
            step=step,
            parent_event_id=parent_event_id,
            level=level,
            message=message,
            payload=payload or {},
            metadata=metadata or {},
            error=error,
        )

    @classmethod
    def run_started(
        cls,
        *,
        run_id: str,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 run 启动事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.RUN_STARTED,
            message="Run started.",
            payload=payload,
        )

    @classmethod
    def run_completed(
        cls,
        *,
        run_id: str,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 run 完成事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.RUN_COMPLETED,
            message="Run completed.",
            payload=payload,
        )

    @classmethod
    def run_failed(
        cls,
        *,
        run_id: str,
        error: BaseException,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 run 失败事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.RUN_FAILED,
            level=RunEventLevel.ERROR,
            message="Run failed.",
            payload=payload,
            error=error_to_payload(error),
        )

    @classmethod
    def step_started(
        cls,
        *,
        run_id: str,
        step: int,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 step 启动事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.STEP_STARTED,
            step=step,
            message="Step started.",
            payload=payload,
        )

    @classmethod
    def step_completed(
        cls,
        *,
        run_id: str,
        step: int,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 step 完成事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.STEP_COMPLETED,
            step=step,
            message="Step completed.",
            payload=payload,
        )

    @classmethod
    def message_created(
        cls,
        *,
        run_id: str,
        message_id: str,
        role: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> "RunEvent":
        """创建消息生成事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.MESSAGE_CREATED,
            step=step,
            message="Message created.",
            payload={"message_id": message_id, "role": role},
        )

    @classmethod
    def llm_requested(
        cls,
        *,
        run_id: str,
        provider: str,
        model: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 LLM 请求事件。"""

        event_payload = {"provider": provider, "model": model}
        if payload:
            event_payload.update(payload)
        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.LLM_REQUESTED,
            step=step,
            message="LLM requested.",
            payload=event_payload,
        )

    @classmethod
    def llm_responded(
        cls,
        *,
        run_id: str,
        provider: str,
        model: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建 LLM 响应事件。"""

        event_payload = {"provider": provider, "model": model}
        if payload:
            event_payload.update(payload)
        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.LLM_RESPONDED,
            step=step,
            message="LLM responded.",
            payload=event_payload,
        )

    @classmethod
    def tool_call_started(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> "RunEvent":
        """创建工具调用开始事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.TOOL_CALL_STARTED,
            step=step,
            message="Tool call started.",
            payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
        )

    @classmethod
    def tool_call_completed(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """创建工具调用完成事件。"""

        event_payload = {"tool_call_id": tool_call_id, "tool_name": tool_name}
        if payload:
            event_payload.update(payload)
        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.TOOL_CALL_COMPLETED,
            step=step,
            message="Tool call completed.",
            payload=event_payload,
        )

    @classmethod
    def tool_call_failed(
        cls,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        error: BaseException,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> "RunEvent":
        """创建工具调用失败事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.TOOL_CALL_FAILED,
            step=step,
            level=RunEventLevel.ERROR,
            message="Tool call failed.",
            payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
            error=error_to_payload(error),
        )

    @classmethod
    def checkpoint_created(
        cls,
        *,
        run_id: str,
        checkpoint_id: str,
        session_id: Optional[str] = None,
        step: Optional[int] = None,
    ) -> "RunEvent":
        """创建 checkpoint 事件。"""

        return cls.create(
            run_id=run_id,
            session_id=session_id,
            event_type=RunEventType.CHECKPOINT_CREATED,
            step=step,
            message="Checkpoint created.",
            payload={"checkpoint_id": checkpoint_id},
        )


def error_to_payload(error: BaseException) -> Dict[str, Any]:
    """把异常转换为可写入 RunEvent 的结构化错误。"""

    if isinstance(error, HarnessError):
        return error.to_dict()
    return {
        "type": error.__class__.__name__,
        "message": str(error),
    }


__all__ = [
    "RunEvent",
    "RunEventLevel",
    "RunEventType",
    "error_to_payload",
]
