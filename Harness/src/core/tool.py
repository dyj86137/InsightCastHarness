"""核心工具调用协议。"""

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


def _new_tool_call_id() -> str:
    return f"call_{uuid4().hex}"


def _new_tool_result_id() -> str:
    return f"tool_result_{uuid4().hex}"


class ToolCallStatus(str, Enum):
    """工具调用请求的生命周期状态。"""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolResultStatus(str, Enum):
    """工具执行结果状态。"""

    SUCCESS = "success"
    ERROR = "error"


class ToolCall(SerializableModel):
    """模型或 Agent Loop 产生的标准工具调用请求。"""

    id: str = Field(default_factory=_new_tool_call_id)
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.CREATED
    message_id: Optional[str] = None
    step: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验工具调用请求的基础领域规则。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError(
                "ToolCall requires non-empty name.",
                details={"tool_call_id": self.id},
            )
        if self.step is not None and self.step < 0:
            raise HarnessValidationError(
                "ToolCall step cannot be negative.",
                details={"tool_call_id": self.id, "step": self.step},
            )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolCall":
        """创建工具调用请求。"""

        return cls(
            name=name,
            arguments=arguments or {},
            message_id=message_id,
            step=step,
            metadata=metadata or {},
        )

    def with_status(self, status: ToolCallStatus) -> "ToolCall":
        """返回状态变更后的新工具调用请求。"""

        return self.clone(status=status)

    def mark_running(self) -> "ToolCall":
        """标记工具调用进入执行中。"""

        return self.with_status(ToolCallStatus.RUNNING)

    def mark_completed(self) -> "ToolCall":
        """标记工具调用执行完成。"""

        return self.with_status(ToolCallStatus.COMPLETED)

    def mark_failed(self) -> "ToolCall":
        """标记工具调用执行失败。"""

        return self.with_status(ToolCallStatus.FAILED)


class ToolResult(SerializableModel):
    """工具执行后的标准返回对象。"""

    id: str = Field(default_factory=_new_tool_result_id)
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    output: str = ""
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验工具结果的基础领域规则。"""

        if not self.tool_call_id:
            raise HarnessValidationError("ToolResult requires tool_call_id.")
        if not self.tool_name or not self.tool_name.strip():
            raise HarnessValidationError(
                "ToolResult requires non-empty tool_name.",
                details={"tool_call_id": self.tool_call_id},
            )
        if self.duration_ms is not None and self.duration_ms < 0:
            raise HarnessValidationError(
                "ToolResult duration_ms cannot be negative.",
                details={"tool_call_id": self.tool_call_id, "duration_ms": self.duration_ms},
            )
        if self.status == ToolResultStatus.ERROR and not self.error:
            raise HarnessValidationError(
                "Error ToolResult requires error payload.",
                details={"tool_call_id": self.tool_call_id},
            )
        if self.status == ToolResultStatus.SUCCESS and self.error:
            raise HarnessValidationError(
                "Successful ToolResult cannot carry error payload.",
                details={"tool_call_id": self.tool_call_id},
            )

    @property
    def is_error(self) -> bool:
        """判断工具结果是否表示失败。"""

        return self.status == ToolResultStatus.ERROR

    @classmethod
    def success(
        cls,
        *,
        tool_call: ToolCall,
        output: str = "",
        data: Optional[Any] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """创建成功工具结果。"""

        return cls(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            status=ToolResultStatus.SUCCESS,
            output=output,
            data=data,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        *,
        tool_call: ToolCall,
        error: BaseException,
        output: str = "",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """创建失败工具结果。"""

        return cls(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            status=ToolResultStatus.ERROR,
            output=output,
            error=error_to_payload(error),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )


def error_to_payload(error: BaseException) -> Dict[str, Any]:
    """把异常转换为可写入 ToolResult 的结构化错误。"""

    if isinstance(error, HarnessError):
        return error.to_dict()
    return {
        "type": error.__class__.__name__,
        "message": str(error),
    }


__all__ = [
    "ToolCall",
    "ToolCallStatus",
    "ToolResult",
    "ToolResultStatus",
    "error_to_payload",
]
