"""核心 Agent 状态协议。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

from pydantic import Field

from core.message import Message, MessageRole
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_run_id() -> str:
    return f"run_{uuid4().hex}"


class AgentStatus(str, Enum):
    """Agent 单次运行的状态。"""

    CREATED = "created"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(SerializableModel):
    """Agent 运行过程中的状态快照。

    State 是 Runner、Loop、Checkpoint 之间共享的标准对象。
    V1 版本中状态对象不可变，任何更新都返回一个新的 AgentState，便于追踪和重放。
    """

    run_id: str = Field(default_factory=_new_run_id)
    session_id: Optional[str] = None
    status: AgentStatus = AgentStatus.CREATED
    step: int = 0
    max_steps: Optional[int] = None
    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    final_output: Optional[str] = None
    summary: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 AgentState 的基础领域规则。"""

        if not self.run_id:
            raise HarnessValidationError("AgentState requires run_id.")
        if self.step < 0:
            raise HarnessValidationError(
                "AgentState step cannot be negative.",
                details={"run_id": self.run_id, "step": self.step},
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise HarnessValidationError(
                "AgentState max_steps must be positive.",
                details={"run_id": self.run_id, "max_steps": self.max_steps},
            )
        if self.max_steps is not None and self.step > self.max_steps:
            raise HarnessValidationError(
                "AgentState step cannot exceed max_steps.",
                details={
                    "run_id": self.run_id,
                    "step": self.step,
                    "max_steps": self.max_steps,
                },
            )
        if self.status == AgentStatus.COMPLETED and self.error:
            raise HarnessValidationError(
                "Completed AgentState cannot carry error.",
                details={"run_id": self.run_id},
            )

    @classmethod
    def new(
        cls,
        *,
        session_id: Optional[str] = None,
        messages: Optional[Iterable[Message]] = None,
        max_steps: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentState":
        """创建一份新的 Agent 状态。"""

        return cls(
            session_id=session_id,
            messages=tuple(messages or ()),
            max_steps=max_steps,
            metadata=metadata or {},
        )

    @property
    def is_terminal(self) -> bool:
        """判断状态是否已经结束。"""

        return self.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }

    @property
    def last_message(self) -> Optional[Message]:
        """返回最后一条消息。"""

        if not self.messages:
            return None
        return self.messages[-1]

    @property
    def last_user_message(self) -> Optional[Message]:
        """返回最后一条用户消息。"""

        return self._last_message_by_role(MessageRole.USER)

    @property
    def last_assistant_message(self) -> Optional[Message]:
        """返回最后一条 assistant 消息。"""

        return self._last_message_by_role(MessageRole.ASSISTANT)

    def _last_message_by_role(self, role: MessageRole) -> Optional[Message]:
        for message in reversed(self.messages):
            if message.role == role:
                return message
        return None

    def _with_updates(self, **updates: object) -> "AgentState":
        data = self.to_dict()
        data.update(updates)
        data["updated_at"] = _utc_now()
        return self.__class__.from_dict(data)

    def with_status(self, status: AgentStatus) -> "AgentState":
        """返回状态变更后的新快照。"""

        return self._with_updates(status=status)

    def mark_running(self) -> "AgentState":
        """标记为运行中。"""

        return self.with_status(AgentStatus.RUNNING)

    def mark_waiting_tool(self) -> "AgentState":
        """标记为等待工具结果。"""

        return self.with_status(AgentStatus.WAITING_TOOL)

    def mark_completed(self, final_output: str) -> "AgentState":
        """标记为完成，并写入最终输出。"""

        return self._with_updates(
            status=AgentStatus.COMPLETED,
            final_output=final_output,
            error=None,
        )

    def mark_failed(self, error: Dict[str, Any]) -> "AgentState":
        """标记为失败，并写入结构化错误。"""

        return self._with_updates(
            status=AgentStatus.FAILED,
            error=error,
        )

    def mark_cancelled(self) -> "AgentState":
        """标记为取消。"""

        return self.with_status(AgentStatus.CANCELLED)

    def next_step(self) -> "AgentState":
        """进入下一步执行。"""

        return self._with_updates(step=self.step + 1)

    def add_message(self, message: Message) -> "AgentState":
        """追加一条消息并返回新状态。"""

        return self._with_updates(messages=self.messages + (message,))

    def add_messages(self, messages: Iterable[Message]) -> "AgentState":
        """批量追加消息并返回新状态。"""

        return self._with_updates(messages=self.messages + tuple(messages))

    def with_summary(self, summary: str) -> "AgentState":
        """更新上下文摘要。"""

        return self._with_updates(summary=summary)

    def with_variable(self, key: str, value: Any) -> "AgentState":
        """写入一个运行变量。"""

        variables = dict(self.variables)
        variables[key] = value
        return self._with_updates(variables=variables)

    def with_metadata(self, key: str, value: Any) -> "AgentState":
        """写入一个元数据字段。"""

        metadata = dict(self.metadata)
        metadata[key] = value
        return self._with_updates(metadata=metadata)


__all__ = [
    "AgentState",
    "AgentStatus",
]
