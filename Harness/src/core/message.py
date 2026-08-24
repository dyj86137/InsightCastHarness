"""核心消息协议。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import Field

from infra.exception import ValidationError
from infra.serialization import SerializableModel


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_message_id() -> str:
    return f"msg_{uuid4().hex}"


class MessageRole(str, Enum):
    """框架内部统一使用的消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(SerializableModel):
    """Agent、LLM、Runner 之间传递的标准消息对象。

    V1 阶段只支持文本消息。工具调用本身由 ToolCall/ToolResult 表示；
    当工具结果需要回灌给模型时，使用 role=tool 的 Message 承载文本化结果。
    """

    id: str = Field(default_factory=_new_message_id)
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Pydantic 类型系统无法表达的消息领域规则。
        Pydantic 只能做「类型对不对、字段有没有」这类基础校验，但像「工具消息必须带工具调用 ID」、
        「非工具消息不能带这个字段」这类和业务语义绑定的规则，类型系统表达不了
        """

        if self.content is None:
            raise ValidationError(
                "Message content cannot be None.",
                details={"message_id": self.id, "role": self.role.value},
            )

        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValidationError(
                "Tool message requires tool_call_id.",
                details={"message_id": self.id, "role": self.role.value},
            )

        if self.role != MessageRole.TOOL and self.tool_call_id:
            raise ValidationError(
                "Only tool messages can carry tool_call_id.",
                details={
                    "message_id": self.id,
                    "role": self.role.value,
                    "tool_call_id": self.tool_call_id,
                },
            )

    # 下面是一组工厂方法，用来快速创建不同角色的消息
    @classmethod
    def system(cls, content: str, **kwargs: object) -> "Message":
        """创建 system 消息。"""

        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)


    @classmethod
    def user(cls, content: str, **kwargs: object) -> "Message":
        """创建 user 消息。"""

        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str, **kwargs: object) -> "Message":
        """创建 assistant 消息。"""

        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool(
        cls,
        content: str,
        *,
        tool_call_id: str,
        name: Optional[str] = None,
        **kwargs: object,
    ) -> "Message":
        """创建工具结果回灌消息（指工具的执行结果）。"""

        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs,
        )

    @property
    def is_system(self) -> bool:
        return self.role == MessageRole.SYSTEM

    @property
    def is_user(self) -> bool:
        return self.role == MessageRole.USER

    @property
    def is_assistant(self) -> bool:
        return self.role == MessageRole.ASSISTANT

    @property
    def is_tool(self) -> bool:
        return self.role == MessageRole.TOOL


__all__ = [
    "Message",
    "MessageRole",
]
