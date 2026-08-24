"""LLM 抽象层基础协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Type, TypeVar
from uuid import uuid4

from pydantic import Field

from core.message import Message, MessageRole
from infra.exception import LLMProviderError, ValidationError as HarnessValidationError
from infra.json import parse_json_value
from infra.serialization import SerializableModel


T = TypeVar("T", bound=SerializableModel)


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_request_id() -> str:
    return f"llm_req_{uuid4().hex}"


def _new_response_id() -> str:
    return f"llm_resp_{uuid4().hex}"


def _new_chunk_id() -> str:
    return f"llm_chunk_{uuid4().hex}"


class FinishReason(str, Enum):
    """LLM 响应结束原因。"""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class LLMUsage(SerializableModel):
    """LLM Token 用量统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 token 计数不能为负数。"""

        for field_name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "LLM token usage cannot be negative.",
                    details={"field": field_name, "value": value},
                )

    @property
    def total_tokens(self) -> int:
        """返回总 token 数。"""

        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def empty(cls) -> "LLMUsage":
        """创建空用量对象。"""

        return cls()

    def add(self, other: "LLMUsage") -> "LLMUsage":
        """合并两份 token 用量。"""

        metadata = dict(self.metadata)
        metadata.update(other.metadata)
        return self.__class__(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            metadata=metadata,
        )


class ChatRequest(SerializableModel):
    """统一 LLM Chat 请求。"""

    id: str = Field(default_factory=_new_request_id)
    messages: Tuple[Message, ...]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    stop: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Chat 请求的基础领域规则。"""

        if not self.messages:
            raise HarnessValidationError(
                "ChatRequest requires at least one message.",
                details={"request_id": self.id},
            )
        if self.temperature is not None and self.temperature < 0:
            raise HarnessValidationError(
                "ChatRequest temperature cannot be negative.",
                details={"request_id": self.id, "temperature": self.temperature},
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise HarnessValidationError(
                "ChatRequest max_tokens must be positive.",
                details={"request_id": self.id, "max_tokens": self.max_tokens},
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise HarnessValidationError(
                "ChatRequest timeout_seconds must be positive.",
                details={"request_id": self.id, "timeout_seconds": self.timeout_seconds},
            )

    @classmethod
    def create(
        cls,
        *,
        messages: Sequence[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ChatRequest":
        """创建 Chat 请求。"""

        return cls(
            messages=tuple(messages),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            stop=tuple(stop or ()),
            metadata=metadata or {},
        )

    def with_message(self, message: Message) -> "ChatRequest":
        """追加一条消息并返回新请求。"""

        return self.clone(messages=self.messages + (message,))


class ChatResponse(SerializableModel):
    """统一 LLM Chat 响应。"""

    id: str = Field(default_factory=_new_response_id)
    request_id: Optional[str] = None
    message: Message
    provider: str
    model: str
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: LLMUsage = Field(default_factory=LLMUsage.empty)
    raw: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Chat 响应的基础领域规则。"""

        if self.message.role != MessageRole.ASSISTANT:
            raise HarnessValidationError(
                "ChatResponse message must be an assistant message.",
                details={"response_id": self.id, "role": self.message.role.value},
            )
        if not self.provider:
            raise HarnessValidationError(
                "ChatResponse requires provider.",
                details={"response_id": self.id},
            )
        if not self.model:
            raise HarnessValidationError(
                "ChatResponse requires model.",
                details={"response_id": self.id},
            )

    @property
    def text(self) -> str:
        """返回 assistant 文本内容。"""

        return self.message.content


class StreamChunk(SerializableModel):
    """统一 LLM 流式响应分片。"""

    id: str = Field(default_factory=_new_chunk_id)
    request_id: Optional[str] = None
    response_id: Optional[str] = None
    index: int = 0
    delta: str = ""
    finish_reason: Optional[FinishReason] = None
    usage: Optional[LLMUsage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验流式分片的基础领域规则。"""

        if self.index < 0:
            raise HarnessValidationError(
                "StreamChunk index cannot be negative.",
                details={"chunk_id": self.id, "index": self.index},
            )

    @property
    def is_final(self) -> bool:
        """判断当前分片是否为结束分片。"""

        return self.finish_reason is not None


class BaseLLM(ABC):
    """所有 LLM Provider 的统一抽象接口。"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回 Provider 名称。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回默认模型名称。"""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """执行非流式 Chat 调用。"""

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """执行流式 Chat 调用。

        V1 默认用非流式响应模拟一个完整分片；真实 Provider 可以覆盖该方法。
        """

        response = await self.chat(request)
        yield StreamChunk(
            request_id=request.id,
            response_id=response.id,
            index=0,
            delta=response.text,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def structured(
        self,
        request: ChatRequest,
        output_model: Type[T],
    ) -> T:
        """执行结构化输出调用。

        支持纯 JSON、Markdown code fence 包裹的 JSON，以及模型回答中夹带的
        第一个 JSON object/array。
        """

        response = await self.chat(request)
        try:
            payload = parse_json_value(response.text)
            return output_model.from_dict(payload)
        except Exception as exc:
            raise LLMProviderError(
                "Failed to parse structured LLM output.",
                details={
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "output_model": output_model.__name__,
                    "response_id": response.id,
                    "response_preview": response.text[:500],
                },
                cause=exc,
            ) from exc


__all__ = [
    "BaseLLM",
    "ChatRequest",
    "ChatResponse",
    "FinishReason",
    "LLMUsage",
    "StreamChunk",
]
