"""模型摘要器。

摘要器只依赖 BaseLLM 抽象，不直接调用任何模型厂商 SDK。
它负责把历史对话或工具结果压缩成结构化文本，供上下文压缩层使用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.exception import ContextError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest


DEFAULT_HISTORY_SUMMARY_PROMPT = """你是一个 Agent Harness 的上下文摘要器。

请把下面较早的历史对话压缩成结构化摘要。摘要必须保留可以支撑后续任务继续执行的信息。
输出必须使用下面固定结构：

用户目标:
- ...

Agent 已完成:
- ...

关键输出:
- ...

错误或阻塞:
- ...

待继续事项:
- ...
"""


DEFAULT_TOOL_SUMMARY_PROMPT = """你是一个 Agent Harness 的工具结果摘要器。

请把下面的工具结果压缩成结构化摘要。摘要必须尽量保留后续推理需要的信息，尤其是错误、路径、数值、字段名、命令输出结论。
输出必须使用下面固定结构：

工具名称:
- ...

调用标识:
- ...

结果摘要:
- ...

关键数据:
- ...

错误信息:
- ...

截断说明:
- ...
"""


class SummaryKind(str, Enum):
    """摘要类型常量。"""

    HISTORY = "history"
    TOOL_RESULT = "tool_result"


class SummaryRequest(SerializableModel):
    """一次摘要请求。"""

    kind: SummaryKind
    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    existing_summary: Optional[str] = None
    max_summary_tokens: int = 800
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验摘要请求。"""

        if not self.kind.value.strip():
            raise HarnessValidationError("SummaryRequest requires non-empty kind.")
        if self.max_summary_tokens <= 0:
            raise HarnessValidationError(
                "SummaryRequest max_summary_tokens must be positive.",
                details={"max_summary_tokens": self.max_summary_tokens},
            )


class SummaryResult(SerializableModel):
    """一次摘要结果。"""

    kind: SummaryKind
    summary: str
    source_message_ids: Tuple[str, ...] = Field(default_factory=tuple)
    source_message_count: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验摘要结果。"""

        if not self.kind.value.strip():
            raise HarnessValidationError("SummaryResult requires non-empty kind.")
        if self.summary is None:
            raise HarnessValidationError("SummaryResult summary cannot be None.")
        if self.source_message_count < 0:
            raise HarnessValidationError(
                "SummaryResult source_message_count cannot be negative.",
                details={"source_message_count": self.source_message_count},
            )


class BaseSummarizer(ABC):
    """摘要器抽象。"""

    @abstractmethod
    async def summarize_history(self, request: SummaryRequest) -> SummaryResult:
        """压缩历史对话。"""

    @abstractmethod
    async def summarize_tool_result(self, request: SummaryRequest) -> SummaryResult:
        """压缩工具结果。"""


class LLMSummarizer(BaseSummarizer):
    """基于 BaseLLM 的模型摘要器。"""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None,
        history_system_prompt: str = DEFAULT_HISTORY_SUMMARY_PROMPT,
        tool_system_prompt: str = DEFAULT_TOOL_SUMMARY_PROMPT,
    ) -> None:
        if temperature < 0:
            raise HarnessValidationError(
                "LLMSummarizer temperature cannot be negative.",
                details={"temperature": temperature},
            )
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.history_system_prompt = history_system_prompt
        self.tool_system_prompt = tool_system_prompt

    async def summarize_history(self, request: SummaryRequest) -> SummaryResult:
        """调用模型压缩历史对话。"""

        return await self._summarize(
            request=request,
            system_prompt=self.history_system_prompt,
            user_content=format_history_summary_input(request),
        )

    async def summarize_tool_result(self, request: SummaryRequest) -> SummaryResult:
        """调用模型压缩工具结果。"""

        return await self._summarize(
            request=request,
            system_prompt=self.tool_system_prompt,
            user_content=format_tool_summary_input(request),
        )

    async def _summarize(
        self,
        *,
        request: SummaryRequest,
        system_prompt: str,
        user_content: str,
    ) -> SummaryResult:
        """执行一次模型摘要调用。"""

        if not request.messages and not request.existing_summary:
            return SummaryResult(
                kind=request.kind,
                summary="",
                source_message_ids=(),
                source_message_count=0,
                metadata={"skipped": True, "reason": "empty_input"},
            )

        chat_request = ChatRequest.create(
            messages=(
                Message.system(system_prompt),
                Message.user(user_content),
            ),
            model=self.model or self.llm.model_name,
            temperature=self.temperature,
            max_tokens=request.max_summary_tokens,
            timeout_seconds=self.timeout_seconds,
            metadata={
                "purpose": "context_summary",
                "summary_kind": request.kind.value,
                "source_message_count": len(request.messages),
            },
        )
        try:
            response = await self.llm.chat(chat_request)
        except Exception as exc:
            raise ContextError(
                "LLM summarization failed.",
                details={
                    "kind": request.kind.value,
                    "provider": self.llm.provider_name,
                    "model": self.model or self.llm.model_name,
                },
                cause=exc,
            ) from exc

        summary = response.text.strip()
        return SummaryResult(
            kind=request.kind,
            summary=summary,
            source_message_ids=tuple(message.id for message in request.messages),
            source_message_count=len(request.messages),
            metadata={
                "provider": response.provider,
                "model": response.model,
                "request_id": chat_request.id,
                "response_id": response.id,
                "usage": response.usage.to_dict(),
            },
        )


def format_history_summary_input(request: SummaryRequest) -> str:
    """格式化历史摘要输入。"""

    parts = []
    if request.existing_summary:
        parts.append("已有摘要:\n" + request.existing_summary.strip())
    parts.append("需要压缩的较早历史对话:")
    parts.append(format_messages_for_summary(request.messages))
    return "\n\n".join(parts).strip()


def format_tool_summary_input(request: SummaryRequest) -> str:
    """格式化工具结果摘要输入。"""

    parts = []
    metadata = request.metadata
    tool_name = metadata.get("tool_name") or metadata.get("name") or "unknown"
    tool_call_id = metadata.get("tool_call_id") or "unknown"
    parts.append(f"工具名称: {tool_name}")
    parts.append(f"调用标识: {tool_call_id}")
    parts.append("原始工具结果:")
    parts.append(format_messages_for_summary(request.messages))
    return "\n\n".join(parts).strip()


def format_messages_for_summary(messages: Iterable[Message]) -> str:
    """把消息列表格式化为摘要器输入文本。"""

    lines = []
    for index, message in enumerate(messages, start=1):
        name = f" name={message.name}" if message.name else ""
        tool_call_id = f" tool_call_id={message.tool_call_id}" if message.tool_call_id else ""
        lines.append(
            f"[{index}] role={message.role.value}{name}{tool_call_id}\n{message.content}"
        )
    return "\n\n".join(lines)


__all__ = [
    "BaseSummarizer",
    "DEFAULT_HISTORY_SUMMARY_PROMPT",
    "DEFAULT_TOOL_SUMMARY_PROMPT",
    "LLMSummarizer",
    "SummaryKind",
    "SummaryRequest",
    "SummaryResult",
    "format_history_summary_input",
    "format_messages_for_summary",
    "format_tool_summary_input",
]
