"""上下文构建器。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import Field

from core.message import Message
from core.state import AgentState
from infra.config import ContextConfig
from infra.exception import ContextError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import ChatRequest
from context.compression import ContextCompressionResult, ContextCompressor
from context.pipeline import (
    LayeredPromptPipeline,
    PromptLayerType,
    PromptSegment,
)
from context.pruner import ContextPruner
from context.window import ContextWindow
from memory.base import MemoryInjection, MemoryKind, MemoryQuery
from memory.manager import MemoryManager


class BuiltContext(SerializableModel):
    """一次上下文构建的结果。"""

    messages: Tuple[Message, ...]
    estimated_tokens: int
    dropped_messages: int = 0
    prompt_segments: Tuple[PromptSegment, ...] = Field(default_factory=tuple)
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验构建结果。"""

        if not self.messages:
            raise HarnessValidationError("BuiltContext requires at least one message.")
        if self.estimated_tokens < 0:
            raise HarnessValidationError(
                "BuiltContext estimated_tokens cannot be negative.",
                details={"estimated_tokens": self.estimated_tokens},
            )
        if self.dropped_messages < 0:
            raise HarnessValidationError(
                "BuiltContext dropped_messages cannot be negative.",
                details={"dropped_messages": self.dropped_messages},
            )

    def to_chat_request(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ChatRequest:
        """把构建结果转换为 LLM ChatRequest。"""

        request_metadata = dict(self.metadata)
        request_metadata.update(
            {
                "estimated_input_tokens": self.estimated_tokens,
                "dropped_messages": self.dropped_messages,
            }
        )
        return ChatRequest.create(
            messages=self.messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=request_metadata,
        )


class ContextBuilder:
    """从 AgentState 构建 LLM 输入上下文。"""

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        *,
        system_message: Optional[Message] = None,
        prompt_pipeline: Optional[LayeredPromptPipeline] = None,
        context_pruner: Optional[ContextPruner] = None,
        context_compressor: Optional[ContextCompressor] = None,
        memory_manager: Optional[MemoryManager] = None,
        auto_inject_memory: bool = True,
    ) -> None:
        self.config = config or ContextConfig()
        self.system_message = system_message
        self.prompt_pipeline = prompt_pipeline or LayeredPromptPipeline()
        self.pruner = context_pruner or ContextPruner.from_config(self.config)
        self.compressor = context_compressor
        self.window = ContextWindow.from_config(self.config)
        self.memory_manager = memory_manager
        self.auto_inject_memory = auto_inject_memory

    def build(self, state: AgentState) -> BuiltContext:
        """根据当前 AgentState 构建 LLM 上下文。"""

        return self._build_from_state(
            state,
            compression_result=None,
            memory_injection=None,
        )

    async def build_async(self, state: AgentState) -> BuiltContext:
        """异步构建 LLM 上下文，可启用模型摘要压缩。"""

        target_state = state
        compression_result = None
        if self.compressor is None:
            target_state = state
        else:
            compression_result = await self.compressor.compress(state)
            target_state = compression_result.state
        memory_injection = await self._build_memory_injection(target_state)
        return self._build_from_state(
            target_state,
            compression_result=compression_result,
            memory_injection=memory_injection,
        )

    def _build_from_state(
        self,
        state: AgentState,
        *,
        compression_result: Optional[ContextCompressionResult],
        memory_injection: Optional[MemoryInjection],
    ) -> BuiltContext:
        """从指定状态执行 Prompt 拼装、智能裁剪和窗口选择。"""

        extra_segments = []
        memory_segment = self._memory_injection_to_segment(memory_injection)
        if memory_segment is not None:
            extra_segments.append(memory_segment)

        assembly = self.prompt_pipeline.assemble(
            state,
            system_message=self.system_message,
            extra_segments=extra_segments,
        )
        if not assembly.messages:
            raise ContextError(
                "Cannot build context without messages.",
                details={"run_id": state.run_id},
            )

        prune_result = self.pruner.prune(assembly.segments)
        selection = self.window.select(prune_result.messages)

        if not selection.messages:
            raise ContextError(
                "Context window is empty after trimming.",
                details={"run_id": state.run_id},
            )

        return BuiltContext(
            messages=selection.messages,
            estimated_tokens=selection.estimated_tokens,
            dropped_messages=prune_result.dropped_messages + selection.dropped_messages,
            prompt_segments=assembly.segments,
            metadata={
                "run_id": state.run_id,
                "session_id": state.session_id,
                "max_messages": self.config.max_messages,
                "max_input_tokens": self.config.max_input_tokens,
                "input_messages": len(assembly.messages),
                "output_messages": len(selection.messages),
                "dropped_messages": prune_result.dropped_messages + selection.dropped_messages,
                "context_compression": (
                    compression_result.metadata
                    if compression_result is not None
                    else {"enabled": False}
                ),
                "memory_injection": (
                    memory_injection.to_dict(exclude_none=True)
                    if memory_injection is not None
                    else {"enabled": False}
                ),
                "prompt_pipeline": assembly.metadata,
                "context_pruner": prune_result.metadata,
                "prune_decisions": [
                    decision.to_dict(exclude_none=True)
                    for decision in prune_result.decisions
                ],
                "dropped_by_count": selection.dropped_by_count,
                "dropped_by_tokens": selection.dropped_by_tokens,
                "dropped_by_pruner": prune_result.dropped_messages,
                "context_window": selection.metadata,
                "token_budget_usage": (
                    selection.budget_usage.to_dict(exclude_none=True)
                    if selection.budget_usage is not None
                    else None
                ),
                "content_budget": (
                    selection.content_budget.to_dict(exclude_none=True)
                    if selection.content_budget is not None
                    else None
                ),
            },
        )

    def build_request(
        self,
        state: AgentState,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ChatRequest:
        """直接构建 LLM ChatRequest。"""

        return self.build(state).to_chat_request(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def build_request_async(
        self,
        state: AgentState,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ChatRequest:
        """异步构建 LLM ChatRequest，可启用模型摘要压缩。"""

        return (await self.build_async(state)).to_chat_request(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _collect_messages(self, state: AgentState) -> Tuple[Message, ...]:
        """兼容旧路径：收集系统提示、摘要和历史消息。"""

        return self.prompt_pipeline.assemble(
            state,
            system_message=self.system_message,
        ).messages

    async def _build_memory_injection(
        self,
        state: AgentState,
    ) -> Optional[MemoryInjection]:
        """按当前 AgentState 检索并构造记忆注入。"""

        if self.memory_manager is None or not self.auto_inject_memory:
            return None
        if not self.memory_manager.policy.enabled or not self.memory_manager.policy.auto_inject:
            return None
        query = build_memory_query_from_state(state, policy_top_k=self.memory_manager.policy.default_top_k)
        return await self.memory_manager.build_injection(query)

    def _memory_injection_to_segment(
        self,
        injection: Optional[MemoryInjection],
    ) -> Optional[PromptSegment]:
        """把记忆注入结果转换成 PromptSegment。"""

        if injection is None or not injection.content:
            return None
        message = Message.system(injection.content)
        return PromptSegment(
            name="memory",
            layer=PromptLayerType.MEMORY,
            order=20,
            required=False,
            messages=(message,),
            estimated_tokens=injection.estimated_tokens,
            metadata={
                "result_count": len(injection.results),
                "record_ids": [record.id for record in injection.records],
                "query": injection.query.to_dict(exclude_none=True),
                **injection.metadata,
            },
        )


def build_memory_query_from_state(
    state: AgentState,
    *,
    policy_top_k: int,
) -> MemoryQuery:
    """根据 AgentState 构造默认记忆检索请求。"""

    query_text = build_memory_query_text(state)
    user_id = optional_metadata_string(state, "user_id")
    tags = tuple(str(tag) for tag in state.metadata.get("memory_tags", ()) if str(tag).strip())
    return MemoryQuery(
        text=query_text,
        kinds=(MemoryKind.SHORT_TERM, MemoryKind.LONG_TERM, MemoryKind.SKILL),
        session_id=state.session_id,
        run_id=state.run_id,
        user_id=user_id,
        tags=tags,
        top_k=policy_top_k,
    )


def build_memory_query_text(state: AgentState) -> str:
    """从当前状态抽取适合检索记忆的查询文本。"""

    parts = []
    if state.summary:
        parts.append(state.summary)
    if state.last_user_message is not None:
        parts.append(state.last_user_message.content)
    if state.final_output:
        parts.append(state.final_output)
    if not parts and state.messages:
        parts.append(state.messages[-1].content)
    return "\n".join(part for part in parts if part)


def optional_metadata_string(state: AgentState, key: str) -> Optional[str]:
    """从 state.metadata 中读取可选字符串。"""

    value = state.metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None

__all__ = [
    "BuiltContext",
    "ContextBuilder",
    "build_memory_query_from_state",
    "build_memory_query_text",
    "optional_metadata_string",
]
