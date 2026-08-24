"""上下文压缩策略。

压缩层发生在 Prompt 拼装、智能裁剪和窗口裁剪之前。
它负责把较早历史对话合并为结构化摘要，并把过长工具结果压缩为工具摘要。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from core.message import Message
from core.state import AgentState
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from context.summarizer import BaseSummarizer, SummaryKind, SummaryRequest, SummaryResult
from context.token import (
    DEFAULT_TOKEN_ESTIMATOR,
    TokenEstimator,
    estimate_message_tokens,
)


class ContextCompressionResult(SerializableModel):
    """一次上下文压缩结果。"""

    state: AgentState
    history_summary: Optional[SummaryResult] = None
    tool_summaries: Tuple[SummaryResult, ...] = Field(default_factory=tuple)
    original_message_count: int = 0
    output_message_count: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验压缩结果。"""

        for field_name in ("original_message_count", "output_message_count"):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "ContextCompressionResult count fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )


class ContextCompressor:
    """基于模型摘要器的上下文压缩器。"""

    def __init__(
        self,
        *,
        summarizer: BaseSummarizer,
        keep_recent_turns: int = 4,
        history_summary_max_tokens: int = 1000,
        tool_summary_max_tokens: int = 600,
        tool_result_token_threshold: int = 1200,
        estimator: Optional[TokenEstimator] = None,
    ) -> None:
        if keep_recent_turns <= 0:
            raise HarnessValidationError(
                "ContextCompressor keep_recent_turns must be positive.",
                details={"keep_recent_turns": keep_recent_turns},
            )
        for field_name, value in (
            ("history_summary_max_tokens", history_summary_max_tokens),
            ("tool_summary_max_tokens", tool_summary_max_tokens),
            ("tool_result_token_threshold", tool_result_token_threshold),
        ):
            if value <= 0:
                raise HarnessValidationError(
                    "ContextCompressor token fields must be positive.",
                    details={"field": field_name, "value": value},
                )

        self.summarizer = summarizer
        self.keep_recent_turns = keep_recent_turns
        self.history_summary_max_tokens = history_summary_max_tokens
        self.tool_summary_max_tokens = tool_summary_max_tokens
        self.tool_result_token_threshold = tool_result_token_threshold
        self.estimator = estimator or DEFAULT_TOKEN_ESTIMATOR

    async def compress(self, state: AgentState) -> ContextCompressionResult:
        """压缩 AgentState 中的历史消息和工具结果。"""

        original_messages = tuple(state.messages)
        compressed_state, history_summary = await self._compress_history(
            state,
            messages=original_messages,
        )
        tool_messages, tool_summaries = await self._compress_tool_results(
            compressed_state.messages
        )
        compressed_state = compressed_state.clone(messages=tool_messages)

        metadata = build_compression_metadata(
            original_state=state,
            output_state=compressed_state,
            keep_recent_turns=self.keep_recent_turns,
            tool_result_token_threshold=self.tool_result_token_threshold,
            history_summary=history_summary,
            tool_summaries=tool_summaries,
        )
        compressed_state = compressed_state.clone(
            metadata=merge_state_compression_metadata(compressed_state.metadata, metadata)
        )

        return ContextCompressionResult(
            state=compressed_state,
            history_summary=history_summary,
            tool_summaries=tuple(tool_summaries),
            original_message_count=len(original_messages),
            output_message_count=len(compressed_state.messages),
            metadata=metadata,
        )

    async def _compress_tool_results(
        self,
        messages: Tuple[Message, ...],
    ) -> Tuple[Tuple[Message, ...], List[SummaryResult]]:
        """压缩超过阈值的工具结果消息。"""

        compressed_messages: List[Message] = []
        summaries: List[SummaryResult] = []
        for message in messages:
            if not should_compress_tool_message(
                message,
                threshold_tokens=self.tool_result_token_threshold,
                estimator=self.estimator,
            ):
                compressed_messages.append(message)
                continue

            summary = await self.summarizer.summarize_tool_result(
                SummaryRequest(
                    kind=SummaryKind.TOOL_RESULT,
                    messages=(message,),
                    max_summary_tokens=self.tool_summary_max_tokens,
                    metadata={
                        "tool_name": message.name,
                        "tool_call_id": message.tool_call_id,
                    },
                )
            )
            summaries.append(summary)
            compressed_messages.append(
                build_compressed_tool_message(message, summary)
            )
        return tuple(compressed_messages), summaries

    async def _compress_history(
        self,
        state: AgentState,
        *,
        messages: Tuple[Message, ...],
    ) -> Tuple[AgentState, Optional[SummaryResult]]:
        """按最近轮次保留策略压缩历史消息。"""

        older_messages, recent_messages = split_history_by_recent_turns(
            messages,
            keep_recent_turns=self.keep_recent_turns,
        )
        if not older_messages:
            return state.clone(messages=messages), None

        summary = await self.summarizer.summarize_history(
            SummaryRequest(
                kind=SummaryKind.HISTORY,
                messages=older_messages,
                existing_summary=state.summary,
                max_summary_tokens=self.history_summary_max_tokens,
                metadata={
                    "keep_recent_turns": self.keep_recent_turns,
                    "older_message_count": len(older_messages),
                    "recent_message_count": len(recent_messages),
                },
            )
        )
        return state.clone(messages=recent_messages, summary=summary.summary), summary


def split_history_by_recent_turns(
    messages: Iterable[Message],
    *,
    keep_recent_turns: int,
) -> Tuple[Tuple[Message, ...], Tuple[Message, ...]]:
    """把历史消息拆成较早消息和最近 N 轮消息。

    这里把一条 user 消息以及其后的 assistant/tool 消息视为一轮。
    system 消息不参与历史压缩，会继续保留。
    """

    if keep_recent_turns <= 0:
        raise HarnessValidationError(
            "keep_recent_turns must be positive.",
            details={"keep_recent_turns": keep_recent_turns},
        )

    message_tuple = tuple(messages)
    user_indices = [
        index
        for index, message in enumerate(message_tuple)
        if message.is_user
    ]
    if len(user_indices) <= keep_recent_turns:
        return (), message_tuple

    cutoff_index = user_indices[-keep_recent_turns]
    older: List[Message] = []
    recent: List[Message] = []
    for index, message in enumerate(message_tuple):
        if message.is_system:
            recent.append(message)
        elif index < cutoff_index:
            older.append(message)
        else:
            recent.append(message)
    return tuple(older), tuple(recent)


def should_compress_tool_message(
    message: Message,
    *,
    threshold_tokens: int,
    estimator: TokenEstimator,
) -> bool:
    """判断工具消息是否需要内容级压缩。"""

    if not message.is_tool:
        return False
    return estimate_message_tokens(message, estimator=estimator) > threshold_tokens


def build_compressed_tool_message(message: Message, summary: SummaryResult) -> Message:
    """用工具摘要替换原工具消息内容。"""

    metadata = dict(message.metadata)
    metadata.update(
        {
            "compressed": True,
            "compression_kind": SummaryKind.TOOL_RESULT,
            "source_message_id": message.id,
            "summary_metadata": summary.metadata,
        }
    )
    return Message.tool(
        id=message.id,
        content=summary.summary,
        tool_call_id=message.tool_call_id or "",
        name=message.name,
        metadata=metadata,
        created_at=message.created_at,
    )


def build_compression_metadata(
    *,
    original_state: AgentState,
    output_state: AgentState,
    keep_recent_turns: int,
    tool_result_token_threshold: int,
    history_summary: Optional[SummaryResult],
    tool_summaries: Iterable[SummaryResult],
) -> Dict[str, object]:
    """构建适合写入 State 和 BuiltContext 的压缩摘要信息。"""

    tool_summary_tuple = tuple(tool_summaries)
    return {
        "enabled": True,
        "keep_recent_turns": keep_recent_turns,
        "tool_result_token_threshold": tool_result_token_threshold,
        "input_messages": len(original_state.messages),
        "output_messages": len(output_state.messages),
        "history_compressed": history_summary is not None,
        "history_source_message_count": (
            history_summary.source_message_count
            if history_summary is not None
            else 0
        ),
        "tool_result_compressed_count": len(tool_summary_tuple),
        "tool_source_message_count": sum(
            summary.source_message_count
            for summary in tool_summary_tuple
        ),
    }


def merge_state_compression_metadata(
    metadata: Dict[str, object],
    compression_metadata: Dict[str, object],
) -> Dict[str, object]:
    """把压缩信息写入 state metadata。"""

    merged = dict(metadata)
    merged["context_compression"] = compression_metadata
    return merged


__all__ = [
    "ContextCompressionResult",
    "ContextCompressor",
    "build_compressed_tool_message",
    "build_compression_metadata",
    "merge_state_compression_metadata",
    "should_compress_tool_message",
    "split_history_by_recent_turns",
]
