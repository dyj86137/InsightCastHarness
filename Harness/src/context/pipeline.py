"""分层 Prompt 拼装管线。

Pipeline 负责把 AgentState 中的不同上下文来源组装成有层次的 PromptSegment。
窗口裁剪、token 预算和压缩不在这里直接完成，避免一个模块同时承担太多职责。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from core.message import Message
from core.state import AgentState
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from context.token import (
    DEFAULT_TOKEN_ESTIMATOR,
    TokenEstimator,
    estimate_messages_tokens,
)


class PromptLayerType(str, Enum):
    """Prompt 分层类型。"""

    SYSTEM = "system"
    SUMMARY = "summary"
    MEMORY = "memory"
    RUNTIME = "runtime"
    HISTORY = "history"
    TOOL_RESULT = "tool_result"
    CUSTOM = "custom"


class PromptSegment(SerializableModel):
    """Prompt 管线中的一个上下文片段。"""

    name: str
    layer: PromptLayerType
    order: int
    required: bool = False
    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    estimated_tokens: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 PromptSegment。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError("PromptSegment requires non-empty name.")
        if self.order < 0:
            raise HarnessValidationError(
                "PromptSegment order cannot be negative.",
                details={"name": self.name, "order": self.order},
            )
        if self.estimated_tokens < 0:
            raise HarnessValidationError(
                "PromptSegment estimated_tokens cannot be negative.",
                details={"name": self.name, "estimated_tokens": self.estimated_tokens},
            )


class PromptAssemblyResult(SerializableModel):
    """分层 Prompt 拼装结果。"""

    segments: Tuple[PromptSegment, ...] = Field(default_factory=tuple)
    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    estimated_tokens: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 PromptAssemblyResult。"""

        if self.estimated_tokens < 0:
            raise HarnessValidationError(
                "PromptAssemblyResult estimated_tokens cannot be negative.",
                details={"estimated_tokens": self.estimated_tokens},
            )
        if self.segments and not self.messages:
            raise HarnessValidationError(
                "PromptAssemblyResult cannot have segments without messages."
            )


class PromptPipelineContext(SerializableModel):
    """Prompt 管线执行上下文。"""

    run_id: str
    session_id: Optional[str] = None
    step: int = 0
    system_message: Optional[Message] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class PromptPipelineStage(ABC):
    """Prompt 管线阶段。"""

    name: str
    layer: PromptLayerType
    order: int
    required: bool = False

    @abstractmethod
    def build(
        self,
        state: AgentState,
        context: PromptPipelineContext,
        estimator: TokenEstimator,
    ) -> List[PromptSegment]:
        """根据 AgentState 构建一个或多个 PromptSegment。"""


class SystemPromptStage(PromptPipelineStage):
    """系统提示层。"""

    name = "system"
    layer = PromptLayerType.SYSTEM
    order = 0
    required = True

    def build(
        self,
        state: AgentState,
        context: PromptPipelineContext,
        estimator: TokenEstimator,
    ) -> List[PromptSegment]:
        """构建系统提示片段。"""

        del state
        if context.system_message is None:
            return []
        return [
            create_prompt_segment(
                name=self.name,
                layer=self.layer,
                order=self.order,
                required=self.required,
                messages=(context.system_message,),
                estimator=estimator,
            )
        ]


class SummaryPromptStage(PromptPipelineStage):
    """会话摘要层。"""

    name = "summary"
    layer = PromptLayerType.SUMMARY
    order = 10
    required = False

    def __init__(self, *, prefix: str = "Conversation summary:") -> None:
        self.prefix = prefix

    def build(
        self,
        state: AgentState,
        context: PromptPipelineContext,
        estimator: TokenEstimator,
    ) -> List[PromptSegment]:
        """构建摘要片段。"""

        del context
        if not state.summary:
            return []
        summary_message = Message.system(f"{self.prefix}\n{state.summary}")
        return [
            create_prompt_segment(
                name=self.name,
                layer=self.layer,
                order=self.order,
                required=self.required,
                messages=(summary_message,),
                estimator=estimator,
            )
        ]


class HistoryPromptStage(PromptPipelineStage):
    """历史消息层。"""

    name = "history"
    layer = PromptLayerType.HISTORY
    order = 100
    required = True

    def build(
        self,
        state: AgentState,
        context: PromptPipelineContext,
        estimator: TokenEstimator,
    ) -> List[PromptSegment]:
        """构建历史消息片段。"""

        del context
        if not state.messages:
            return []
        return [
            create_prompt_segment(
                name=self.name,
                layer=self.layer,
                order=self.order,
                required=self.required,
                messages=state.messages,
                estimator=estimator,
                metadata={"role_counts": count_messages_by_role(state.messages)},
            )
        ]


class StaticPromptStage(PromptPipelineStage):
    """静态消息层，可用于记忆注入、运行时提示或自定义上下文。"""

    def __init__(
        self,
        *,
        name: str,
        layer: PromptLayerType,
        order: int,
        messages: Iterable[Message],
        required: bool = False,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self.name = name
        self.layer = layer
        self.order = order
        self.required = required
        self.messages = tuple(messages)
        self.metadata = metadata or {}

    def build(
        self,
        state: AgentState,
        context: PromptPipelineContext,
        estimator: TokenEstimator,
    ) -> List[PromptSegment]:
        """构建静态片段。"""

        del state, context
        if not self.messages:
            return []
        return [
            create_prompt_segment(
                name=self.name,
                layer=self.layer,
                order=self.order,
                required=self.required,
                messages=self.messages,
                estimator=estimator,
                metadata=self.metadata,
            )
        ]


class LayeredPromptPipeline:
    """分层 Prompt 拼装管线。"""

    def __init__(
        self,
        stages: Optional[Iterable[PromptPipelineStage]] = None,
        *,
        estimator: Optional[TokenEstimator] = None,
    ) -> None:
        self.stages = tuple(stages) if stages is not None else default_prompt_stages()
        self.estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
        validate_prompt_stages(self.stages)

    def assemble(
        self,
        state: AgentState,
        *,
        system_message: Optional[Message] = None,
        extra_segments: Optional[Iterable[PromptSegment]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> PromptAssemblyResult:
        """执行分层 Prompt 拼装。"""

        context = PromptPipelineContext(
            run_id=state.run_id,
            session_id=state.session_id,
            step=state.step,
            system_message=system_message,
            metadata=metadata or {},
        )

        segments: List[PromptSegment] = []
        for stage in self.stages:
            segments.extend(stage.build(state, context, self.estimator))
        segments.extend(extra_segments or ())

        ordered_segments = tuple(sort_segments(segments))
        messages = tuple(flatten_segment_messages(ordered_segments))
        estimated_tokens = estimate_messages_tokens(messages, estimator=self.estimator)
        return PromptAssemblyResult(
            segments=ordered_segments,
            messages=messages,
            estimated_tokens=estimated_tokens,
            metadata={
                "run_id": state.run_id,
                "session_id": state.session_id,
                "step": state.step,
                "estimator": self.estimator.name,
                "segment_count": len(ordered_segments),
                "message_count": len(messages),
                "segments": [segment_to_metadata(segment) for segment in ordered_segments],
                "tokens_by_layer": sum_tokens_by_layer(ordered_segments),
            },
        )


def default_prompt_stages() -> Tuple[PromptPipelineStage, ...]:
    """返回默认 Prompt 管线阶段。"""

    return (
        SystemPromptStage(),
        SummaryPromptStage(),
        HistoryPromptStage(),
    )


def validate_prompt_stages(stages: Iterable[PromptPipelineStage]) -> None:
    """校验 Prompt 管线阶段。"""

    seen = set()
    for stage in stages:
        if not isinstance(stage, PromptPipelineStage):
            raise HarnessValidationError(
                "LayeredPromptPipeline stages must be PromptPipelineStage instances.",
                details={"actual_type": stage.__class__.__name__},
            )
        if not stage.name or not stage.name.strip():
            raise HarnessValidationError("PromptPipelineStage requires non-empty name.")
        key = (stage.name, stage.order)
        if key in seen:
            raise HarnessValidationError(
                "PromptPipelineStage name and order must be unique.",
                details={"stage_name": stage.name, "order": stage.order},
            )
        seen.add(key)


def create_prompt_segment(
    *,
    name: str,
    layer: PromptLayerType,
    order: int,
    required: bool,
    messages: Iterable[Message],
    estimator: TokenEstimator,
    metadata: Optional[Dict[str, object]] = None,
) -> PromptSegment:
    """创建 PromptSegment 并估算 token。"""

    message_tuple = tuple(messages)
    return PromptSegment(
        name=name,
        layer=layer,
        order=order,
        required=required,
        messages=message_tuple,
        estimated_tokens=estimate_messages_tokens(message_tuple, estimator=estimator),
        metadata=metadata or {},
    )


def sort_segments(segments: Iterable[PromptSegment]) -> List[PromptSegment]:
    """按 order 排序 PromptSegment。"""

    return sorted(segments, key=lambda segment: (segment.order, segment.name))


def flatten_segment_messages(segments: Iterable[PromptSegment]) -> List[Message]:
    """把多个 PromptSegment 展平为消息列表。"""

    messages: List[Message] = []
    for segment in segments:
        messages.extend(segment.messages)
    return dedupe_messages_preserve_order(messages)


def dedupe_messages_preserve_order(messages: Iterable[Message]) -> List[Message]:
    """按消息 ID 去重并保持顺序。"""

    seen = set()
    result: List[Message] = []
    for message in messages:
        if message.id in seen:
            continue
        seen.add(message.id)
        result.append(message)
    return result


def count_messages_by_role(messages: Iterable[Message]) -> Dict[str, int]:
    """按角色统计消息数量。"""

    counts: Dict[str, int] = {}
    for message in messages:
        role = message.role.value
        counts[role] = counts.get(role, 0) + 1
    return counts


def sum_tokens_by_layer(segments: Iterable[PromptSegment]) -> Dict[str, int]:
    """按层汇总 token 估算值。"""

    totals: Dict[str, int] = {}
    for segment in segments:
        layer = segment.layer.value
        totals[layer] = totals.get(layer, 0) + segment.estimated_tokens
    return totals


def segment_to_metadata(segment: PromptSegment) -> Dict[str, object]:
    """把 PromptSegment 转换成适合写入 metadata 的摘要。"""

    return {
        "name": segment.name,
        "layer": segment.layer.value,
        "order": segment.order,
        "required": segment.required,
        "message_count": len(segment.messages),
        "estimated_tokens": segment.estimated_tokens,
        "metadata": segment.metadata,
    }


__all__ = [
    "HistoryPromptStage",
    "LayeredPromptPipeline",
    "PromptAssemblyResult",
    "PromptLayerType",
    "PromptPipelineContext",
    "PromptPipelineStage",
    "PromptSegment",
    "StaticPromptStage",
    "SummaryPromptStage",
    "SystemPromptStage",
    "count_messages_by_role",
    "create_prompt_segment",
    "dedupe_messages_preserve_order",
    "default_prompt_stages",
    "flatten_segment_messages",
    "segment_to_metadata",
    "sort_segments",
    "sum_tokens_by_layer",
    "validate_prompt_stages",
]
