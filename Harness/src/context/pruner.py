"""智能上下文裁剪。

Pruner 基于分层 PromptSegment 和两层 token 预算执行消息级裁剪。
它优先保留 system 和当前用户输入，再按内容分区预算、消息优先级和新近程度裁剪低价值内容。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.config import ContextConfig
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from context.budget import (
    PromptContentBudget,
    PromptContentSection,
    TokenBudget,
    TokenBudgetManager,
)
from context.pipeline import PromptLayerType, PromptSegment
from context.token import (
    DEFAULT_TOKEN_ESTIMATOR,
    TokenEstimator,
    estimate_message_tokens,
    estimate_messages_tokens,
)


class PruneReason(str, Enum):
    """裁剪原因。"""

    KEPT_REQUIRED = "kept_required"
    KEPT_WITHIN_BUDGET = "kept_within_budget"
    SECTION_BUDGET_EXCEEDED = "section_budget_exceeded"
    TOTAL_BUDGET_EXCEEDED = "total_budget_exceeded"
    MESSAGE_COUNT_EXCEEDED = "message_count_exceeded"


class PrunableMessage(SerializableModel):
    """可裁剪消息的结构化描述。"""

    message: Message
    segment_name: str
    layer: PromptLayerType
    section: PromptContentSection
    original_index: int
    tokens: int
    priority: int
    required: bool = False
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验可裁剪消息。"""

        if self.original_index < 0:
            raise HarnessValidationError(
                "PrunableMessage original_index cannot be negative.",
                details={"original_index": self.original_index},
            )
        if self.tokens < 0:
            raise HarnessValidationError(
                "PrunableMessage tokens cannot be negative.",
                details={"message_id": self.message.id, "tokens": self.tokens},
            )
        if self.priority < 0:
            raise HarnessValidationError(
                "PrunableMessage priority cannot be negative.",
                details={"message_id": self.message.id, "priority": self.priority},
            )


class PruneDecision(SerializableModel):
    """单条消息的裁剪决策。"""

    message_id: str
    section: PromptContentSection
    kept: bool
    reason: PruneReason
    tokens: int
    segment_name: str
    original_index: int
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class ContextPruneResult(SerializableModel):
    """智能裁剪结果。"""

    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    decisions: Tuple[PruneDecision, ...] = Field(default_factory=tuple)
    resource_budget: TokenBudget
    content_budget: PromptContentBudget
    estimated_tokens: int = 0
    kept_tokens: int = 0
    dropped_tokens: int = 0
    dropped_messages: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验智能裁剪结果。"""

        for field_name in ("estimated_tokens", "kept_tokens", "dropped_tokens", "dropped_messages"):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "ContextPruneResult count fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )


class ContextPruner:
    """基于内容分区和优先级的智能裁剪器。"""

    def __init__(
        self,
        *,
        budget_manager: TokenBudgetManager,
        max_messages: Optional[int] = None,
        estimator: Optional[TokenEstimator] = None,
    ) -> None:
        if max_messages is not None and max_messages <= 0:
            raise HarnessValidationError(
                "ContextPruner max_messages must be positive.",
                details={"max_messages": max_messages},
            )
        self.budget_manager = budget_manager
        self.max_messages = max_messages
        self.estimator = estimator or DEFAULT_TOKEN_ESTIMATOR

    @classmethod
    def from_config(
        cls,
        config: ContextConfig,
        *,
        estimator: Optional[TokenEstimator] = None,
    ) -> "ContextPruner":
        """根据 ContextConfig 创建智能裁剪器。"""

        selected_estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
        return cls(
            budget_manager=TokenBudgetManager.from_config(
                config,
                estimator=selected_estimator,
            ),
            max_messages=config.max_messages,
            estimator=selected_estimator,
        )

    def prune(self, segments: Iterable[PromptSegment]) -> ContextPruneResult:
        """执行智能裁剪。"""

        segment_tuple = tuple(segments)
        resource_budget = self.budget_manager.build_budget()
        content_budget = self.budget_manager.build_content_budget(budget=resource_budget)
        items = build_prunable_messages(segment_tuple, estimator=self.estimator)

        kept_items, decisions = self._apply_section_budgets(items, content_budget)
        kept_items, count_decisions = self._apply_message_count_budget(kept_items)
        decisions.extend(count_decisions)
        kept_items, total_decisions = self._apply_total_budget(kept_items, resource_budget)
        decisions.extend(total_decisions)

        kept_items = sorted(kept_items, key=lambda item: item.original_index)
        final_decisions = finalize_prune_decisions(items, decisions, kept_items)
        messages = tuple(item.message for item in kept_items)
        kept_tokens = estimate_messages_tokens(messages, estimator=self.estimator)
        original_tokens = sum(item.tokens for item in items)
        dropped_tokens = max(original_tokens - kept_tokens, 0)
        dropped_messages = len(items) - len(kept_items)

        return ContextPruneResult(
            messages=messages,
            decisions=tuple(final_decisions),
            resource_budget=resource_budget,
            content_budget=content_budget,
            estimated_tokens=kept_tokens,
            kept_tokens=kept_tokens,
            dropped_tokens=dropped_tokens,
            dropped_messages=dropped_messages,
            metadata={
                "input_messages": len(items),
                "output_messages": len(messages),
                "input_tokens": original_tokens,
                "output_tokens": kept_tokens,
                "kept_tokens": kept_tokens,
                "dropped_tokens": dropped_tokens,
                "dropped_messages": dropped_messages,
                "dropped_by_reason": count_decisions_by_reason(decisions),
                "final_dropped_by_reason": count_decisions_by_reason(final_decisions),
                "kept_by_section": count_kept_by_section(final_decisions),
            },
        )

    def _apply_section_budgets(
        self,
        items: Tuple[PrunableMessage, ...],
        content_budget: PromptContentBudget,
    ) -> Tuple[List[PrunableMessage], List[PruneDecision]]:
        """按内容分区预算裁剪。"""

        kept: List[PrunableMessage] = []
        decisions: List[PruneDecision] = []
        limits = content_budget.limits()
        for section in PromptContentSection:
            section_items = [item for item in items if item.section == section]
            if not section_items:
                continue
            section_limit = limits.get(section.value, 0)
            section_used = 0
            for item in sorted(section_items, key=keep_sort_key):
                if item.required:
                    kept.append(item)
                    section_used += item.tokens
                    decisions.append(make_decision(item, kept=True, reason=PruneReason.KEPT_REQUIRED))
                    continue
                if section_used + item.tokens <= section_limit:
                    kept.append(item)
                    section_used += item.tokens
                    decisions.append(make_decision(item, kept=True, reason=PruneReason.KEPT_WITHIN_BUDGET))
                else:
                    decisions.append(
                        make_decision(
                            item,
                            kept=False,
                            reason=PruneReason.SECTION_BUDGET_EXCEEDED,
                            metadata={"section_limit": section_limit, "section_used": section_used},
                        )
                    )
        return kept, decisions

    def _apply_message_count_budget(
        self,
        kept_items: List[PrunableMessage],
    ) -> Tuple[List[PrunableMessage], List[PruneDecision]]:
        """按最大消息数裁剪。"""

        if self.max_messages is None or len(kept_items) <= self.max_messages:
            return kept_items, []

        kept = list(kept_items)
        decisions: List[PruneDecision] = []
        while len(kept) > self.max_messages:
            index = select_drop_candidate_index(kept)
            if index is None:
                break
            item = kept.pop(index)
            decisions.append(
                make_decision(
                    item,
                    kept=False,
                    reason=PruneReason.MESSAGE_COUNT_EXCEEDED,
                    metadata={"max_messages": self.max_messages},
                )
            )
        return kept, decisions

    def _apply_total_budget(
        self,
        kept_items: List[PrunableMessage],
        resource_budget: TokenBudget,
    ) -> Tuple[List[PrunableMessage], List[PruneDecision]]:
        """按全局可用输入预算裁剪。"""

        kept = list(kept_items)
        decisions: List[PruneDecision] = []
        while sum(item.tokens for item in kept) > resource_budget.available_input_tokens:
            index = select_drop_candidate_index(kept)
            if index is None:
                break
            item = kept.pop(index)
            decisions.append(
                make_decision(
                    item,
                    kept=False,
                    reason=PruneReason.TOTAL_BUDGET_EXCEEDED,
                    metadata={"available_input_tokens": resource_budget.available_input_tokens},
                )
            )
        return kept, decisions


def build_prunable_messages(
    segments: Iterable[PromptSegment],
    *,
    estimator: TokenEstimator,
) -> Tuple[PrunableMessage, ...]:
    """把 PromptSegment 展开成可裁剪消息。"""

    segment_tuple = tuple(segments)
    last_user_message_id = find_last_user_message_id(segment_tuple)
    items: List[PrunableMessage] = []
    seen = set()
    original_index = 0
    for segment in segment_tuple:
        for message in segment.messages:
            if message.id in seen:
                continue
            seen.add(message.id)
            section = classify_message_section(
                segment,
                message,
                last_user_message_id=last_user_message_id,
            )
            required = is_required_message(segment, message, section)
            items.append(
                PrunableMessage(
                    message=message,
                    segment_name=segment.name,
                    layer=segment.layer,
                    section=section,
                    original_index=original_index,
                    tokens=estimate_message_tokens(message, estimator=estimator),
                    priority=message_priority(section, message, segment),
                    required=required,
                    metadata={
                        "message_role": message.role.value,
                        "segment_required": segment.required,
                        "is_last_user_message": message.id == last_user_message_id,
                    },
                )
            )
            original_index += 1
    return tuple(items)


def classify_message_section(
    segment: PromptSegment,
    message: Message,
    *,
    last_user_message_id: Optional[str],
) -> PromptContentSection:
    """识别消息所属内容预算分区。"""

    if segment.layer == PromptLayerType.SYSTEM:
        return PromptContentSection.SYSTEM
    if segment.layer == PromptLayerType.RUNTIME:
        return PromptContentSection.SYSTEM
    if segment.layer == PromptLayerType.MEMORY:
        return PromptContentSection.MEMORY
    if segment.layer == PromptLayerType.TOOL_RESULT:
        return PromptContentSection.TOOL_RESULTS
    if segment.layer == PromptLayerType.CUSTOM:
        return PromptContentSection.CUSTOM
    if segment.layer == PromptLayerType.SUMMARY:
        return PromptContentSection.HISTORY

    if message.id == last_user_message_id:
        return PromptContentSection.USER_INPUT
    if message.is_system:
        return PromptContentSection.SYSTEM
    if message.is_tool:
        return PromptContentSection.TOOL_RESULTS
    return PromptContentSection.HISTORY


def find_last_user_message_id(segments: Iterable[PromptSegment]) -> Optional[str]:
    """返回最后一条用户消息 ID。"""

    last_user_message_id = None
    for segment in segments:
        for message in segment.messages:
            if message.is_user:
                last_user_message_id = message.id
    return last_user_message_id


def is_required_message(
    segment: PromptSegment,
    message: Message,
    section: PromptContentSection,
) -> bool:
    """判断消息是否必须保留。"""

    if segment.required and message.is_system:
        return True
    if section == PromptContentSection.SYSTEM:
        return True
    if section == PromptContentSection.USER_INPUT:
        return True
    return False


def message_priority(
    section: PromptContentSection,
    message: Message,
    segment: PromptSegment,
) -> int:
    """返回消息优先级；数值越小越应该保留。"""

    del message
    base_priority = {
        PromptContentSection.SYSTEM: 0,
        PromptContentSection.USER_INPUT: 10,
        PromptContentSection.TOOL_DEFINITIONS: 20,
        PromptContentSection.RETRIEVAL_CONTEXT: 30,
        PromptContentSection.MEMORY: 40,
        PromptContentSection.TOOL_RESULTS: 50,
        PromptContentSection.HISTORY: 60,
        PromptContentSection.CUSTOM: 100,
    }[section]
    if segment.required:
        return max(base_priority - 5, 0)
    return base_priority


def keep_sort_key(item: PrunableMessage) -> Tuple[int, int, int]:
    """保留排序：required 优先、高优先级优先、较新的消息优先。"""

    required_rank = 0 if item.required else 1
    recency_rank = -item.original_index
    return (required_rank, item.priority, recency_rank)


def drop_sort_key(item: PrunableMessage) -> Tuple[int, int, int, int]:
    """丢弃排序：低优先级、较旧、较长的消息先丢。"""

    required_rank = 1 if item.required else 0
    return (required_rank, -item.priority, -item.tokens, item.original_index)


def select_drop_candidate_index(items: List[PrunableMessage]) -> Optional[int]:
    """选择一个可丢弃消息的位置。"""

    candidates = [
        (index, item)
        for index, item in enumerate(items)
        if not item.required
    ]
    if not candidates:
        return None
    index, _ = sorted(candidates, key=lambda pair: drop_sort_key(pair[1]))[0]
    return index


def make_decision(
    item: PrunableMessage,
    *,
    kept: bool,
    reason: PruneReason,
    metadata: Optional[Dict[str, object]] = None,
) -> PruneDecision:
    """创建裁剪决策。"""

    decision_metadata = dict(item.metadata)
    decision_metadata.update(metadata or {})
    return PruneDecision(
        message_id=item.message.id,
        section=item.section,
        kept=kept,
        reason=reason,
        tokens=item.tokens,
        segment_name=item.segment_name,
        original_index=item.original_index,
        metadata=decision_metadata,
    )


def finalize_prune_decisions(
    items: Iterable[PrunableMessage],
    decisions: Iterable[PruneDecision],
    kept_items: Iterable[PrunableMessage],
) -> List[PruneDecision]:
    """把多阶段裁剪记录收敛为每条消息的最终裁剪决策。"""

    item_tuple = tuple(items)
    kept_ids = {item.message.id for item in kept_items}
    latest_decision_by_id: Dict[str, PruneDecision] = {}
    for decision in decisions:
        latest_decision_by_id[decision.message_id] = decision

    final_decisions: List[PruneDecision] = []
    for item in item_tuple:
        latest_decision = latest_decision_by_id.get(item.message.id)
        if item.message.id in kept_ids:
            if latest_decision is not None and latest_decision.kept:
                final_decisions.append(latest_decision)
            else:
                reason = (
                    PruneReason.KEPT_REQUIRED
                    if item.required
                    else PruneReason.KEPT_WITHIN_BUDGET
                )
                final_decisions.append(
                    make_decision(
                        item,
                        kept=True,
                        reason=reason,
                        metadata={"finalized": True},
                    )
                )
            continue

        if latest_decision is not None and not latest_decision.kept:
            final_decisions.append(latest_decision)
            continue

        final_decisions.append(
            make_decision(
                item,
                kept=False,
                reason=PruneReason.TOTAL_BUDGET_EXCEEDED,
                metadata={"finalized": True},
            )
        )
    return final_decisions


def count_decisions_by_reason(decisions: Iterable[PruneDecision]) -> Dict[str, int]:
    """按裁剪原因统计数量。"""

    counts: Dict[str, int] = {}
    for decision in decisions:
        if decision.kept:
            continue
        reason = decision.reason.value
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def count_kept_by_section(decisions: Iterable[PruneDecision]) -> Dict[str, int]:
    """按内容分区统计保留数量。"""

    counts: Dict[str, int] = {}
    for decision in decisions:
        if not decision.kept:
            continue
        section = decision.section.value
        counts[section] = counts.get(section, 0) + 1
    return counts


__all__ = [
    "ContextPruneResult",
    "ContextPruner",
    "PrunableMessage",
    "PruneDecision",
    "PruneReason",
    "build_prunable_messages",
    "classify_message_section",
    "count_decisions_by_reason",
    "count_kept_by_section",
    "find_last_user_message_id",
    "finalize_prune_decisions",
    "is_required_message",
    "drop_sort_key",
    "keep_sort_key",
    "make_decision",
    "message_priority",
    "select_drop_candidate_index",
]
