"""上下文窗口选择策略。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.config import ContextConfig
from infra.exception import ContextError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from context.budget import (
    PromptContentBudget,
    TokenBudget,
    TokenBudgetManager,
    TokenBudgetUsage,
)
from context.token import (
    DEFAULT_TOKEN_ESTIMATOR,
    TokenEstimator,
    estimate_messages_tokens,
    estimate_text_tokens,
)


class ContextWindowSelection(SerializableModel):
    """上下文窗口选择结果。"""

    messages: Tuple[Message, ...] = Field(default_factory=tuple)
    estimated_tokens: int = 0
    dropped_messages: int = 0
    dropped_by_count: int = 0
    dropped_by_tokens: int = 0
    budget_usage: Optional[TokenBudgetUsage] = None
    content_budget: Optional[PromptContentBudget] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验窗口选择结果。"""

        for field_name in (
            "estimated_tokens",
            "dropped_messages",
            "dropped_by_count",
            "dropped_by_tokens",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "ContextWindowSelection count fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )


class ContextWindow:
    """上下文窗口选择策略。

    当前策略优先保留 system 消息，然后保留最近的非 system 消息；
    如果 token 预算超限，则继续从最早的非 system 消息开始裁剪。
    """

    def __init__(
        self,
        *,
        max_messages: int,
        max_input_tokens: int,
        preserve_system_messages: bool = True,
        token_estimator: Optional[TokenEstimator] = None,
        budget_manager: Optional[TokenBudgetManager] = None,
    ) -> None:
        if max_messages <= 0:
            raise ContextError(
                "ContextWindow max_messages must be positive.",
                details={"max_messages": max_messages},
            )
        if max_input_tokens <= 0:
            raise ContextError(
                "ContextWindow max_input_tokens must be positive.",
                details={"max_input_tokens": max_input_tokens},
            )

        self.max_messages = max_messages
        self.max_input_tokens = max_input_tokens
        self.preserve_system_messages = preserve_system_messages
        self.token_estimator = token_estimator or DEFAULT_TOKEN_ESTIMATOR
        self.budget_manager = budget_manager or TokenBudgetManager.from_config(
            ContextConfig(max_messages=max_messages, max_input_tokens=max_input_tokens),
            estimator=self.token_estimator,
        )

    @classmethod
    def from_config(cls, config: ContextConfig) -> "ContextWindow":
        """根据 ContextConfig 创建窗口策略。"""

        return cls(
            max_messages=config.max_messages,
            max_input_tokens=config.max_input_tokens,
            preserve_system_messages=True,
            budget_manager=TokenBudgetManager.from_config(config),
        )

    def select(self, messages: Iterable[Message]) -> ContextWindowSelection:
        """选择最终进入 LLM 输入的消息窗口。"""

        original_messages = tuple(messages)
        budget = self.budget_manager.build_budget()
        content_budget = self.budget_manager.build_content_budget(budget=budget)
        selected, dropped_by_count = self._select_by_message_count(original_messages)
        selected, dropped_by_tokens = self._trim_by_token_budget(selected, budget=budget)
        budget_usage = self.budget_manager.measure(selected, budget=budget)
        estimated_tokens = budget_usage.used_input_tokens

        return ContextWindowSelection(
            messages=tuple(selected),
            estimated_tokens=estimated_tokens,
            dropped_messages=dropped_by_count + dropped_by_tokens,
            dropped_by_count=dropped_by_count,
            dropped_by_tokens=dropped_by_tokens,
            budget_usage=budget_usage,
            content_budget=content_budget,
            metadata={
                "max_messages": self.max_messages,
                "max_input_tokens": self.max_input_tokens,
                "available_input_tokens": budget.available_input_tokens,
                "reserved_output_tokens": budget.reserved_output_tokens,
                "safety_margin_tokens": budget.safety_margin_tokens,
                "max_tool_result_tokens": budget.max_tool_result_tokens,
                "content_budget": content_budget.to_dict(exclude_none=True),
                "preserve_system_messages": self.preserve_system_messages,
                "token_estimator": self.token_estimator.name,
                "input_messages": len(original_messages),
                "output_messages": len(selected),
            },
        )

    def _select_by_message_count(
        self,
        messages: Tuple[Message, ...],
    ) -> Tuple[List[Message], int]:
        """按最大消息数选择最近消息。"""

        if not messages:
            return [], 0

        if not self.preserve_system_messages:
            selected = list(messages[-self.max_messages :])
            return selected, max(len(messages) - len(selected), 0)

        system_messages = [message for message in messages if message.is_system]
        non_system_messages = [message for message in messages if not message.is_system]

        remaining_slots = max(self.max_messages - len(system_messages), 0)
        selected_non_system = (
            non_system_messages[-remaining_slots:] if remaining_slots else []
        )
        selected = dedupe_preserve_order(system_messages + selected_non_system)
        dropped = max(len(messages) - len(selected), 0)
        return selected, dropped

    def _trim_by_token_budget(
        self,
        messages: List[Message],
        *,
        budget: TokenBudget,
    ) -> Tuple[List[Message], int]:
        """按 token 预算裁剪消息。"""

        selected = list(messages)
        dropped = 0

        while (
            selected
            and estimate_messages_tokens(selected, estimator=self.token_estimator)
            > budget.available_input_tokens
        ):
            removable_index = first_removable_message_index(
                selected,
                preserve_system_messages=self.preserve_system_messages,
            )
            if removable_index is None:
                break
            del selected[removable_index]
            dropped += 1

        return selected, dropped


def first_removable_message_index(
    messages: List[Message],
    *,
    preserve_system_messages: bool = True,
) -> Optional[int]:
    """返回第一条可裁剪消息的位置。"""

    for index, message in enumerate(messages):
        if preserve_system_messages and message.is_system:
            continue
        return index
    return None


def dedupe_preserve_order(messages: Iterable[Message]) -> List[Message]:
    """按消息 ID 去重并保持顺序。"""

    seen = set()
    result = []
    for message in messages:
        if message.id in seen:
            continue
        seen.add(message.id)
        result.append(message)
    return result

__all__ = [
    "ContextWindow",
    "ContextWindowSelection",
    "dedupe_preserve_order",
    "estimate_messages_tokens",
    "estimate_text_tokens",
    "first_removable_message_index",
]
