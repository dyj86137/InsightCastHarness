"""上下文 token 预算管理。

预算系统把「最大输入 token」拆成可控的预算对象，后续分层 Prompt、智能裁剪、
工具结果压缩都应该依赖这里的预算结果，而不是到处直接读取 max_input_tokens。
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.config import ContextConfig
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from context.token import (
    DEFAULT_TOKEN_ESTIMATOR,
    TokenEstimator,
    estimate_messages_tokens,
)


class TokenBudgetSection(str, Enum):
    """底层资源预算分区。"""

    INPUT = "input"
    OUTPUT_RESERVE = "output_reserve"
    SAFETY_MARGIN = "safety_margin"
    TOOL_RESULT = "tool_result"


class PromptContentSection(str, Enum):
    """上层内容预算分区。"""

    SYSTEM = "system"
    TOOL_DEFINITIONS = "tool_definitions"
    RETRIEVAL_CONTEXT = "retrieval_context"
    MEMORY = "memory"
    HISTORY = "history"
    USER_INPUT = "user_input"
    TOOL_RESULTS = "tool_results"
    CUSTOM = "custom"


DEFAULT_CONTENT_BUDGET_RATIOS: Dict[str, float] = {
    PromptContentSection.SYSTEM.value: 0.10,
    PromptContentSection.TOOL_DEFINITIONS.value: 0.20,
    PromptContentSection.RETRIEVAL_CONTEXT.value: 0.25,
    PromptContentSection.HISTORY.value: 0.30,
    PromptContentSection.USER_INPUT.value: 0.10,
    PromptContentSection.TOOL_RESULTS.value: 0.05,
}


class TokenBudget(SerializableModel):
    """底层 token 预算。

    这个对象刻意保持简单，只回答「输入侧最多还能用多少 token」。
    上层 system、工具定义、检索上下文、历史记录怎么分配，由 PromptContentBudget 负责。
    """

    max_input_tokens: int
    available_input_tokens: int
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0
    max_tool_result_tokens: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 token 预算。"""

        for field_name in (
            "max_input_tokens",
            "available_input_tokens",
            "reserved_output_tokens",
            "safety_margin_tokens",
            "max_tool_result_tokens",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "TokenBudget fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )
        if self.max_input_tokens <= 0:
            raise HarnessValidationError(
                "TokenBudget max_input_tokens must be positive.",
                details={"max_input_tokens": self.max_input_tokens},
            )
        if self.available_input_tokens > self.max_input_tokens:
            raise HarnessValidationError(
                "TokenBudget available_input_tokens cannot exceed max_input_tokens.",
                details={
                    "available_input_tokens": self.available_input_tokens,
                    "max_input_tokens": self.max_input_tokens,
                },
            )

    @property
    def reserved_tokens(self) -> int:
        """返回不可用于历史消息的保留 token 数。"""

        return self.reserved_output_tokens + self.safety_margin_tokens

    @property
    def sections(self) -> Dict[str, int]:
        """返回预算分区字典，便于 Trace 和诊断。"""

        return {
            TokenBudgetSection.INPUT.value: self.available_input_tokens,
            TokenBudgetSection.OUTPUT_RESERVE.value: self.reserved_output_tokens,
            TokenBudgetSection.SAFETY_MARGIN.value: self.safety_margin_tokens,
            TokenBudgetSection.TOOL_RESULT.value: self.max_tool_result_tokens,
        }


class TokenBudgetUsage(SerializableModel):
    """一次上下文构建的预算使用情况。"""

    budget: TokenBudget
    used_input_tokens: int
    remaining_input_tokens: int
    overflow_tokens: int = 0
    message_count: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验预算使用情况。"""

        for field_name in (
            "used_input_tokens",
            "remaining_input_tokens",
            "overflow_tokens",
            "message_count",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "TokenBudgetUsage fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )

    @property
    def is_over_budget(self) -> bool:
        """判断是否超出输入预算。"""

        return self.overflow_tokens > 0


class PromptSectionBudget(SerializableModel):
    """单个内容分区的 token 预算。"""

    section: PromptContentSection
    token_limit: int
    ratio: float
    priority: int = 100
    required: bool = False
    elastic: bool = True
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验内容分区预算。"""

        if self.token_limit < 0:
            raise HarnessValidationError(
                "PromptSectionBudget token_limit cannot be negative.",
                details={"section": self.section.value, "token_limit": self.token_limit},
            )
        if self.ratio < 0:
            raise HarnessValidationError(
                "PromptSectionBudget ratio cannot be negative.",
                details={"section": self.section.value, "ratio": self.ratio},
            )
        if self.priority < 0:
            raise HarnessValidationError(
                "PromptSectionBudget priority cannot be negative.",
                details={"section": self.section.value, "priority": self.priority},
            )


class PromptContentBudget(SerializableModel):
    """上层内容预算。"""

    total_available_tokens: int
    sections: Tuple[PromptSectionBudget, ...] = Field(default_factory=tuple)
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验内容预算。"""

        if self.total_available_tokens < 0:
            raise HarnessValidationError(
                "PromptContentBudget total_available_tokens cannot be negative.",
                details={"total_available_tokens": self.total_available_tokens},
            )
        total_allocated = self.total_allocated_tokens
        if total_allocated > self.total_available_tokens:
            raise HarnessValidationError(
                "PromptContentBudget allocated tokens cannot exceed total available tokens.",
                details={
                    "total_allocated_tokens": total_allocated,
                    "total_available_tokens": self.total_available_tokens,
                },
            )

    @property
    def total_allocated_tokens(self) -> int:
        """返回已分配 token 总量。"""

        return sum(section.token_limit for section in self.sections)

    @property
    def unallocated_tokens(self) -> int:
        """返回尚未分配的 token 数。"""

        return max(self.total_available_tokens - self.total_allocated_tokens, 0)

    def limits(self) -> Dict[str, int]:
        """返回分区名到 token 上限的映射。"""

        return {
            section.section.value: section.token_limit
            for section in self.sections
        }

    def get_limit(self, section: PromptContentSection) -> int:
        """返回指定分区的 token 上限。"""

        return self.limits().get(section.value, 0)


class TokenBudgetPolicy(SerializableModel):
    """token 预算策略。"""

    max_input_tokens: int
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0
    max_tool_result_tokens: int = 2000
    content_budget_ratios: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CONTENT_BUDGET_RATIOS)
    )
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验预算策略。"""

        if self.max_input_tokens <= 0:
            raise HarnessValidationError(
                "TokenBudgetPolicy max_input_tokens must be positive.",
                details={"max_input_tokens": self.max_input_tokens},
            )
        for field_name in (
            "reserved_output_tokens",
            "safety_margin_tokens",
            "max_tool_result_tokens",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "TokenBudgetPolicy fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )
        validate_content_budget_ratios(self.content_budget_ratios)

    @classmethod
    def from_config(cls, config: ContextConfig) -> "TokenBudgetPolicy":
        """根据 ContextConfig 创建预算策略。"""

        return cls(
            max_input_tokens=config.max_input_tokens,
            reserved_output_tokens=getattr(config, "reserved_output_tokens", 0),
            safety_margin_tokens=getattr(config, "safety_margin_tokens", 0),
            max_tool_result_tokens=getattr(config, "max_tool_result_tokens", 2000),
            content_budget_ratios=dict(DEFAULT_CONTENT_BUDGET_RATIOS),
            metadata={"source": "ContextConfig"},
        )

    def build_budget(self) -> TokenBudget:
        """构建可用于一次上下文选择的预算。"""

        reserved_tokens = self.reserved_output_tokens + self.safety_margin_tokens
        available_input_tokens = max(self.max_input_tokens - reserved_tokens, 1)
        return TokenBudget(
            max_input_tokens=self.max_input_tokens,
            available_input_tokens=available_input_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
            max_tool_result_tokens=self.max_tool_result_tokens,
            metadata=dict(self.metadata),
        )

    def build_content_budget(
        self,
        budget: Optional[TokenBudget] = None,
    ) -> PromptContentBudget:
        """基于底层预算构建内容分区预算。"""

        selected_budget = budget or self.build_budget()
        return build_prompt_content_budget(
            selected_budget,
            ratios=self.content_budget_ratios,
            max_tool_result_tokens=self.max_tool_result_tokens,
            metadata={"source": "TokenBudgetPolicy"},
        )


class TokenBudgetManager:
    """预算计算入口。"""

    def __init__(
        self,
        *,
        policy: TokenBudgetPolicy,
        estimator: Optional[TokenEstimator] = None,
    ) -> None:
        self.policy = policy
        self.estimator = estimator or DEFAULT_TOKEN_ESTIMATOR

    @classmethod
    def from_config(
        cls,
        config: ContextConfig,
        *,
        estimator: Optional[TokenEstimator] = None,
    ) -> "TokenBudgetManager":
        """根据 ContextConfig 创建预算管理器。"""

        return cls(policy=TokenBudgetPolicy.from_config(config), estimator=estimator)

    def build_budget(self) -> TokenBudget:
        """构建 token 预算。"""

        return self.policy.build_budget()

    def build_content_budget(
        self,
        *,
        budget: Optional[TokenBudget] = None,
    ) -> PromptContentBudget:
        """构建内容分区预算。"""

        return self.policy.build_content_budget(budget=budget)

    def measure(
        self,
        messages: Iterable[Message],
        *,
        budget: Optional[TokenBudget] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TokenBudgetUsage:
        """测量消息集合的预算使用情况。"""

        selected_budget = budget or self.build_budget()
        message_tuple = tuple(messages)
        used_tokens = estimate_messages_tokens(message_tuple, estimator=self.estimator)
        overflow_tokens = max(used_tokens - selected_budget.available_input_tokens, 0)
        remaining_tokens = max(selected_budget.available_input_tokens - used_tokens, 0)
        usage_metadata = dict(metadata or {})
        usage_metadata["estimator"] = self.estimator.name
        return TokenBudgetUsage(
            budget=selected_budget,
            used_input_tokens=used_tokens,
            remaining_input_tokens=remaining_tokens,
            overflow_tokens=overflow_tokens,
            message_count=len(message_tuple),
            metadata=usage_metadata,
        )


def allocate_section_budget(total_tokens: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """按比例分配预算，返回每个分区的 token 上限。"""

    if total_tokens < 0:
        raise HarnessValidationError(
            "allocate_section_budget total_tokens cannot be negative.",
            details={"total_tokens": total_tokens},
        )
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise HarnessValidationError("allocate_section_budget ratios must sum to positive value.")
    allocations: Dict[str, int] = {}
    consumed = 0
    items = list(ratios.items())
    for index, (name, ratio) in enumerate(items):
        if ratio < 0:
            raise HarnessValidationError(
                "allocate_section_budget ratios cannot be negative.",
                details={"section": name, "ratio": ratio},
            )
        if index == len(items) - 1:
            allocation = max(total_tokens - consumed, 0)
        else:
            allocation = int(total_tokens * ratio / ratio_sum)
            consumed += allocation
        allocations[name] = allocation
    return allocations


def build_prompt_content_budget(
    budget: TokenBudget,
    *,
    ratios: Mapping[str, float],
    max_tool_result_tokens: int = 0,
    metadata: Optional[Dict[str, object]] = None,
) -> PromptContentBudget:
    """基于底层预算和比例构建内容分区预算。"""

    normalized_ratios = normalize_content_budget_ratios(ratios)
    allocations = allocate_section_budget(
        budget.available_input_tokens,
        normalized_ratios,
    )
    sections = []
    for section_name, token_limit in allocations.items():
        section = PromptContentSection(section_name)
        section_metadata: Dict[str, object] = {}
        if section == PromptContentSection.TOOL_RESULTS and max_tool_result_tokens > 0:
            section_metadata["per_tool_result_limit"] = max_tool_result_tokens
        sections.append(
            PromptSectionBudget(
                section=section,
                token_limit=token_limit,
                ratio=normalized_ratios[section_name],
                priority=default_section_priority(section),
                required=section in required_content_sections(),
                elastic=section not in required_content_sections(),
                metadata=section_metadata,
            )
        )

    budget_metadata = dict(metadata or {})
    budget_metadata.update(
        {
            "resource_budget": budget.to_dict(exclude_none=True),
            "ratios": dict(normalized_ratios),
        }
    )
    return PromptContentBudget(
        total_available_tokens=budget.available_input_tokens,
        sections=tuple(sections),
        metadata=budget_metadata,
    )


def normalize_content_budget_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    """规范化内容预算比例。"""

    normalized: Dict[str, float] = {}
    for raw_name, raw_ratio in ratios.items():
        section = parse_prompt_content_section(raw_name)
        normalized[section.value] = float(raw_ratio)
    validate_content_budget_ratios(normalized)
    return normalized


def validate_content_budget_ratios(ratios: Mapping[str, float]) -> None:
    """校验内容预算比例。"""

    if not ratios:
        raise HarnessValidationError("Content budget ratios cannot be empty.")
    ratio_sum = 0.0
    for raw_name, ratio in ratios.items():
        parse_prompt_content_section(raw_name)
        if ratio < 0:
            raise HarnessValidationError(
                "Content budget ratio cannot be negative.",
                details={"section": raw_name, "ratio": ratio},
            )
        ratio_sum += ratio
    if ratio_sum <= 0:
        raise HarnessValidationError("Content budget ratios must sum to positive value.")


def parse_prompt_content_section(value: object) -> PromptContentSection:
    """把输入值解析为 PromptContentSection。"""

    try:
        return PromptContentSection(str(value))
    except ValueError as exc:
        raise HarnessValidationError(
            "Unknown prompt content budget section.",
            details={"section": value},
            cause=exc,
        ) from exc


def required_content_sections() -> Tuple[PromptContentSection, ...]:
    """返回必须优先保留的内容分区。"""

    return (
        PromptContentSection.SYSTEM,
        PromptContentSection.USER_INPUT,
    )


def default_section_priority(section: PromptContentSection) -> int:
    """返回内容分区默认优先级；数值越小优先级越高。"""

    priorities = {
        PromptContentSection.SYSTEM: 0,
        PromptContentSection.USER_INPUT: 10,
        PromptContentSection.TOOL_DEFINITIONS: 20,
        PromptContentSection.RETRIEVAL_CONTEXT: 30,
        PromptContentSection.MEMORY: 40,
        PromptContentSection.TOOL_RESULTS: 50,
        PromptContentSection.HISTORY: 60,
        PromptContentSection.CUSTOM: 100,
    }
    return priorities[section]


def sum_token_usages(usages: Iterable[TokenBudgetUsage]) -> int:
    """汇总多个预算使用结果的输入 token 数。"""

    return sum(usage.used_input_tokens for usage in usages)


__all__ = [
    "DEFAULT_CONTENT_BUDGET_RATIOS",
    "PromptContentBudget",
    "PromptContentSection",
    "PromptSectionBudget",
    "TokenBudget",
    "TokenBudgetManager",
    "TokenBudgetPolicy",
    "TokenBudgetSection",
    "TokenBudgetUsage",
    "allocate_section_budget",
    "build_prompt_content_budget",
    "default_section_priority",
    "normalize_content_budget_ratios",
    "parse_prompt_content_section",
    "required_content_sections",
    "sum_token_usages",
    "validate_content_budget_ratios",
]
