"""LLM 用量统计与成本估算。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from pydantic import Field

from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import ChatResponse, LLMUsage


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_usage_record_id() -> str:
    return f"usage_{uuid4().hex}"


class ModelPricing(SerializableModel):
    """模型 token 价格配置。"""

    provider: str
    model: str
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    cached_prompt_cost_per_1k: float = 0.0
    currency: str = "USD"
    metadata: Dict[str, str] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验价格配置。"""

        if not self.provider or not self.provider.strip():
            raise HarnessValidationError("ModelPricing requires provider.")
        if not self.model or not self.model.strip():
            raise HarnessValidationError(
                "ModelPricing requires model.",
                details={"provider": self.provider},
            )
        if not self.currency or not self.currency.strip():
            raise HarnessValidationError(
                "ModelPricing requires currency.",
                details={"provider": self.provider, "model": self.model},
            )
        for field_name in (
            "prompt_cost_per_1k",
            "completion_cost_per_1k",
            "cached_prompt_cost_per_1k",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "ModelPricing cost cannot be negative.",
                    details={
                        "provider": self.provider,
                        "model": self.model,
                        "field": field_name,
                        "value": value,
                    },
                )

    def estimate_cost(self, usage: LLMUsage) -> float:
        """根据 token 用量估算成本。"""

        cached_tokens = min(usage.cached_tokens, usage.prompt_tokens)
        billable_prompt_tokens = max(usage.prompt_tokens - cached_tokens, 0)
        prompt_cost = billable_prompt_tokens / 1000 * self.prompt_cost_per_1k
        cached_cost = cached_tokens / 1000 * self.cached_prompt_cost_per_1k
        completion_cost = usage.completion_tokens / 1000 * self.completion_cost_per_1k
        return prompt_cost + cached_cost + completion_cost


class UsageRecord(SerializableModel):
    """一次 LLM 调用产生的用量记录。"""

    id: str = Field(default_factory=_new_usage_record_id)
    provider: str
    model: str
    usage: LLMUsage
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    response_id: Optional[str] = None
    estimated_cost: Optional[float] = None
    currency: str = "USD"
    metadata: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验用量记录。"""

        if not self.provider or not self.provider.strip():
            raise HarnessValidationError("UsageRecord requires provider.")
        if not self.model or not self.model.strip():
            raise HarnessValidationError(
                "UsageRecord requires model.",
                details={"provider": self.provider},
            )
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise HarnessValidationError(
                "UsageRecord estimated_cost cannot be negative.",
                details={"record_id": self.id, "estimated_cost": self.estimated_cost},
            )


class UsageTracker:
    """内存版 LLM 用量统计器。"""

    def __init__(self, pricing: Optional[Iterable[ModelPricing]] = None) -> None:
        self._pricing: Dict[Tuple[str, str], ModelPricing] = {}
        self._records: List[UsageRecord] = []
        for item in pricing or ():
            self.set_pricing(item)

    def set_pricing(self, pricing: ModelPricing) -> None:
        """设置或覆盖某个模型的价格。"""

        self._pricing[pricing_key(pricing.provider, pricing.model)] = pricing

    def get_pricing(self, provider: str, model: str) -> Optional[ModelPricing]:
        """获取模型价格配置。"""

        return self._pricing.get(pricing_key(provider, model))

    def estimate_cost(self, provider: str, model: str, usage: LLMUsage) -> Optional[float]:
        """估算某次调用成本，没有价格配置时返回 None。"""

        pricing = self.get_pricing(provider, model)
        if pricing is None:
            return None
        return pricing.estimate_cost(usage)

    def add_record(self, record: UsageRecord) -> UsageRecord:
        """追加一条用量记录。"""

        self._records.append(record)
        return record

    def add_response(
        self,
        response: ChatResponse,
        *,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UsageRecord:
        """从 ChatResponse 生成并追加用量记录。"""

        pricing = self.get_pricing(response.provider, response.model)
        estimated_cost = pricing.estimate_cost(response.usage) if pricing else None
        currency = pricing.currency if pricing else "USD"
        record = UsageRecord(
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            run_id=run_id,
            session_id=session_id,
            request_id=response.request_id,
            response_id=response.id,
            estimated_cost=estimated_cost,
            currency=currency,
            metadata=metadata or {},
        )
        return self.add_record(record)

    def records(self) -> List[UsageRecord]:
        """返回所有用量记录副本。"""

        return list(self._records)

    def total_usage(self) -> LLMUsage:
        """汇总所有记录的 token 用量。"""

        total = LLMUsage.empty()
        for record in self._records:
            total = total.add(record.usage)
        return total

    def total_cost(self) -> float:
        """汇总所有可估算的成本。"""

        return sum(record.estimated_cost or 0.0 for record in self._records)

    def records_for_run(self, run_id: str) -> List[UsageRecord]:
        """返回指定 run 的用量记录。"""

        return [record for record in self._records if record.run_id == run_id]

    def total_usage_for_run(self, run_id: str) -> LLMUsage:
        """汇总指定 run 的 token 用量。"""

        total = LLMUsage.empty()
        for record in self.records_for_run(run_id):
            total = total.add(record.usage)
        return total

    def total_cost_for_run(self, run_id: str) -> float:
        """汇总指定 run 的可估算成本。"""

        return sum(record.estimated_cost or 0.0 for record in self.records_for_run(run_id))

    def reset(self) -> None:
        """清空用量记录。"""

        self._records.clear()


def pricing_key(provider: str, model: str) -> Tuple[str, str]:
    """生成价格表 key。"""

    return provider.strip().lower(), model.strip().lower()


__all__ = [
    "ModelPricing",
    "UsageRecord",
    "UsageTracker",
    "pricing_key",
]
