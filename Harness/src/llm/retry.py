"""LLM 重试策略与包装器。"""

from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator, Optional

from infra.exception import HarnessError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest, ChatResponse, StreamChunk


class RetryPolicy(SerializableModel):
    """LLM 调用重试策略。"""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    backoff_factor: float = 2.0
    jitter_ratio: float = 0.0

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验重试策略参数。"""

        if self.max_attempts < 1:
            raise HarnessValidationError(
                "RetryPolicy max_attempts must be at least 1.",
                details={"max_attempts": self.max_attempts},
            )
        if self.initial_delay_seconds < 0:
            raise HarnessValidationError(
                "RetryPolicy initial_delay_seconds cannot be negative.",
                details={"initial_delay_seconds": self.initial_delay_seconds},
            )
        if self.max_delay_seconds < 0:
            raise HarnessValidationError(
                "RetryPolicy max_delay_seconds cannot be negative.",
                details={"max_delay_seconds": self.max_delay_seconds},
            )
        if self.backoff_factor < 1:
            raise HarnessValidationError(
                "RetryPolicy backoff_factor must be at least 1.",
                details={"backoff_factor": self.backoff_factor},
            )
        if self.jitter_ratio < 0 or self.jitter_ratio > 1:
            raise HarnessValidationError(
                "RetryPolicy jitter_ratio must be between 0 and 1.",
                details={"jitter_ratio": self.jitter_ratio},
            )

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断某次失败是否还应该重试。"""

        if attempt >= self.max_attempts:
            return False
        return is_retryable_error(error)

    def delay_for_attempt(self, attempt: int) -> float:
        """计算某次重试前的等待时间。attempt 从 1 开始。"""

        delay = self.initial_delay_seconds * (self.backoff_factor ** max(0, attempt - 1))
        delay = min(delay, self.max_delay_seconds)
        if self.jitter_ratio:
            lower = 1 - self.jitter_ratio
            upper = 1 + self.jitter_ratio
            delay *= random.uniform(lower, upper)
        return delay


class RetryLLM(BaseLLM):
    """为任意 BaseLLM 增加重试能力的包装器。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._llm = llm
        self._policy = policy or RetryPolicy()

    @property
    def provider_name(self) -> str:
        """返回被包装 Provider 的名称。"""

        return self._llm.provider_name

    @property
    def model_name(self) -> str:
        """返回被包装 Provider 的默认模型名称。"""

        return self._llm.model_name

    @property
    def policy(self) -> RetryPolicy:
        """返回当前重试策略。"""

        return self._policy

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """带重试地执行非流式 Chat 调用。"""

        attempt = 1
        while True:
            try:
                return await self._llm.chat(request)
            except Exception as exc:
                if not self._policy.should_retry(exc, attempt):
                    raise
                await asyncio.sleep(self._policy.delay_for_attempt(attempt))
                attempt += 1

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """带重试地执行流式调用。

        如果流式输出已经产生了分片，中途失败不再自动重试，避免重复输出污染上层状态。
        """

        attempt = 1
        while True:
            emitted = False
            try:
                async for chunk in self._llm.stream(request):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted or not self._policy.should_retry(exc, attempt):
                    raise
                await asyncio.sleep(self._policy.delay_for_attempt(attempt))
                attempt += 1


def is_retryable_error(error: Exception) -> bool:
    """判断异常是否适合由 LLM 层自动重试。"""

    if isinstance(error, HarnessError):
        return error.retryable
    return isinstance(error, (TimeoutError, ConnectionError))


def with_retry(llm: BaseLLM, policy: Optional[RetryPolicy] = None) -> RetryLLM:
    """把任意 LLM 包装为带重试能力的 LLM。"""

    return RetryLLM(llm, policy=policy)


__all__ = [
    "RetryLLM",
    "RetryPolicy",
    "is_retryable_error",
    "with_retry",
]
