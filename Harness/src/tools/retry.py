"""工具级重试策略。

重试属于 Harness 的执行管控能力，不属于单个工具自身的业务逻辑。
本模块只负责判断是否应该重试、计算退避等待时间、生成可写入 ToolResult 的重试元数据。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from core.tool import ToolResult, error_to_payload
from infra.exception import (
    HarnessError,
    ToolNotFoundError,
    ToolValidationError,
    ValidationError as HarnessValidationError,
)
from infra.serialization import SerializableModel


NON_RETRYABLE_ERROR_CODES = {
    "tool_not_found",
    "tool_validation_error",
    "validation_error",
}


class ToolRetryErrorRecord(SerializableModel):
    """一次失败尝试的结构化记录。"""

    attempt: int
    error: Dict[str, Any]
    delay_seconds: Optional[float] = None

    model_config = SerializableModel.config(frozen=True)


class ToolRetryPolicy(SerializableModel):
    """工具级重试策略。

    max_attempts 表示总尝试次数，包含第一次正常执行。
    默认只自动重试只读工具，避免写文件、发请求、写数据库等副作用被重复执行。
    """

    enabled: bool = True
    max_attempts: int = 2
    initial_delay_seconds: float = 0.1
    max_delay_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    retry_read_only_only: bool = True
    retry_timeout_errors: bool = True
    retryable_error_types: Tuple[str, ...] = Field(default_factory=tuple)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验重试策略配置。"""

        if self.max_attempts < 1:
            raise HarnessValidationError(
                "ToolRetryPolicy max_attempts must be greater than or equal to 1.",
                details={"max_attempts": self.max_attempts},
            )
        if self.initial_delay_seconds < 0:
            raise HarnessValidationError(
                "ToolRetryPolicy initial_delay_seconds cannot be negative.",
                details={"initial_delay_seconds": self.initial_delay_seconds},
            )
        if self.max_delay_seconds < 0:
            raise HarnessValidationError(
                "ToolRetryPolicy max_delay_seconds cannot be negative.",
                details={"max_delay_seconds": self.max_delay_seconds},
            )
        if self.backoff_multiplier < 1:
            raise HarnessValidationError(
                "ToolRetryPolicy backoff_multiplier must be greater than or equal to 1.",
                details={"backoff_multiplier": self.backoff_multiplier},
            )

    def should_retry_exception(
        self,
        *,
        attempt: int,
        is_read_only: bool,
        error: BaseException,
    ) -> bool:
        """判断异常是否应该触发下一次重试。"""

        if not self._can_retry(attempt=attempt, is_read_only=is_read_only):
            return False
        return is_retryable_exception(
            error,
            retry_timeout_errors=self.retry_timeout_errors,
            retryable_error_types=self.retryable_error_types,
        )

    def should_retry_result(
        self,
        *,
        attempt: int,
        is_read_only: bool,
        result: ToolResult,
    ) -> bool:
        """判断失败 ToolResult 是否应该触发下一次重试。"""

        if not self._can_retry(attempt=attempt, is_read_only=is_read_only):
            return False
        return is_retryable_result(result)

    def next_delay_seconds(self, failed_attempt: int) -> float:
        """计算某次失败后的退避等待时间。"""

        delay = self.initial_delay_seconds * (
            self.backoff_multiplier ** max(failed_attempt - 1, 0)
        )
        return min(delay, self.max_delay_seconds)

    def _can_retry(self, *, attempt: int, is_read_only: bool) -> bool:
        """判断当前尝试次数和工具属性是否允许继续重试。"""

        if not self.enabled:
            return False
        if attempt >= self.max_attempts:
            return False
        if self.retry_read_only_only and not is_read_only:
            return False
        return True


def is_retryable_exception(
    error: BaseException,
    *,
    retry_timeout_errors: bool = True,
    retryable_error_types: Tuple[str, ...] = (),
) -> bool:
    """判断异常是否属于可重试错误。"""

    if isinstance(error, (ToolNotFoundError, ToolValidationError)):
        return False
    if isinstance(error, TimeoutError):
        return retry_timeout_errors
    if retry_timeout_errors and isinstance(getattr(error, "cause", None), asyncio.TimeoutError):
        return True
    if error.__class__.__name__ in retryable_error_types:
        return True
    if isinstance(error, HarnessError):
        return bool(error.retryable)
    return False


def is_retryable_result(result: ToolResult) -> bool:
    """判断失败 ToolResult 是否属于可重试结果。"""

    if not result.is_error or not result.error:
        return False

    error_code = result.error.get("code")
    if error_code in NON_RETRYABLE_ERROR_CODES:
        return False
    return bool(result.error.get("retryable"))


def build_retry_record_from_exception(
    error: BaseException,
    *,
    attempt: int,
    delay_seconds: Optional[float],
) -> ToolRetryErrorRecord:
    """从异常创建重试记录。"""

    return ToolRetryErrorRecord(
        attempt=attempt,
        error=error_to_payload(error),
        delay_seconds=delay_seconds,
    )


def build_retry_record_from_result(
    result: ToolResult,
    *,
    attempt: int,
    delay_seconds: Optional[float],
) -> ToolRetryErrorRecord:
    """从失败 ToolResult 创建重试记录。"""

    return ToolRetryErrorRecord(
        attempt=attempt,
        error=dict(result.error or {}),
        delay_seconds=delay_seconds,
    )


def attach_retry_metadata(
    result: ToolResult,
    *,
    policy: ToolRetryPolicy,
    attempt_count: int,
    records: List[ToolRetryErrorRecord],
) -> ToolResult:
    """把重试过程写入 ToolResult.metadata。"""

    if attempt_count <= 1 and not records:
        return result

    metadata = dict(result.metadata)
    metadata["retry"] = {
        "enabled": policy.enabled,
        "max_attempts": policy.max_attempts,
        "attempt_count": attempt_count,
        "retried": attempt_count > 1,
        "errors": [record.to_dict(exclude_none=True) for record in records],
    }
    return result.clone(metadata=metadata)


__all__ = [
    "NON_RETRYABLE_ERROR_CODES",
    "ToolRetryErrorRecord",
    "ToolRetryPolicy",
    "attach_retry_metadata",
    "build_retry_record_from_exception",
    "build_retry_record_from_result",
    "is_retryable_exception",
    "is_retryable_result",
]
