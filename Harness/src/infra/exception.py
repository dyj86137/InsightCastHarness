"""myHarness V1 的统一异常体系。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


# ErrorCode 类的价值是降低人为出错概率
# 如果不用枚举，错误码就是散落在代码各处的字符串；字符串拼写错误不会有语法报错，直到逻辑不生效了才需要逐行排查，属于非常隐蔽的低级 bug。
class ErrorCode(str, Enum):
    """框架内稳定使用的错误码。"""

    CONFIG_ERROR = "config_error"
    SERIALIZATION_ERROR = "serialization_error"
    VALIDATION_ERROR = "validation_error"

    LLM_ERROR = "llm_error"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_PROVIDER_ERROR = "llm_provider_error"

    TOOL_ERROR = "tool_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_VALIDATION_ERROR = "tool_validation_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"

    RUNNER_ERROR = "runner_error"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TRACE_ERROR = "trace_error"
    CONTEXT_ERROR = "context_error"


class HarnessError(Exception):
    """所有框架异常的统一基类。

    统一异常基类用于保证错误都能带上错误码、可记录的上下文和是否可重试等信息。
    其中 to_dict 函数用于写入日志、Trace、RunEvent。
    """

    code: ErrorCode = ErrorCode.VALIDATION_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.details = dict(details or {})
        self.retryable = self.retryable if retryable is None else retryable
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """转换为适合写入日志或 RunEvent 的普通字典。"""

        data: dict[str, Any] = {
            "type": self.__class__.__name__,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
        if self.cause is not None:
            data["cause"] = {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause),
            }
        return data

    def __str__(self) -> str:
        return self.message


class ConfigError(HarnessError):
    """配置加载或配置值非法。"""

    code = ErrorCode.CONFIG_ERROR


class SerializationError(HarnessError):
    """标准对象序列化或反序列化失败。"""

    code = ErrorCode.SERIALIZATION_ERROR


class ValidationError(HarnessError):
    """框架内部输入校验失败。"""

    code = ErrorCode.VALIDATION_ERROR


class LLMError(HarnessError):
    """LLM 调用相关错误。"""

    code = ErrorCode.LLM_ERROR
    retryable = True


class LLMTimeoutError(LLMError):
    """LLM 调用超时。"""

    code = ErrorCode.LLM_TIMEOUT
    retryable = True


class LLMRateLimitError(LLMError):
    """LLM Provider 触发限流。"""

    code = ErrorCode.LLM_RATE_LIMIT
    retryable = True


class LLMProviderError(LLMError):
    """LLM Provider 返回异常或不可解析响应。"""

    code = ErrorCode.LLM_PROVIDER_ERROR
    retryable = True


class ToolError(HarnessError):
    """工具系统相关错误。"""

    code = ErrorCode.TOOL_ERROR


class ToolNotFoundError(ToolError):
    """请求的工具没有注册。"""

    code = ErrorCode.TOOL_NOT_FOUND

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"Tool not found: {tool_name}",
            details={"tool_name": tool_name},
        )


class ToolValidationError(ToolError):
    """工具参数校验失败。"""

    code = ErrorCode.TOOL_VALIDATION_ERROR


class ToolExecutionError(ToolError):
    """工具执行阶段失败。"""

    code = ErrorCode.TOOL_EXECUTION_ERROR


class RunnerError(HarnessError):
    """Runner 运行时错误。"""

    code = ErrorCode.RUNNER_ERROR


class MaxStepsExceededError(RunnerError):
    """Agent Loop 超过最大执行步数。"""

    code = ErrorCode.MAX_STEPS_EXCEEDED

    def __init__(self, max_steps: int) -> None:
        super().__init__(
            f"Max steps exceeded: {max_steps}",
            details={"max_steps": max_steps},
        )


class TraceError(HarnessError):
    """Trace 读写失败。"""

    code = ErrorCode.TRACE_ERROR
    retryable = True


class ContextError(HarnessError):
    """上下文构建或裁剪失败。"""

    code = ErrorCode.CONTEXT_ERROR


__all__ = [
    "ConfigError",
    "ContextError",
    "ErrorCode",
    "HarnessError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "MaxStepsExceededError",
    "RunnerError",
    "SerializationError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "TraceError",
    "ValidationError",
]
