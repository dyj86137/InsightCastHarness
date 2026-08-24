"""工具执行器。"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

from core.tool import ToolCall, ToolResult
from infra.config import ToolConfig
from infra.exception import ToolExecutionError, ToolNotFoundError, ToolValidationError
from tools.base import BaseTool, ToolExecutionContext
from tools.fallback import (
    ToolFallbackAttemptRecord,
    ToolFallbackPolicy,
    attach_fallback_metadata,
    build_fallback_attempt_record,
)
from tools.registry import ToolRegistry
from tools.retry import (
    ToolRetryErrorRecord,
    ToolRetryPolicy,
    attach_retry_metadata,
    build_retry_record_from_exception,
    build_retry_record_from_result,
)

if TYPE_CHECKING:
    from tools.discovery import ToolDiscoveryManager, ToolDiscoveryReport


class ToolExecutor:
    """统一工具执行入口。"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        *,
        config: Optional[ToolConfig] = None,
        retry_policy: Optional[ToolRetryPolicy] = None,
        fallback_policy: Optional[ToolFallbackPolicy] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.config = config or ToolConfig()
        self.retry_policy = retry_policy or ToolRetryPolicy()
        self.fallback_policy = fallback_policy or ToolFallbackPolicy()

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[BaseTool],
        *,
        config: Optional[ToolConfig] = None,
        retry_policy: Optional[ToolRetryPolicy] = None,
        fallback_policy: Optional[ToolFallbackPolicy] = None,
    ) -> "ToolExecutor":
        """从工具实例集合创建执行器。"""

        return cls(
            registry=ToolRegistry(tools),
            config=config,
            retry_policy=retry_policy,
            fallback_policy=fallback_policy,
        )

    @classmethod
    def from_discovery(
        cls,
        discovery: Optional["ToolDiscoveryManager"] = None,
        *,
        config: Optional[ToolConfig] = None,
        retry_policy: Optional[ToolRetryPolicy] = None,
        fallback_policy: Optional[ToolFallbackPolicy] = None,
        override: bool = False,
        continue_on_error: bool = True,
    ) -> Tuple["ToolExecutor", "ToolDiscoveryReport"]:
        """通过工具发现机制创建执行器，并返回发现注册报告。"""

        executor = cls(
            config=config,
            retry_policy=retry_policy,
            fallback_policy=fallback_policy,
        )
        report = executor.discover_and_register(
            discovery,
            override=override,
            continue_on_error=continue_on_error,
        )
        return executor, report

    def discover_and_register(
        self,
        discovery: Optional["ToolDiscoveryManager"] = None,
        *,
        override: bool = False,
        continue_on_error: bool = True,
    ) -> "ToolDiscoveryReport":
        """发现工具并注册到当前执行器的 ToolRegistry。"""

        if discovery is None:
            from tools.discovery import create_default_discovery

            discovery = create_default_discovery()
        return discovery.register_all(
            self.registry,
            override=override,
            continue_on_error=continue_on_error,
        )

    async def execute(
        self,
        tool_call: ToolCall,
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """执行一次工具调用，并统一返回 ToolResult。"""

        started_at = time.perf_counter()
        try:
            tool = self.registry.get(tool_call.name)
            execution_context = self._build_context(context)
            self._check_permission(tool, execution_context)

            running_call = tool_call.mark_running()
            result = await self._execute_with_retry(tool, running_call, execution_context)
            result = await self._execute_with_fallback(
                tool,
                running_call,
                execution_context,
                result,
            )
            duration_ms = elapsed_ms(started_at)

            if result.duration_ms is None:
                result = result.clone(duration_ms=duration_ms)
            return result

        except ToolNotFoundError as exc:
            return ToolResult.failure(
                tool_call=tool_call,
                error=exc,
                duration_ms=elapsed_ms(started_at),
            )
        except ToolValidationError as exc:
            return ToolResult.failure(
                tool_call=tool_call,
                error=exc,
                duration_ms=elapsed_ms(started_at),
            )
        except Exception as exc:
            return ToolResult.failure(
                tool_call=tool_call,
                error=normalize_tool_exception(exc, tool_call),
                duration_ms=elapsed_ms(started_at),
            )

    def register(self, tool: BaseTool, *, override: bool = False) -> None:
        """注册工具。"""

        self.registry.register(tool, override=override)

    def _build_context(
        self,
        context: Optional[ToolExecutionContext],
    ) -> ToolExecutionContext:
        """合并默认配置和调用级上下文。"""

        if context is None:
            return ToolExecutionContext(
                cwd=self.config.cwd,
                allow_mutation=self.config.allow_mutating_tools,
                timeout_seconds=self.config.tool_timeout_seconds,
            )

        timeout_seconds = (
            context.timeout_seconds
            if context.timeout_seconds is not None
            else self.config.tool_timeout_seconds
        )
        return context.clone(
            cwd=context.cwd or self.config.cwd,
            allow_mutation=context.allow_mutation or self.config.allow_mutating_tools,
            timeout_seconds=timeout_seconds,
        )

    def _check_permission(self, tool: BaseTool, context: ToolExecutionContext) -> None:
        """检查工具是否允许在当前上下文执行。"""

        if not tool.is_read_only and not context.allow_mutation:
            raise ToolExecutionError(
                "Mutating tool is not allowed in current context.",
                details={"tool_name": tool.name},
            )

    async def _execute_with_timeout(
        self,
        tool: BaseTool,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """带超时地执行工具。"""

        coroutine = tool.run(tool_call, context)
        if context.timeout_seconds is None:
            return await coroutine

        try:
            return await asyncio.wait_for(coroutine, timeout=context.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise ToolExecutionError(
                "Tool execution timed out.",
                details={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "timeout_seconds": context.timeout_seconds,
                },
                cause=exc,
            ) from exc

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        tool_call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """按 ToolRetryPolicy 执行工具。"""

        attempt = 1
        retry_records: List[ToolRetryErrorRecord] = []

        while True:
            try:
                result = await self._execute_with_timeout(tool, tool_call, context)
            except Exception as exc:
                should_retry = self.retry_policy.should_retry_exception(
                    attempt=attempt,
                    is_read_only=tool.is_read_only,
                    error=exc,
                )
                if not should_retry:
                    failure = ToolResult.failure(
                        tool_call=tool_call,
                        error=normalize_tool_exception(exc, tool_call),
                    )
                    final_records = list(retry_records)
                    if retry_records or attempt > 1:
                        final_records.append(
                            build_retry_record_from_exception(
                                exc,
                                attempt=attempt,
                                delay_seconds=None,
                            )
                        )
                    return attach_retry_metadata(
                        failure,
                        policy=self.retry_policy,
                        attempt_count=attempt,
                        records=final_records,
                    )

                delay_seconds = self.retry_policy.next_delay_seconds(attempt)
                retry_records.append(
                    build_retry_record_from_exception(
                        exc,
                        attempt=attempt,
                        delay_seconds=delay_seconds,
                    )
                )
                await sleep_if_needed(delay_seconds)
                attempt += 1
                continue

            if not self.retry_policy.should_retry_result(
                attempt=attempt,
                is_read_only=tool.is_read_only,
                result=result,
            ):
                final_records = list(retry_records)
                if result.is_error and retry_records:
                    final_records.append(
                        build_retry_record_from_result(
                            result,
                            attempt=attempt,
                            delay_seconds=None,
                        )
                    )
                return attach_retry_metadata(
                    result,
                    policy=self.retry_policy,
                    attempt_count=attempt,
                    records=final_records,
                )

            delay_seconds = self.retry_policy.next_delay_seconds(attempt)
            retry_records.append(
                build_retry_record_from_result(
                    result,
                    attempt=attempt,
                    delay_seconds=delay_seconds,
                )
            )
            await sleep_if_needed(delay_seconds)
            attempt += 1

    async def _execute_with_fallback(
        self,
        source_tool: BaseTool,
        source_call: ToolCall,
        context: ToolExecutionContext,
        source_result: ToolResult,
    ) -> ToolResult:
        """主工具失败后按 ToolFallbackPolicy 尝试备用工具。"""

        if not self.fallback_policy.should_fallback(
            source_tool_name=source_tool.name,
            result=source_result,
            source_is_read_only=source_tool.is_read_only,
        ):
            return source_result

        fallback_names = self.fallback_policy.get_fallback_names(source_tool.name)
        records: List[ToolFallbackAttemptRecord] = []
        for fallback_name in fallback_names:
            fallback_call = build_fallback_tool_call(
                source_call,
                fallback_tool_name=fallback_name,
            )
            try:
                fallback_tool = self.registry.get(fallback_name)
                self._check_fallback_tool_allowed(fallback_tool)
                self._check_permission(fallback_tool, context)
                fallback_result = await self._execute_with_retry(
                    fallback_tool,
                    fallback_call,
                    context,
                )
            except Exception as exc:
                fallback_result = ToolResult.failure(
                    tool_call=fallback_call,
                    error=normalize_tool_exception(exc, fallback_call),
                )

            records.append(
                build_fallback_attempt_record(
                    source_tool_name=source_tool.name,
                    fallback_tool_name=fallback_name,
                    result=fallback_result,
                )
            )
            if fallback_result.is_error:
                continue

            normalized_result = fallback_result.clone(
                tool_call_id=source_call.id,
                tool_name=source_call.name,
            )
            return attach_fallback_metadata(
                normalized_result,
                source_result=source_result,
                source_tool_name=source_tool.name,
                attempted_names=fallback_names,
                records=records,
            )

        return attach_fallback_metadata(
            source_result,
            source_result=source_result,
            source_tool_name=source_tool.name,
            attempted_names=fallback_names,
            records=records,
        )

    def _check_fallback_tool_allowed(self, tool: BaseTool) -> None:
        """检查 fallback 策略是否允许使用该备用工具。"""

        if self.fallback_policy.fallback_read_only_only and not tool.is_read_only:
            raise ToolExecutionError(
                "Mutating fallback tool is not allowed by fallback policy.",
                details={"tool_name": tool.name},
            )


def elapsed_ms(started_at: float) -> float:
    """计算从 started_at 到当前的毫秒耗时。"""

    return (time.perf_counter() - started_at) * 1000


def normalize_tool_exception(error: BaseException, tool_call: ToolCall) -> BaseException:
    """把未知异常转换成统一的 ToolExecutionError。"""

    if isinstance(error, (ToolExecutionError, ToolNotFoundError, ToolValidationError)):
        return error
    return ToolExecutionError(
        "Tool execution failed.",
        details={"tool_call_id": tool_call.id, "tool_name": tool_call.name},
        cause=error,
    )


def build_fallback_tool_call(
    source_call: ToolCall,
    *,
    fallback_tool_name: str,
) -> ToolCall:
    """基于原始工具调用创建备用工具调用。"""

    metadata = dict(source_call.metadata)
    metadata["fallback_from_tool_name"] = source_call.name
    metadata["fallback_from_tool_call_id"] = source_call.id
    return ToolCall.create(
        name=fallback_tool_name,
        arguments=dict(source_call.arguments),
        message_id=source_call.message_id,
        step=source_call.step,
        metadata=metadata,
    )


async def sleep_if_needed(delay_seconds: float) -> None:
    """按需等待重试退避时间。"""

    if delay_seconds <= 0:
        return
    await asyncio.sleep(delay_seconds)


__all__ = [
    "ToolExecutor",
    "build_fallback_tool_call",
    "elapsed_ms",
    "normalize_tool_exception",
    "sleep_if_needed",
]
