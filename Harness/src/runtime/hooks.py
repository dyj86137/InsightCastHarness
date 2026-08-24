"""运行时生命周期 Hook 系统。"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from pydantic import Field

from core.event import RunEvent
from core.state import AgentState
from infra.exception import HarnessError, RunnerError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


class HookEvent(str, Enum):
    """Runner 和 Loop 可触发的生命周期事件。"""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"

    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    LLM_CALL_FAILED = "llm_call_failed"

    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    TOOL_CALL_FAILED = "tool_call_failed"

    CONTEXT_BUILT = "context_built"

    BEFORE_CHECKPOINT_SAVE = "before_checkpoint_save"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_FAILED = "checkpoint_failed"

    BEFORE_REPLAY = "before_replay"
    AFTER_REPLAY = "after_replay"
    REPLAY_FAILED = "replay_failed"

    BEFORE_RESUME = "before_resume"
    AFTER_RESUME = "after_resume"
    RESUME_FAILED = "resume_failed"


class HookContext(SerializableModel):
    """传递给 Hook 的运行时上下文。"""

    run_id: str
    session_id: Optional[str] = None
    step: Optional[int] = None
    state: Optional[AgentState] = None
    event: Optional[RunEvent] = None
    checkpoint_id: Optional[str] = None
    replay_plan_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Hook 上下文。"""

        if not self.run_id:
            raise HarnessValidationError("HookContext requires run_id.")
        if self.step is not None and self.step < 0:
            raise HarnessValidationError(
                "HookContext step cannot be negative.",
                details={"run_id": self.run_id, "step": self.step},
            )


class BaseHook(ABC):
    """Hook 基类。

    子类只需要覆盖自己关心的方法；默认实现为空操作。
    """

    name: str = "base_hook"

    async def on_event(self, event: HookEvent, context: HookContext) -> None:
        """通用事件入口。"""

        del event, context

    async def on_run_started(self, context: HookContext) -> None:
        """run 启动时触发。"""

        del context

    async def on_run_completed(self, context: HookContext) -> None:
        """run 完成时触发。"""

        del context

    async def on_run_failed(self, context: HookContext) -> None:
        """run 失败时触发。"""

        del context

    async def on_step_started(self, context: HookContext) -> None:
        """step 启动时触发。"""

        del context

    async def on_step_completed(self, context: HookContext) -> None:
        """step 完成时触发。"""

        del context

    async def on_step_failed(self, context: HookContext) -> None:
        """step 失败时触发。"""

        del context

    async def on_before_llm_call(self, context: HookContext) -> None:
        """LLM 调用前触发。"""

        del context

    async def on_after_llm_call(self, context: HookContext) -> None:
        """LLM 调用后触发。"""

        del context

    async def on_llm_call_failed(self, context: HookContext) -> None:
        """LLM 调用失败时触发。"""

        del context

    async def on_before_tool_call(self, context: HookContext) -> None:
        """工具调用前触发。"""

        del context

    async def on_after_tool_call(self, context: HookContext) -> None:
        """工具调用后触发。"""

        del context

    async def on_tool_call_failed(self, context: HookContext) -> None:
        """工具调用失败时触发。"""

        del context

    async def on_context_built(self, context: HookContext) -> None:
        """Context built event hook."""

        del context

    async def on_checkpoint_created(self, context: HookContext) -> None:
        """Checkpoint 创建后触发。"""

        del context

    async def on_before_checkpoint_save(self, context: HookContext) -> None:
        """Checkpoint 保存前触发。"""

        del context

    async def on_checkpoint_failed(self, context: HookContext) -> None:
        """Checkpoint 保存失败时触发。"""

        del context

    async def on_before_replay(self, context: HookContext) -> None:
        """Replay 执行前触发。"""

        del context

    async def on_after_replay(self, context: HookContext) -> None:
        """Replay 执行后触发。"""

        del context

    async def on_replay_failed(self, context: HookContext) -> None:
        """Replay 执行失败时触发。"""

        del context

    async def on_before_resume(self, context: HookContext) -> None:
        """从 Checkpoint 恢复前触发。"""

        del context

    async def on_after_resume(self, context: HookContext) -> None:
        """从 Checkpoint 恢复后触发。"""

        del context

    async def on_resume_failed(self, context: HookContext) -> None:
        """从 Checkpoint 恢复失败时触发。"""

        del context


HookCallback = Callable[[HookEvent, HookContext], Awaitable[None]]


class HookManager:
    """管理并触发生命周期 Hook。"""

    def __init__(
        self,
        hooks: Optional[Iterable[BaseHook]] = None,
        *,
        raise_on_error: bool = False,
    ) -> None:
        self._hooks: List[BaseHook] = list(hooks or ())
        self._callbacks: Dict[HookEvent, List[HookCallback]] = {}
        self.raise_on_error = raise_on_error
        self.errors: List[RunnerError] = []

    def register(self, hook: BaseHook) -> None:
        """注册 Hook。"""

        if not isinstance(hook, BaseHook):
            raise HarnessValidationError(
                "HookManager can only register BaseHook instances.",
                details={"actual_type": hook.__class__.__name__},
            )
        self._hooks.append(hook)

    def list_hooks(self) -> List[BaseHook]:
        """返回所有 Hook。"""

        return list(self._hooks)

    def register_callback(self, event: HookEvent, callback: HookCallback) -> None:
        """注册某个生命周期事件的轻量回调。"""

        if not callable(callback):
            raise HarnessValidationError(
                "Hook callback must be callable.",
                details={"event": event.value},
            )
        self._callbacks.setdefault(event, []).append(callback)

    def list_callbacks(self, event: Optional[HookEvent] = None) -> List[HookCallback]:
        """返回已注册的回调。"""

        if event is not None:
            return list(self._callbacks.get(event, ()))

        callbacks: List[HookCallback] = []
        for items in self._callbacks.values():
            callbacks.extend(items)
        return callbacks

    async def emit(self, event: HookEvent, context: HookContext) -> None:
        """顺序触发所有 Hook。"""

        for hook in self._hooks:
            await self._emit_one(hook, event, context)
        for callback in self._callbacks.get(event, ()):
            await self._emit_callback(callback, event, context)

    async def _emit_one(
        self,
        hook: BaseHook,
        event: HookEvent,
        context: HookContext,
    ) -> None:
        """触发单个 Hook，并按策略处理 Hook 自身异常。"""

        try:
            await hook.on_event(event, context)
            method = getattr(hook, hook_method_name(event), None)
            if method is not None:
                await method(context)
        except Exception as exc:
            error = normalize_hook_error(hook, event, exc)
            self.errors.append(error)
            if self.raise_on_error:
                raise error from exc

    async def _emit_callback(
        self,
        callback: HookCallback,
        event: HookEvent,
        context: HookContext,
    ) -> None:
        """触发单个回调，并按策略处理回调自身异常。"""

        try:
            await callback(event, context)
        except Exception as exc:
            error = normalize_callback_error(callback, event, exc)
            self.errors.append(error)
            if self.raise_on_error:
                raise error from exc


def hook_method_name(event: HookEvent) -> str:
    """把 HookEvent 映射到 BaseHook 方法名。"""

    return f"on_{event.value}"


def normalize_hook_error(
    hook: BaseHook,
    event: HookEvent,
    error: BaseException,
) -> RunnerError:
    """把 Hook 抛出的异常转换为 RunnerError。"""

    if isinstance(error, RunnerError):
        return error

    details = {
        "hook": hook.name,
        "hook_type": hook.__class__.__name__,
        "event": event.value,
    }
    if isinstance(error, HarnessError):
        details["cause"] = error.to_dict()

    return RunnerError(
        "Hook execution failed.",
        details=details,
        cause=error,
    )


def normalize_callback_error(
    callback: HookCallback,
    event: HookEvent,
    error: BaseException,
) -> RunnerError:
    """把 Hook 回调抛出的异常转换为 RunnerError。"""

    if isinstance(error, RunnerError):
        return error

    details = {
        "callback": getattr(callback, "__name__", repr(callback)),
        "event": event.value,
    }
    if isinstance(error, HarnessError):
        details["cause"] = error.to_dict()

    return RunnerError(
        "Hook callback execution failed.",
        details=details,
        cause=error,
    )


__all__ = [
    "BaseHook",
    "HookCallback",
    "HookContext",
    "HookEvent",
    "HookManager",
    "hook_method_name",
    "normalize_callback_error",
    "normalize_hook_error",
]
