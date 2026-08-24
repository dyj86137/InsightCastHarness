"""Agent Loop 基础抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from typing_extensions import Protocol

from core.event import RunEvent
from core.state import AgentState
from infra.exception import MaxStepsExceededError


class LoopRunContext(Protocol):
    """Agent Loop 运行时上下文协议。

    当前协议由 runtime.runner.RunnerContext 实现。
    使用 Protocol 是为了避免 loops 层直接 import runtime 层，保持依赖方向清晰。
    """

    async def record_event(
        self,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        """记录运行事件。"""


class BaseAgentLoop(ABC):
    """所有 Agent Loop 的统一基类。"""

    name: str = "base"

    @abstractmethod
    async def run(self, state: AgentState, context: LoopRunContext) -> AgentState:
        """执行 Agent Loop，并返回最终 AgentState。"""

    async def record_event(
        self,
        context: LoopRunContext,
        event: RunEvent,
        *,
        state: Optional[AgentState] = None,
    ) -> None:
        """记录事件的快捷方法。"""

        await context.record_event(event, state=state)

    def ensure_can_continue(self, state: AgentState) -> None:
        """检查 Loop 是否还能继续执行下一步。"""

        if state.max_steps is not None and state.step >= state.max_steps:
            raise MaxStepsExceededError(state.max_steps)

    async def record_step_started(
        self,
        state: AgentState,
        context: LoopRunContext,
    ) -> None:
        """记录 step 启动事件。"""

        await self.record_event(
            context,
            RunEvent.step_started(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
            ),
            state=state,
        )

    async def record_step_completed(
        self,
        state: AgentState,
        context: LoopRunContext,
    ) -> None:
        """记录 step 完成事件。"""

        await self.record_event(
            context,
            RunEvent.step_completed(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
            ),
            state=state,
        )


__all__ = [
    "BaseAgentLoop",
    "LoopRunContext",
]
