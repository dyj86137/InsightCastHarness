"""记忆系统运行时 Hook。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from core.state import AgentState
from memory.base import (
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    TaskStateSnapshot,
)
from memory.manager import MemoryManager
from runtime.hooks import BaseHook, HookContext


class MemoryLifecycleHook(BaseHook):
    """在 Runner 生命周期中自动持久化记忆和任务状态。"""

    name = "memory_lifecycle"

    def __init__(
        self,
        memory_manager: MemoryManager,
        *,
        write_short_term_on_completed: bool = True,
        write_short_term_on_failed: bool = True,
        save_task_state_on_completed: bool = True,
        save_task_state_on_failed: bool = True,
        short_term_tags: Iterable[str] = ("run_summary",),
    ) -> None:
        self.memory_manager = memory_manager
        self.write_short_term_on_completed = write_short_term_on_completed
        self.write_short_term_on_failed = write_short_term_on_failed
        self.save_task_state_on_completed = save_task_state_on_completed
        self.save_task_state_on_failed = save_task_state_on_failed
        self.short_term_tags = tuple(short_term_tags)

    async def on_run_completed(self, context: HookContext) -> None:
        """run 完成时保存 task state 和短期记忆。"""

        state = context.state
        if state is None:
            return
        if self.save_task_state_on_completed:
            await self.memory_manager.save_task_state(build_task_state_snapshot(state))
        if self.write_short_term_on_completed:
            await self._write_short_term_summary(
                state,
                tags=self.short_term_tags + ("completed",),
            )

    async def on_run_failed(self, context: HookContext) -> None:
        """run 失败时保存 task state 和失败摘要。"""

        state = context.state
        if state is None:
            return
        if self.save_task_state_on_failed:
            await self.memory_manager.save_task_state(build_task_state_snapshot(state))
        if self.write_short_term_on_failed:
            await self._write_short_term_summary(
                state,
                tags=self.short_term_tags + ("failed",),
            )

    async def _write_short_term_summary(
        self,
        state: AgentState,
        *,
        tags: Tuple[str, ...],
    ) -> None:
        summary = build_run_summary(state)
        if not summary or not state.session_id:
            return
        await self.memory_manager.add_memory(
            MemoryWriteRequest(
                kind=MemoryKind.SHORT_TERM,
                content=summary,
                summary=first_line(summary),
                scope=MemoryScope.SESSION,
                session_id=state.session_id,
                run_id=state.run_id,
                user_id=optional_metadata_string(state, "user_id"),
                tags=tags,
                importance=0.5 if state.error is None else 0.65,
                metadata={
                    "source": "memory_lifecycle_hook",
                    "status": state.status.value,
                    "step": state.step,
                },
            )
        )


def build_task_state_snapshot(state: AgentState) -> TaskStateSnapshot:
    """从 AgentState 构造可跨 run/session 保存的任务状态快照。"""

    session_id = state.session_id or state.run_id
    open_tasks = tuple(str(item) for item in state.variables.get("open_tasks", ()) if str(item).strip())
    completed_tasks = tuple(
        str(item)
        for item in state.variables.get("completed_tasks", ())
        if str(item).strip()
    )
    return TaskStateSnapshot(
        session_id=session_id,
        latest_run_id=state.run_id,
        summary=state.summary or first_line(state.final_output or ""),
        variables=dict(state.variables),
        open_tasks=open_tasks,
        completed_tasks=completed_tasks,
        metadata={
            "status": state.status.value,
            "step": state.step,
            "max_steps": state.max_steps,
            "error": state.error,
            "user_id": optional_metadata_string(state, "user_id"),
        },
    )


def build_run_summary(state: AgentState) -> str:
    """构造适合写入短期记忆的 run 摘要。"""

    lines = [
        f"Run status: {state.status.value}",
        f"Run id: {state.run_id}",
    ]
    if state.session_id:
        lines.append(f"Session id: {state.session_id}")
    if state.summary:
        lines.append(f"Conversation summary: {state.summary}")
    if state.final_output:
        lines.append(f"Final output: {state.final_output}")
    if state.error:
        lines.append(f"Error: {state.error}")
    if state.variables:
        lines.append(f"Variables: {format_variables(state.variables)}")
    return "\n".join(lines)


def format_variables(variables: Dict[str, object]) -> str:
    """格式化运行变量，避免写入过长内容。"""

    parts = []
    for key, value in sorted(variables.items()):
        text = str(value)
        if len(text) > 200:
            text = text[:197] + "..."
        parts.append(f"{key}={text}")
    return "; ".join(parts)


def first_line(text: str) -> Optional[str]:
    """返回第一行非空文本。"""

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def optional_metadata_string(state: AgentState, key: str) -> Optional[str]:
    """从 state.metadata 中读取可选字符串。"""

    value = state.metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MemoryLifecycleHook",
    "build_run_summary",
    "build_task_state_snapshot",
    "first_line",
    "format_variables",
    "optional_metadata_string",
]
