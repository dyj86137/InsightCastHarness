"""ReAct Agent Loop。"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from context.builder import ContextBuilder
from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.message import Message
from core.state import AgentState
from core.tool import ToolCall, ToolResult
from infra.config import LLMConfig
from infra.exception import RunnerError, ToolExecutionError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest, ChatResponse
from loops.base import BaseAgentLoop, LoopRunContext
from tools.base import ToolExecutionContext
from tools.executor import ToolExecutor


DEFAULT_REACT_SYSTEM_PROMPT = """You are a ReAct agent.

Use this exact format when you need a tool:
Thought: brief reasoning
Action: tool_name
Action Input: {"arg": "value"}

Use this exact format when you can answer:
Final Answer: your answer
"""


class ReActDecisionType(str, Enum):
    """ReAct 单步决策类型。"""

    FINAL = "final"
    TOOL_CALL = "tool_call"


class ReActDecision(SerializableModel):
    """从 assistant 消息中解析出来的 ReAct 决策。"""

    decision_type: ReActDecisionType
    final_answer: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 ReAct 决策的领域规则。"""

        if self.decision_type == ReActDecisionType.FINAL and self.final_answer is None:
            raise HarnessValidationError("Final ReAct decision requires final_answer.")
        if self.decision_type == ReActDecisionType.TOOL_CALL and not self.tool_name:
            raise HarnessValidationError("Tool-call ReAct decision requires tool_name.")


class ReactLoop(BaseAgentLoop):
    """V1 最小可用 ReAct 循环。"""

    name = "react"

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tool_executor: ToolExecutor,
        context_builder: Optional[ContextBuilder] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        self.llm = llm
        self.tool_executor = tool_executor
        self.context_builder = context_builder or ContextBuilder(
            system_message=Message.system(DEFAULT_REACT_SYSTEM_PROMPT)
        )
        self.llm_config = llm_config or LLMConfig(
            provider=llm.provider_name,
            model=llm.model_name,
        )

    async def run(self, state: AgentState, context: LoopRunContext) -> AgentState:
        """执行 ReAct 循环，直到得到 Final Answer 或达到最大步数。"""

        current_state = state

        while not current_state.is_terminal:
            self.ensure_can_continue(current_state)
            await self.record_step_started(current_state, context)

            response = await self._call_llm(current_state, context)
            current_state = current_state.add_message(response.message)

            decision = parse_react_decision(response.text)
            if decision.decision_type == ReActDecisionType.FINAL:
                final_state = current_state.mark_completed(decision.final_answer or "")
                await self.record_step_completed(final_state, context)
                return final_state

            current_state = await self._execute_tool_decision(
                current_state,
                decision,
                context,
                response,
            )
            await self.record_step_completed(current_state, context)
            current_state = current_state.next_step()

        return current_state

    async def _call_llm(
        self,
        state: AgentState,
        context: LoopRunContext,
    ) -> ChatResponse:
        """构建上下文并调用 LLM。"""

        request = await self._build_request(state)
        await self.record_event(
            context,
            RunEvent.llm_requested(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                provider=self.llm.provider_name,
                model=request.model or self.llm.model_name,
                payload={
                    "request_id": request.id,
                    "message_count": len(request.messages),
                    "metadata": request.metadata,
                },
            ),
            state=state,
        )

        try:
            response = await self.llm.chat(request)
        except Exception as exc:
            await self.record_event(
                context,
                RunEvent.create(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    event_type=RunEventType.LLM_FAILED,
                    step=state.step,
                    level=RunEventLevel.ERROR,
                    message="LLM call failed.",
                    payload={"request_id": request.id},
                    error=error_to_payload(exc),
                ),
                state=state,
            )
            raise

        await self.record_event(
            context,
            RunEvent.llm_responded(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                provider=response.provider,
                model=response.model,
                payload={
                    "request_id": request.id,
                    "response_id": response.id,
                    "finish_reason": response.finish_reason.value,
                    "usage": response.usage.to_dict(),
                },
            ),
            state=state,
        )
        return response

    async def _build_request(self, state: AgentState) -> ChatRequest:
        """根据当前状态构建 ChatRequest。"""

        return await self.context_builder.build_request_async(
            state,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            timeout_seconds=self.llm_config.timeout_seconds,
        )

    async def _execute_tool_decision(
        self,
        state: AgentState,
        decision: ReActDecision,
        context: LoopRunContext,
        response: ChatResponse,
    ) -> AgentState:
        """执行工具调用决策，并把 observation 回灌到状态。"""

        tool_call = ToolCall.create(
            name=decision.tool_name or "",
            arguments=decision.tool_arguments,
            message_id=response.message.id,
            step=state.step,
        )

        await self.record_event(
            context,
            RunEvent.tool_call_started(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            ),
            state=state,
        )

        result = await self.tool_executor.execute(
            tool_call,
            ToolExecutionContext(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
            ),
        )

        if result.is_error:
            await self.record_event(
                context,
                RunEvent.tool_call_failed(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    step=state.step,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    error=tool_result_error(result),
                ),
                state=state,
            )
        else:
            await self.record_event(
                context,
                RunEvent.tool_call_completed(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    step=state.step,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    payload={
                        "tool_result_id": result.id,
                        "duration_ms": result.duration_ms,
                    },
                ),
                state=state,
            )

        return state.add_message(tool_result_to_message(result))


def parse_react_decision(text: str) -> ReActDecision:
    """从 assistant 文本中解析 ReAct 决策。"""

    final_match = re.search(r"Final Answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        return ReActDecision(
            decision_type=ReActDecisionType.FINAL,
            final_answer=final_match.group(1).strip(),
        )

    action_match = re.search(r"^Action\s*:\s*([a-zA-Z0-9_.-]+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not action_match:
        raise RunnerError(
            "ReAct response does not contain Final Answer or Action.",
            details={"response_text": text},
        )

    action_input = extract_action_input(text)
    return ReActDecision(
        decision_type=ReActDecisionType.TOOL_CALL,
        tool_name=action_match.group(1).strip(),
        tool_arguments=parse_action_input(action_input),
    )


def extract_action_input(text: str) -> str:
    """提取 Action Input 后面的内容。"""

    match = re.search(r"Action Input\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "{}"
    return match.group(1).strip()


def parse_action_input(raw: str) -> Dict[str, Any]:
    """把 Action Input 解析为工具参数字典。"""

    cleaned = strip_code_fence(raw.strip())
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"input": cleaned}

    if isinstance(parsed, dict):
        return parsed
    return {"input": parsed}


def strip_code_fence(text: str) -> str:
    """移除包裹 JSON 的 Markdown code fence。"""

    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def tool_result_to_message(result: ToolResult) -> Message:
    """把工具结果转换成回灌给 LLM 的 tool 消息。"""

    if result.is_error:
        content = format_tool_error(result)
    else:
        content = result.output

    return Message.tool(
        content=content,
        tool_call_id=result.tool_call_id,
        name=result.tool_name,
    )


def format_tool_error(result: ToolResult) -> str:
    """格式化工具错误 observation。"""

    if not result.error:
        return result.output or "Tool execution failed."

    error_type = result.error.get("type", "ToolError")
    message = result.error.get("message", "Tool execution failed.")
    return f"Tool error: {error_type}: {message}"


def tool_result_error(result: ToolResult) -> ToolExecutionError:
    """把失败 ToolResult 包装成可写入 RunEvent 的异常。"""

    return ToolExecutionError(
        "Tool call failed.",
        details={
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "tool_result_id": result.id,
            "tool_error": result.error,
        },
    )


__all__ = [
    "DEFAULT_REACT_SYSTEM_PROMPT",
    "ReActDecision",
    "ReActDecisionType",
    "ReactLoop",
    "extract_action_input",
    "format_tool_error",
    "parse_action_input",
    "parse_react_decision",
    "strip_code_fence",
    "tool_result_to_message",
]
