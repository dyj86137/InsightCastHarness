"""Plan-and-Solve Agent Loop.

该循环先生成结构化计划，再逐步执行。每个计划步骤可以直接给出结果，
也可以在有 ToolExecutor 时发起工具调用，工具 observation 会回灌到状态后继续求解。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from context.builder import ContextBuilder
from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.message import Message
from core.state import AgentState
from core.tool import ToolCall, ToolResult
from infra.config import LLMConfig
from infra.exception import RunnerError, ToolExecutionError, ValidationError as HarnessValidationError
from infra.json import (
    compact_json as compact_json_value,
    parse_json_value as parse_json_payload,
    strip_json_code_fence,
)
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest, ChatResponse
from loops.base import BaseAgentLoop, LoopRunContext
from tools.base import ToolExecutionContext
from tools.executor import ToolExecutor


DEFAULT_PLAN_SOLVE_SYSTEM_PROMPT = """You are a Plan-and-Solve agent.

Break the task into a concise plan, execute each step carefully, and synthesize a final answer.
When tools are available, use them for facts, calculations, file/workspace operations, or other external actions instead of guessing.
"""

PLAN_PROMPT = """Create a concise executable plan for the user's task.

Return exactly one JSON object:
{
  "steps": [
    {
      "instruction": "what to do",
      "requires_tool": false,
      "tool_name": null,
      "tool_arguments": {}
    }
  ]
}

Keep the plan short. Do not solve the task yet.
"""

FINAL_SYNTHESIS_PROMPT = """Use the completed plan steps below to produce the final answer.

Return only the final answer.
"""


class PlanStepStatus(str):
    """计划步骤状态常量。"""

    PENDING = "pending"
    COMPLETED = "completed"


class PlanStep(SerializableModel):
    """Plan-and-Solve 中的一条计划步骤。"""

    index: int
    instruction: str
    status: str = PlanStepStatus.PENDING
    output: Optional[str] = None
    requires_tool: bool = False
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验计划步骤。"""

        if self.index <= 0:
            raise HarnessValidationError(
                "PlanStep index must be positive.",
                details={"index": self.index},
            )
        if not self.instruction or not self.instruction.strip():
            raise HarnessValidationError("PlanStep requires non-empty instruction.")

    def mark_completed(
        self,
        output: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "PlanStep":
        """标记步骤完成。"""

        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata or {})
        return self.clone(
            status=PlanStepStatus.COMPLETED,
            output=output,
            metadata=merged_metadata,
        )


class PlanAndSolveTrace(SerializableModel):
    """Plan-and-Solve 执行摘要。"""

    plan: Tuple[PlanStep, ...] = Field(default_factory=tuple)
    final_answer: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class PlanStepDecisionType(str, Enum):
    """单个计划步骤的执行决策。"""

    FINAL = "final"
    TOOL_CALL = "tool_call"


class PlanStepDecision(SerializableModel):
    """模型在执行计划步骤时返回的结构化决策。"""

    decision_type: PlanStepDecisionType
    output: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    thought: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验步骤执行决策。"""

        if self.decision_type == PlanStepDecisionType.FINAL and self.output is None:
            raise HarnessValidationError("Final plan-step decision requires output.")
        if self.decision_type == PlanStepDecisionType.TOOL_CALL and not self.tool_name:
            raise HarnessValidationError("Tool-call plan-step decision requires tool_name.")


class PlanAndSolveLoop(BaseAgentLoop):
    """具备结构化计划、步骤执行、工具调用和最终综合的 Agent Loop。"""

    name = "plan_and_solve"

    def __init__(
        self,
        *,
        llm: BaseLLM,
        context_builder: Optional[ContextBuilder] = None,
        llm_config: Optional[LLMConfig] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_plan_steps: int = 8,
        max_step_iterations: int = 4,
    ) -> None:
        if max_plan_steps <= 0:
            raise HarnessValidationError(
                "PlanAndSolveLoop max_plan_steps must be positive.",
                details={"max_plan_steps": max_plan_steps},
            )
        if max_step_iterations <= 0:
            raise HarnessValidationError(
                "PlanAndSolveLoop max_step_iterations must be positive.",
                details={"max_step_iterations": max_step_iterations},
            )
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(
            system_message=Message.system(DEFAULT_PLAN_SOLVE_SYSTEM_PROMPT)
        )
        self.llm_config = llm_config or LLMConfig(
            provider=llm.provider_name,
            model=llm.model_name,
        )
        self.tool_executor = tool_executor
        self.max_plan_steps = max_plan_steps
        self.max_step_iterations = max_step_iterations

    async def run(self, state: AgentState, context: LoopRunContext) -> AgentState:
        """执行 Plan-and-Solve 循环。"""

        current_state = state

        self.ensure_can_continue(current_state)
        await self.record_step_started(current_state, context)
        plan_response = await self._call_llm(
            current_state,
            context,
            purpose="plan",
            prompt=PLAN_PROMPT,
        )
        plan = parse_plan_response(plan_response.text, max_steps=self.max_plan_steps)
        if not plan:
            raise RunnerError(
                "Plan-and-Solve failed to parse a non-empty plan.",
                details={"response_text": plan_response.text},
            )

        current_state = current_state.add_message(plan_response.message)
        await self._record_loop_event(
            current_state,
            context,
            message="Plan created.",
            payload={"plan": [step.to_dict(exclude_none=True) for step in plan]},
        )
        await self.record_step_completed(current_state, context)
        current_state = current_state.next_step()

        completed_steps: List[PlanStep] = []
        for step in plan:
            self.ensure_can_continue(current_state)
            await self.record_step_started(current_state, context)
            current_state, completed_step = await self._execute_plan_step(
                current_state,
                step,
                completed_steps,
                context,
            )
            completed_steps.append(completed_step)
            await self._record_loop_event(
                current_state,
                context,
                message="Plan step completed.",
                payload={"step": completed_step.to_dict(exclude_none=True)},
            )
            await self.record_step_completed(current_state, context)
            current_state = current_state.next_step()

        self.ensure_can_continue(current_state)
        await self.record_step_started(current_state, context)
        final_response = await self._call_llm(
            current_state,
            context,
            purpose="final_synthesis",
            prompt=format_final_synthesis_prompt(completed_steps),
        )
        final_answer = final_response.text.strip()
        current_state = current_state.add_message(final_response.message)
        current_state = current_state.with_metadata(
            "plan_and_solve",
            PlanAndSolveTrace(
                plan=tuple(completed_steps),
                final_answer=final_answer,
                metadata={
                    "max_plan_steps": self.max_plan_steps,
                    "max_step_iterations": self.max_step_iterations,
                    "tools_enabled": self.tool_executor is not None,
                    "completed_step_count": len(completed_steps),
                    "total_tool_calls": count_tool_calls(completed_steps),
                },
            ).to_dict(exclude_none=True),
        )
        final_state = current_state.mark_completed(final_answer)
        await self.record_step_completed(final_state, context)
        return final_state

    async def _execute_plan_step(
        self,
        state: AgentState,
        step: PlanStep,
        completed_steps: List[PlanStep],
        context: LoopRunContext,
    ) -> Tuple[AgentState, PlanStep]:
        """执行单个计划步骤，必要时循环调用工具直到得到步骤结果。"""

        current_state = state
        tool_records: List[Dict[str, object]] = []
        for iteration in range(1, self.max_step_iterations + 1):
            prompt = format_solve_step_prompt(
                step,
                completed_steps,
                available_tools=format_available_tools(self.tool_executor),
                iteration=iteration,
                max_iterations=self.max_step_iterations,
            )
            response = await self._call_llm(
                current_state,
                context,
                purpose="solve_step",
                prompt=prompt,
                metadata={
                    "plan_step_index": step.index,
                    "iteration": iteration,
                },
            )
            current_state = current_state.add_message(response.message)
            decision = parse_plan_step_decision(response.text)

            if decision.decision_type == PlanStepDecisionType.FINAL:
                completed_step = step.mark_completed(
                    decision.output or "",
                    metadata={
                        "iterations": iteration,
                        "tool_calls": tool_records,
                    },
                )
                return current_state, completed_step

            current_state, tool_result = await self._execute_tool_decision(
                current_state,
                step,
                decision,
                context,
                response,
            )
            tool_records.append(tool_result_to_record(tool_result, iteration=iteration))

        raise RunnerError(
            "Plan step did not produce a final output within iteration limit.",
            details={
                "step_index": step.index,
                "max_step_iterations": self.max_step_iterations,
            },
        )

    async def _call_llm(
        self,
        state: AgentState,
        context: LoopRunContext,
        *,
        purpose: str,
        prompt: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ChatResponse:
        """构建带临时用户提示的请求并调用 LLM。"""

        prompt_metadata = {"loop": self.name, "purpose": purpose}
        prompt_metadata.update(metadata or {})
        request_state = state.add_message(Message.user(prompt, metadata=prompt_metadata))
        request = await self._build_request(request_state, purpose=purpose, metadata=metadata)
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
                    "purpose": purpose,
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
                    message="Plan-and-Solve LLM call failed.",
                    payload={"request_id": request.id, "purpose": purpose},
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
                    "purpose": purpose,
                    "finish_reason": response.finish_reason.value,
                    "usage": response.usage.to_dict(),
                },
            ),
            state=state,
        )
        return response

    async def _build_request(
        self,
        state: AgentState,
        *,
        purpose: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ChatRequest:
        """构建 LLM 请求。"""

        request = await self.context_builder.build_request_async(
            state,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            timeout_seconds=self.llm_config.timeout_seconds,
        )
        request_metadata = dict(request.metadata)
        request_metadata.update({"loop": self.name, "purpose": purpose})
        request_metadata.update(metadata or {})
        return request.clone(metadata=request_metadata)

    async def _execute_tool_decision(
        self,
        state: AgentState,
        step: PlanStep,
        decision: PlanStepDecision,
        context: LoopRunContext,
        response: ChatResponse,
    ) -> Tuple[AgentState, ToolResult]:
        """执行步骤中的工具调用，并把 observation 回灌到状态。"""

        if self.tool_executor is None:
            raise RunnerError(
                "Plan step requested a tool but no ToolExecutor is configured.",
                details={
                    "step_index": step.index,
                    "tool_name": decision.tool_name,
                },
            )

        tool_call = ToolCall.create(
            name=decision.tool_name or "",
            arguments=decision.tool_arguments,
            message_id=response.message.id,
            step=state.step,
            metadata={
                "loop": self.name,
                "plan_step_index": step.index,
                "thought": decision.thought,
            },
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

        return state.add_message(tool_result_to_message(result)), result

    async def _record_loop_event(
        self,
        state: AgentState,
        context: LoopRunContext,
        *,
        message: str,
        payload: Dict[str, object],
    ) -> None:
        """记录 Plan-and-Solve 内部事件。"""

        await self.record_event(
            context,
            RunEvent.create(
                run_id=state.run_id,
                session_id=state.session_id,
                event_type=RunEventType.MESSAGE_CREATED,
                step=state.step,
                message=message,
                payload={"loop": self.name, **payload},
            ),
            state=state,
        )


def parse_plan_response(text: str, *, max_steps: int) -> Tuple[PlanStep, ...]:
    """优先解析结构化计划，失败时兼容编号文本计划。"""

    structured = parse_structured_plan(text, max_steps=max_steps)
    if structured:
        return structured
    return parse_numbered_plan(text, max_steps=max_steps)


def parse_structured_plan(text: str, *, max_steps: int) -> Tuple[PlanStep, ...]:
    """解析 JSON 计划。"""

    try:
        payload = parse_json_value(text)
    except ValueError:
        return ()

    if isinstance(payload, dict):
        raw_steps = payload.get("steps") or payload.get("plan") or ()
    elif isinstance(payload, list):
        raw_steps = payload
    else:
        return ()

    steps: List[PlanStep] = []
    for raw_step in raw_steps:
        step = build_plan_step(raw_step, index=len(steps) + 1)
        if step is None:
            continue
        steps.append(step)
        if len(steps) >= max_steps:
            break
    return tuple(steps)


def build_plan_step(raw_step: object, *, index: int) -> Optional[PlanStep]:
    """从结构化或字符串步骤创建 PlanStep。"""

    if isinstance(raw_step, str):
        instruction = raw_step.strip()
        if not instruction:
            return None
        return PlanStep(index=index, instruction=instruction)

    if not isinstance(raw_step, dict):
        return None

    instruction = first_non_empty_string(
        raw_step,
        "instruction",
        "task",
        "description",
        "step",
    )
    if instruction is None:
        return None

    tool_name = optional_string(raw_step.get("tool_name") or raw_step.get("tool"))
    tool_arguments = parse_tool_arguments(
        raw_step.get("tool_arguments", raw_step.get("arguments", {}))
    )
    requires_tool = bool(raw_step.get("requires_tool") or tool_name)
    return PlanStep(
        index=index,
        instruction=instruction,
        requires_tool=requires_tool,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        metadata={
            "source": "structured_plan",
        },
    )


def parse_numbered_plan(text: str, *, max_steps: int) -> Tuple[PlanStep, ...]:
    """解析模型返回的编号计划。"""

    steps: List[PlanStep] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)[\).、]\s+(.+?)\s*$", line)
        if not match:
            continue
        instruction = match.group(2).strip()
        if not instruction:
            continue
        steps.append(PlanStep(index=len(steps) + 1, instruction=instruction))
        if len(steps) >= max_steps:
            break
    return tuple(steps)


def parse_plan_step_decision(text: str) -> PlanStepDecision:
    """解析单步执行决策，兼容 JSON、ReAct 文本和普通答案。"""

    structured = parse_structured_step_decision(text)
    if structured is not None:
        return structured

    final_match = re.search(r"Final Answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.FINAL,
            output=final_match.group(1).strip(),
        )

    result_match = re.search(r"Step Result\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if result_match:
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.FINAL,
            output=result_match.group(1).strip(),
        )

    action_match = re.search(
        r"^Action\s*:\s*([a-zA-Z0-9_.-]+)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if action_match:
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.TOOL_CALL,
            tool_name=action_match.group(1).strip(),
            tool_arguments=parse_action_input(extract_action_input(text)),
        )

    return PlanStepDecision(
        decision_type=PlanStepDecisionType.FINAL,
        output=text.strip(),
    )


def parse_structured_step_decision(text: str) -> Optional[PlanStepDecision]:
    """解析 JSON 步骤决策。"""

    try:
        payload = parse_json_value(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    raw_type = optional_string(
        payload.get("type")
        or payload.get("decision_type")
        or payload.get("kind")
    )
    action = optional_string(payload.get("action"))
    normalized_type = (raw_type or "").strip().lower().replace("-", "_")

    if normalized_type in {"tool", "tool_call", "use_tool"} or payload.get("tool_name"):
        tool_name = optional_string(
            payload.get("tool_name")
            or payload.get("tool")
            or payload.get("name")
            or action
        )
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.TOOL_CALL,
            tool_name=tool_name,
            tool_arguments=parse_tool_arguments(
                payload.get(
                    "tool_arguments",
                    payload.get("arguments", payload.get("action_input", {})),
                )
            ),
            thought=optional_string(payload.get("thought") or payload.get("reason")),
            metadata={"source": "json"},
        )

    if normalized_type in {"final", "answer", "result", "done"} or any(
        key in payload for key in ("output", "answer", "final_answer", "result")
    ):
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.FINAL,
            output=str(
                payload.get(
                    "output",
                    payload.get("answer", payload.get("final_answer", payload.get("result", ""))),
                )
            ).strip(),
            thought=optional_string(payload.get("thought") or payload.get("reason")),
            metadata={"source": "json"},
        )

    if action and action.lower() not in {"final", "answer", "result", "done"}:
        return PlanStepDecision(
            decision_type=PlanStepDecisionType.TOOL_CALL,
            tool_name=action,
            tool_arguments=parse_tool_arguments(
                payload.get("action_input", payload.get("arguments", {}))
            ),
            thought=optional_string(payload.get("thought") or payload.get("reason")),
            metadata={"source": "json_action"},
        )

    return None


def format_solve_step_prompt(
    step: PlanStep,
    completed_steps: List[PlanStep],
    *,
    available_tools: str,
    iteration: int,
    max_iterations: int,
) -> str:
    """格式化单步求解提示。"""

    tool_hint = ""
    if step.requires_tool:
        tool_hint = (
            "\nThe plan suggests using a tool"
            f"{' named ' + step.tool_name if step.tool_name else ''}."
            f"\nSuggested arguments: {compact_json(step.tool_arguments)}"
        )

    return (
        f"Execute plan step {step.index}.\n"
        f"Instruction: {step.instruction}\n"
        f"{tool_hint}\n\n"
        f"Completed previous steps:\n{format_completed_steps(completed_steps)}\n\n"
        f"Available tools:\n{available_tools}\n\n"
        f"Iteration {iteration} of {max_iterations} for this step.\n"
        "Return exactly one JSON object in one of these forms:\n"
        '{"type":"tool_call","thought":"why the tool is needed","tool_name":"tool_name","tool_arguments":{}}\n'
        '{"type":"final","output":"the completed result for this step"}\n'
        "Do not call a tool unless the result is needed to complete this step."
    )


def format_final_synthesis_prompt(completed_steps: List[PlanStep]) -> str:
    """格式化最终综合提示。"""

    return FINAL_SYNTHESIS_PROMPT + "\n\n" + format_completed_steps(completed_steps)


def format_completed_steps(completed_steps: List[PlanStep]) -> str:
    """格式化已完成步骤。"""

    if not completed_steps:
        return "None."
    return "\n\n".join(
        f"{step.index}. {step.instruction}\nResult: {step.output or ''}"
        for step in completed_steps
    )


def format_available_tools(tool_executor: Optional[ToolExecutor]) -> str:
    """格式化当前可用工具。"""

    if tool_executor is None:
        return "No tools are available."

    specs = tool_executor.registry.list_specs()
    if not specs:
        return "No tools are registered."

    lines = []
    for spec in specs:
        lines.append(
            f"- {spec.name}: {spec.description}; input_schema={compact_json(spec.input_schema)}"
        )
    return "\n".join(lines)


def extract_action_input(text: str) -> str:
    """提取 Action Input 后面的内容。"""

    match = re.search(r"Action Input\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "{}"
    return match.group(1).strip()


def parse_action_input(raw: str) -> Dict[str, Any]:
    """把 Action Input 解析为工具参数字典。"""

    return parse_tool_arguments(strip_code_fence(raw.strip()))


def parse_tool_arguments(value: object) -> Dict[str, Any]:
    """把工具参数规范化为字典。"""

    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = parse_json_value(stripped)
        except ValueError:
            return {"input": stripped}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {"input": parsed}
    return {"input": value}


def parse_json_value(text: str) -> object:
    """从文本中解析第一个 JSON 值。"""

    return parse_json_payload(text)


def strip_code_fence(text: str) -> str:
    """移除包裹 JSON 的 Markdown code fence。"""

    return strip_json_code_fence(text)


def first_non_empty_string(data: Dict[str, object], *keys: str) -> Optional[str]:
    """返回第一个非空字符串字段。"""

    for key in keys:
        value = optional_string(data.get(key))
        if value:
            return value
    return None


def optional_string(value: object) -> Optional[str]:
    """把可选字段转换为去空白字符串。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def compact_json(value: object) -> str:
    """把结构化值紧凑格式化为 JSON。"""

    return compact_json_value(value)


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


def tool_result_to_record(result: ToolResult, *, iteration: int) -> Dict[str, object]:
    """把工具结果压缩为计划步骤 metadata。"""

    return {
        "iteration": iteration,
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "tool_result_id": result.id,
        "status": result.status.value,
        "is_error": result.is_error,
        "duration_ms": result.duration_ms,
    }


def count_tool_calls(steps: Iterable[PlanStep]) -> int:
    """统计计划执行中的工具调用次数。"""

    total = 0
    for step in steps:
        tool_calls = step.metadata.get("tool_calls", ())
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total


__all__ = [
    "DEFAULT_PLAN_SOLVE_SYSTEM_PROMPT",
    "FINAL_SYNTHESIS_PROMPT",
    "PLAN_PROMPT",
    "PlanAndSolveLoop",
    "PlanAndSolveTrace",
    "PlanStep",
    "PlanStepDecision",
    "PlanStepDecisionType",
    "PlanStepStatus",
    "build_plan_step",
    "compact_json",
    "count_tool_calls",
    "extract_action_input",
    "format_available_tools",
    "format_completed_steps",
    "format_final_synthesis_prompt",
    "format_solve_step_prompt",
    "format_tool_error",
    "parse_action_input",
    "parse_json_value",
    "parse_numbered_plan",
    "parse_plan_response",
    "parse_plan_step_decision",
    "parse_structured_plan",
    "parse_structured_step_decision",
    "parse_tool_arguments",
    "strip_code_fence",
    "tool_result_to_message",
]
