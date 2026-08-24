"""Reflection Agent Loop.

该循环先生成候选答案，再使用结构化评审结果判断是否通过；
未通过时按评审意见修订，直到通过或达到最大反思次数。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from context.builder import ContextBuilder
from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.message import Message
from core.state import AgentState
from infra.config import LLMConfig
from infra.exception import ValidationError as HarnessValidationError
from infra.json import parse_json_value as parse_json_payload, strip_json_code_fence
from infra.serialization import SerializableModel
from llm.base import BaseLLM, ChatRequest, ChatResponse
from loops.base import BaseAgentLoop, LoopRunContext


DEFAULT_REFLECTION_SYSTEM_PROMPT = """You are a reflective agent.

Produce a useful answer, then evaluate it for correctness, completeness, assumptions, and clarity.
When the answer needs improvement, revise it directly instead of only describing the problem.
"""

INITIAL_ANSWER_PROMPT = """Answer the user's task directly.

Return only the candidate answer.
"""

CRITIQUE_PROMPT_TEMPLATE = """Review the candidate answer below.

Evaluate correctness, completeness, missing assumptions, and clarity.
Return exactly one JSON object:
{{
  "passed": true,
  "score": 0.0,
  "critique": "short assessment",
  "issues": [],
  "revised_answer": null
}}

Rules:
- Set passed=true only when the candidate answer is good enough to return.
- Use score from 0.0 to 1.0.
- If passed=false, include concrete issues and provide revised_answer as a full replacement answer whenever possible.
- Do not wrap the JSON in Markdown.

Candidate answer:
{answer}
"""

REVISION_PROMPT_TEMPLATE = """Revise the candidate answer using the critique.

Candidate answer:
{answer}

Critique:
{critique}

Return only the improved answer.
"""


class ReflectionDecision(SerializableModel):
    """一次结构化反思评审结果。"""

    passed: bool
    critique: str = ""
    score: Optional[float] = None
    issues: Tuple[str, ...] = Field(default_factory=tuple)
    revised_answer: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验反思决策。"""

        if self.score is not None and (self.score < 0 or self.score > 1):
            raise HarnessValidationError(
                "ReflectionDecision score must be in [0, 1].",
                details={"score": self.score},
            )
        if not self.passed and not self.critique and not self.issues and not self.revised_answer:
            raise HarnessValidationError(
                "Failed ReflectionDecision requires critique, issues, or revised_answer."
            )


class ReflectionIteration(SerializableModel):
    """一次反思迭代记录。"""

    index: int
    answer: str
    critique: Optional[str] = None
    passed: bool = False
    score: Optional[float] = None
    issues: Tuple[str, ...] = Field(default_factory=tuple)
    revised_answer: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验反思迭代。"""

        if self.index <= 0:
            raise HarnessValidationError(
                "ReflectionIteration index must be positive.",
                details={"index": self.index},
            )
        if self.answer is None:
            raise HarnessValidationError("ReflectionIteration answer cannot be None.")
        if self.score is not None and (self.score < 0 or self.score > 1):
            raise HarnessValidationError(
                "ReflectionIteration score must be in [0, 1].",
                details={"index": self.index, "score": self.score},
            )


class ReflectionTrace(SerializableModel):
    """Reflection 执行摘要。"""

    iterations: Tuple[ReflectionIteration, ...] = Field(default_factory=tuple)
    final_answer: Optional[str] = None
    accepted: bool = False
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class ReflectionLoop(BaseAgentLoop):
    """带结构化自评和修订能力的 Agent Loop。"""

    name = "reflection"

    def __init__(
        self,
        *,
        llm: BaseLLM,
        context_builder: Optional[ContextBuilder] = None,
        llm_config: Optional[LLMConfig] = None,
        max_reflections: int = 2,
    ) -> None:
        if max_reflections < 0:
            raise HarnessValidationError(
                "ReflectionLoop max_reflections cannot be negative.",
                details={"max_reflections": max_reflections},
            )
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(
            system_message=Message.system(DEFAULT_REFLECTION_SYSTEM_PROMPT)
        )
        self.llm_config = llm_config or LLMConfig(
            provider=llm.provider_name,
            model=llm.model_name,
        )
        self.max_reflections = max_reflections

    async def run(self, state: AgentState, context: LoopRunContext) -> AgentState:
        """执行 Reflection 循环。"""

        current_state = state

        self.ensure_can_continue(current_state)
        await self.record_step_started(current_state, context)
        answer_response = await self._call_llm(
            current_state,
            context,
            purpose="initial_answer",
            prompt=INITIAL_ANSWER_PROMPT,
        )
        current_state = current_state.add_message(answer_response.message)
        current_answer = answer_response.text.strip()
        await self.record_step_completed(current_state, context)
        current_state = current_state.next_step()

        iterations: List[ReflectionIteration] = []
        accepted = False

        for index in range(1, self.max_reflections + 2):
            self.ensure_can_continue(current_state)
            await self.record_step_started(current_state, context)

            candidate_answer = current_answer
            critique_response = await self._call_llm(
                current_state,
                context,
                purpose="critique",
                prompt=format_critique_prompt(candidate_answer),
                metadata={"reflection_iteration": index},
            )
            current_state = current_state.add_message(critique_response.message)
            decision = parse_reflection_decision(critique_response.text)
            revised_answer: Optional[str] = None
            can_revise = index <= self.max_reflections

            if decision.passed:
                accepted = True
            elif can_revise:
                if decision.revised_answer:
                    revised_answer = decision.revised_answer.strip()
                    current_answer = revised_answer
                    current_state = current_state.add_message(
                        Message.assistant(
                            current_answer,
                            metadata={
                                "loop": self.name,
                                "purpose": "structured_revision",
                                "reflection_iteration": index,
                                "source_response_id": critique_response.id,
                            },
                        )
                    )
                else:
                    revision_response = await self._call_llm(
                        current_state,
                        context,
                        purpose="revision",
                        prompt=format_revision_prompt(candidate_answer, decision.critique),
                        metadata={"reflection_iteration": index},
                    )
                    revised_answer = revision_response.text.strip()
                    current_answer = revised_answer
                    current_state = current_state.add_message(revision_response.message)

            iteration = ReflectionIteration(
                index=index,
                answer=candidate_answer,
                critique=decision.critique,
                passed=decision.passed,
                score=decision.score,
                issues=decision.issues,
                revised_answer=revised_answer,
                metadata=decision.metadata,
            )
            iterations.append(iteration)
            await self._record_loop_event(
                current_state,
                context,
                message="Reflection critique completed.",
                payload={"iteration": iteration.to_dict(exclude_none=True)},
            )

            if accepted or not can_revise:
                break

            await self.record_step_completed(current_state, context)
            current_state = current_state.next_step()

        current_state = current_state.with_metadata(
            "reflection",
            ReflectionTrace(
                iterations=tuple(iterations),
                final_answer=current_answer,
                accepted=accepted,
                metadata={
                    "max_reflections": self.max_reflections,
                    "iteration_count": len(iterations),
                },
            ).to_dict(exclude_none=True),
        )
        final_state = current_state.mark_completed(current_answer)
        await self.record_step_completed(final_state, context)
        return final_state

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
                    message="Reflection LLM call failed.",
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

    async def _record_loop_event(
        self,
        state: AgentState,
        context: LoopRunContext,
        *,
        message: str,
        payload: Dict[str, object],
    ) -> None:
        """记录 Reflection 内部事件。"""

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


def parse_reflection_decision(text: str) -> ReflectionDecision:
    """解析结构化反思结果，失败时兼容 PASS/REVISE 文本协议。"""

    structured = parse_structured_reflection_decision(text)
    if structured is not None:
        return structured

    passed = is_reflection_passed(text)
    critique = strip_reflection_prefix(text).strip()
    revised_answer = extract_revised_answer(text)
    if not passed and not critique:
        critique = "Reflection requested revision."
    return ReflectionDecision(
        passed=passed,
        critique=critique,
        revised_answer=revised_answer,
        metadata={"source": "text_protocol"},
    )


def parse_structured_reflection_decision(text: str) -> Optional[ReflectionDecision]:
    """解析 JSON 反思结果。"""

    try:
        payload = parse_json_value(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    passed = parse_passed(payload)
    if passed is None:
        return None

    return ReflectionDecision(
        passed=passed,
        critique=str(payload.get("critique", payload.get("assessment", ""))).strip(),
        score=parse_score(payload.get("score")),
        issues=tuple(normalize_issues(payload.get("issues", ()))),
        revised_answer=optional_string(
            payload.get("revised_answer")
            or payload.get("revision")
            or payload.get("improved_answer")
        ),
        metadata={"source": "json"},
    )


def parse_passed(payload: Dict[str, object]) -> Optional[bool]:
    """从结构化 payload 中解析是否通过。"""

    if isinstance(payload.get("passed"), bool):
        return bool(payload["passed"])
    if isinstance(payload.get("approved"), bool):
        return bool(payload["approved"])

    status = optional_string(
        payload.get("status")
        or payload.get("verdict")
        or payload.get("decision")
    )
    if status is None:
        return None

    normalized = status.lower().replace("-", "_").strip()
    if normalized in {"pass", "passed", "approve", "approved", "accept", "accepted"}:
        return True
    if normalized in {"revise", "revision", "fail", "failed", "reject", "rejected"}:
        return False
    return None


def parse_score(value: object) -> Optional[float]:
    """解析 0 到 1 的分数。"""

    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def normalize_issues(value: object) -> List[str]:
    """把 issues 字段规范化为字符串列表。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        issues = []
        for item in value:
            text = optional_string(item)
            if text:
                issues.append(text)
        return issues
    text = optional_string(value)
    return [text] if text else []


def is_reflection_passed(text: str) -> bool:
    """判断反思结果是否通过。"""

    first_token = first_meaningful_token(text)
    return first_token in {"PASS", "PASSED", "APPROVED", "ACCEPTED"}


def first_meaningful_token(text: str) -> str:
    """提取响应中的第一个语义 token。"""

    match = re.search(r"[A-Za-z]+", text or "")
    if not match:
        return ""
    return match.group(0).upper()


def strip_reflection_prefix(text: str) -> str:
    """去掉 PASS/REVISE 这类旧协议前缀。"""

    return re.sub(
        r"^\s*(PASS|PASSED|APPROVED|ACCEPTED|REVISE|REVISION|FAIL|FAILED|REJECTED)\b\s*:?",
        "",
        text or "",
        flags=re.IGNORECASE,
    )


def extract_revised_answer(text: str) -> Optional[str]:
    """从文本协议中提取 Revised Answer 段落。"""

    match = re.search(
        r"Revised Answer\s*:\s*(.*)",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip() or None


def format_critique_prompt(answer: str) -> str:
    """格式化结构化评审提示。"""

    return CRITIQUE_PROMPT_TEMPLATE.format(answer=answer)


def format_revision_prompt(answer: str, critique: str) -> str:
    """格式化修订提示。"""

    return REVISION_PROMPT_TEMPLATE.format(answer=answer, critique=critique)


def parse_json_value(text: str) -> object:
    """从文本中解析第一个 JSON 值。"""

    return parse_json_payload(text)


def strip_code_fence(text: str) -> str:
    """移除包裹 JSON 的 Markdown code fence。"""

    return strip_json_code_fence(text)


def optional_string(value: object) -> Optional[str]:
    """把可选字段转换为去空白字符串。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CRITIQUE_PROMPT_TEMPLATE",
    "DEFAULT_REFLECTION_SYSTEM_PROMPT",
    "INITIAL_ANSWER_PROMPT",
    "REVISION_PROMPT_TEMPLATE",
    "ReflectionDecision",
    "ReflectionIteration",
    "ReflectionLoop",
    "ReflectionTrace",
    "extract_revised_answer",
    "first_meaningful_token",
    "format_critique_prompt",
    "format_revision_prompt",
    "is_reflection_passed",
    "normalize_issues",
    "parse_reflection_decision",
    "parse_structured_reflection_decision",
    "strip_reflection_prefix",
]
