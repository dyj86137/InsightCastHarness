"""按任务场景选择 Agent Loop。

V2 先实现规则型选择器：根据任务文本识别工具需求、规划复杂度和反思校验需求，
再选择最合适的 Loop。后续可以替换为模型选择器或策略引擎。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import Field

from context.builder import ContextBuilder
from infra.config import LLMConfig
from infra.exception import ConfigError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM
from loops.base import BaseAgentLoop
from loops.registry import LoopRegistry, create_loop, normalize_loop_name
from tools.executor import ToolExecutor


class TaskScenario(str, Enum):
    """任务场景类型。"""

    TOOL_USE = "tool_use"
    PLANNING = "planning"
    REFLECTION = "reflection"
    SIMPLE = "simple"


class TaskProfile(SerializableModel):
    """任务画像。"""

    task: str
    scenarios: Tuple[TaskScenario, ...] = Field(default_factory=tuple)
    needs_tools: bool = False
    needs_planning: bool = False
    needs_reflection: bool = False
    complexity_score: int = 0
    matched_keywords: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验任务画像。"""

        if self.task is None:
            raise HarnessValidationError("TaskProfile task cannot be None.")
        if self.complexity_score < 0:
            raise HarnessValidationError(
                "TaskProfile complexity_score cannot be negative.",
                details={"complexity_score": self.complexity_score},
            )


class LoopSelectionResult(SerializableModel):
    """Loop 选择结果。"""

    loop_name: str
    scenario: TaskScenario
    profile: TaskProfile
    reason: str
    confidence: float = 1.0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Loop 选择结果。"""

        if not self.loop_name or not self.loop_name.strip():
            raise HarnessValidationError("LoopSelectionResult requires loop_name.")
        if self.confidence < 0 or self.confidence > 1:
            raise HarnessValidationError(
                "LoopSelectionResult confidence must be in [0, 1].",
                details={"confidence": self.confidence},
            )


class RuleBasedLoopSelector:
    """规则型 Loop 选择器。"""

    def __init__(
        self,
        *,
        default_loop: str = "plan_and_solve",
        tool_loop: str = "react",
        planning_loop: str = "plan_and_solve",
        reflection_loop: str = "reflection",
    ) -> None:
        self.default_loop = normalize_loop_name(default_loop)
        self.tool_loop = normalize_loop_name(tool_loop)
        self.planning_loop = normalize_loop_name(planning_loop)
        self.reflection_loop = normalize_loop_name(reflection_loop)

    def profile_task(self, task: object) -> TaskProfile:
        """识别任务场景画像。"""

        task_text = extract_task_text(task)
        normalized = normalize_task_text(task_text)
        matched_keywords: List[str] = []

        tool_matches = match_keywords(normalized, TOOL_KEYWORDS)
        planning_matches = match_keywords(normalized, PLANNING_KEYWORDS)
        reflection_matches = match_keywords(normalized, REFLECTION_KEYWORDS)
        matched_keywords.extend(tool_matches + planning_matches + reflection_matches)

        complexity_score = estimate_task_complexity(
            normalized,
            planning_matches=planning_matches,
            reflection_matches=reflection_matches,
        )
        needs_tools = bool(tool_matches)
        needs_planning = bool(planning_matches) or complexity_score >= 3
        needs_reflection = bool(reflection_matches)

        scenarios: List[TaskScenario] = []
        if needs_tools:
            scenarios.append(TaskScenario.TOOL_USE)
        if needs_reflection:
            scenarios.append(TaskScenario.REFLECTION)
        if needs_planning:
            scenarios.append(TaskScenario.PLANNING)
        if not scenarios:
            scenarios.append(TaskScenario.SIMPLE)

        return TaskProfile(
            task=task_text,
            scenarios=tuple(scenarios),
            needs_tools=needs_tools,
            needs_planning=needs_planning,
            needs_reflection=needs_reflection,
            complexity_score=complexity_score,
            matched_keywords=tuple(dedupe_preserve_order(matched_keywords)),
            metadata={
                "text_length": len(task_text),
                "normalized_length": len(normalized),
            },
        )

    def select(self, task: object) -> LoopSelectionResult:
        """根据任务画像选择 Loop。"""

        profile = self.profile_task(task)
        loop_name, scenario, reason, confidence = self._choose_loop(profile)
        return LoopSelectionResult(
            loop_name=loop_name,
            scenario=scenario,
            profile=profile,
            reason=reason,
            confidence=confidence,
            metadata={"selector": self.__class__.__name__},
        )

    def _choose_loop(
        self,
        profile: TaskProfile,
    ) -> Tuple[str, TaskScenario, str, float]:
        """根据任务画像选择具体 Loop。"""

        if profile.needs_tools:
            return (
                self.tool_loop,
                TaskScenario.TOOL_USE,
                "任务包含工具、文件、命令、检索或外部操作意图，优先选择 ReAct。",
                0.90,
            )
        if profile.needs_reflection:
            return (
                self.reflection_loop,
                TaskScenario.REFLECTION,
                "任务强调审查、验证、改进或高质量校验，选择 Reflection。",
                0.85,
            )
        if profile.needs_planning:
            return (
                self.planning_loop,
                TaskScenario.PLANNING,
                "任务复杂度较高或明确要求分阶段处理，选择 Plan-and-Solve。",
                0.80,
            )
        return (
            self.default_loop,
            TaskScenario.SIMPLE,
            "任务没有明显工具或反思需求，使用默认 Loop。",
            0.60,
        )


DEFAULT_LOOP_SELECTOR = RuleBasedLoopSelector()


def select_loop_for_task(
    task: object,
    *,
    selector: Optional[RuleBasedLoopSelector] = None,
) -> LoopSelectionResult:
    """选择适合当前任务的 Loop。"""

    selected_selector = selector or DEFAULT_LOOP_SELECTOR
    return selected_selector.select(task)


def create_loop_for_task(
    task: object,
    *,
    llm: BaseLLM,
    llm_config: Optional[LLMConfig] = None,
    context_builder: Optional[ContextBuilder] = None,
    tool_executor: Optional[ToolExecutor] = None,
    registry: Optional[LoopRegistry] = None,
    selector: Optional[RuleBasedLoopSelector] = None,
) -> BaseAgentLoop:
    """根据任务场景自动选择并创建 Loop。"""

    selection = select_loop_for_task(task, selector=selector)
    if selection.loop_name == "react" and tool_executor is None:
        raise ConfigError(
            "Selected React loop requires a ToolExecutor.",
            details={
                "loop": selection.loop_name,
                "scenario": selection.scenario.value,
                "reason": selection.reason,
            },
        )
    return create_loop(
        selection.loop_name,
        llm=llm,
        llm_config=llm_config,
        context_builder=context_builder,
        tool_executor=tool_executor,
        registry=registry,
    )


def extract_task_text(task: object) -> str:
    """从不同输入类型中提取任务文本。"""

    if isinstance(task, str):
        return task
    content = getattr(task, "content", None)
    if isinstance(content, str):
        return content
    messages = getattr(task, "messages", None)
    if messages is not None:
        parts = [
            getattr(message, "content", "")
            for message in messages
            if getattr(message, "content", "")
        ]
        return "\n".join(parts)
    return str(task)


def normalize_task_text(task_text: str) -> str:
    """规范化任务文本，便于关键词匹配。"""

    return task_text.lower().strip()


def match_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    """返回命中的关键词。"""

    matches = []
    for keyword in keywords:
        if keyword in text:
            matches.append(keyword)
    return matches


def estimate_task_complexity(
    text: str,
    *,
    planning_matches: Iterable[str],
    reflection_matches: Iterable[str],
) -> int:
    """估算任务复杂度，数值越高越应该使用规划型 Loop。"""

    score = 0
    if len(text) >= 200:
        score += 1
    if len(text) >= 600:
        score += 1
    score += min(len(tuple(planning_matches)), 3)
    if tuple(reflection_matches):
        score += 1
    if re.search(r"\b(first|then|finally|step by step|multi-step)\b", text):
        score += 1
    if re.search(r"(首先|然后|最后|分阶段|一步步|多步骤|计划)", text):
        score += 1
    return score


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """按顺序去重。"""

    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


TOOL_KEYWORDS = (
    "tool",
    "use tool",
    "search",
    "browse",
    "web",
    "internet",
    "read file",
    "write file",
    "open file",
    "readme",
    "file",
    "directory",
    "current directory",
    "workspace",
    "repo",
    "repository",
    "run command",
    "execute",
    "shell",
    "terminal",
    "python",
    "读取",
    "写入",
    "修改文件",
    "运行命令",
    "执行命令",
    "搜索",
    "联网",
    "浏览器",
    "工具",
)

PLANNING_KEYWORDS = (
    "plan",
    "steps",
    "step by step",
    "multi-step",
    "break down",
    "architecture",
    "design",
    "implement",
    "roadmap",
    "strategy",
    "规划",
    "计划",
    "步骤",
    "分阶段",
    "架构",
    "设计",
    "实现",
    "方案",
)

REFLECTION_KEYWORDS = (
    "review",
    "critique",
    "reflect",
    "verify",
    "validate",
    "check",
    "improve",
    "polish",
    "quality",
    "correctness",
    "审查",
    "评审",
    "反思",
    "验证",
    "检查",
    "改进",
    "优化",
    "质量",
    "正确性",
)


__all__ = [
    "DEFAULT_LOOP_SELECTOR",
    "PLANNING_KEYWORDS",
    "REFLECTION_KEYWORDS",
    "TOOL_KEYWORDS",
    "LoopSelectionResult",
    "RuleBasedLoopSelector",
    "TaskProfile",
    "TaskScenario",
    "create_loop_for_task",
    "dedupe_preserve_order",
    "estimate_task_complexity",
    "extract_task_text",
    "match_keywords",
    "normalize_task_text",
    "select_loop_for_task",
]
