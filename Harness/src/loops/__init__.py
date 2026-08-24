"""Agent Loop 实现集合。"""

from loops.base import BaseAgentLoop, LoopRunContext
from loops.plan_solve import (
    PlanAndSolveLoop,
    PlanAndSolveTrace,
    PlanStep,
    PlanStepDecision,
    PlanStepDecisionType,
)
from loops.react import ReactLoop
from loops.registry import (
    DEFAULT_LOOP_REGISTRY,
    LoopFactoryContext,
    LoopRegistry,
    LoopSpec,
    create_loop,
)
from loops.reflection import ReflectionDecision, ReflectionIteration, ReflectionLoop, ReflectionTrace
from loops.selector import (
    DEFAULT_LOOP_SELECTOR,
    LoopSelectionResult,
    RuleBasedLoopSelector,
    TaskProfile,
    TaskScenario,
    create_loop_for_task,
    select_loop_for_task,
)


__all__ = [
    "BaseAgentLoop",
    "LoopRunContext",
    "PlanAndSolveLoop",
    "PlanAndSolveTrace",
    "PlanStep",
    "PlanStepDecision",
    "PlanStepDecisionType",
    "ReactLoop",
    "DEFAULT_LOOP_REGISTRY",
    "LoopFactoryContext",
    "LoopRegistry",
    "LoopSpec",
    "create_loop",
    "DEFAULT_LOOP_SELECTOR",
    "LoopSelectionResult",
    "RuleBasedLoopSelector",
    "TaskProfile",
    "TaskScenario",
    "create_loop_for_task",
    "select_loop_for_task",
    "ReflectionIteration",
    "ReflectionDecision",
    "ReflectionLoop",
    "ReflectionTrace",
]
