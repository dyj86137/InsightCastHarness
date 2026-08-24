"""Agent Loop 注册表。

注册表提供统一的 Loop 名称、能力描述和创建入口。
上层 Runner 仍然接收已经构造好的 Loop；需要配置化切换时，可以先通过这里创建。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pydantic import Field

from context.builder import ContextBuilder
from infra.config import LLMConfig
from infra.exception import ConfigError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM
from loops.base import BaseAgentLoop
from loops.plan_solve import PlanAndSolveLoop
from loops.react import ReactLoop
from loops.reflection import ReflectionLoop
from tools.executor import ToolExecutor


LoopFactory = Callable[["LoopFactoryContext"], BaseAgentLoop]


class LoopSpec(SerializableModel):
    """Agent Loop 能力描述。"""

    name: str
    description: str
    requires_tools: bool = False
    supports_reflection: bool = False
    supports_planning: bool = False
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 LoopSpec。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError("LoopSpec requires non-empty name.")
        if not self.description or not self.description.strip():
            raise HarnessValidationError("LoopSpec requires non-empty description.")


@dataclass(frozen=True)
class LoopFactoryContext:
    """创建 Agent Loop 所需的依赖。"""

    llm: BaseLLM
    llm_config: Optional[LLMConfig] = None
    context_builder: Optional[ContextBuilder] = None
    tool_executor: Optional[ToolExecutor] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class LoopRegistration:
    """Loop 注册项。"""

    def __init__(self, spec: LoopSpec, factory: LoopFactory) -> None:
        self.spec = spec
        self.factory = factory


class LoopRegistry:
    """管理 Agent Loop 工厂的注册表。"""

    def __init__(self) -> None:
        self._loops: Dict[str, LoopRegistration] = {}

    def register(
        self,
        *,
        spec: LoopSpec,
        factory: LoopFactory,
        override: bool = False,
    ) -> None:
        """注册一个 Loop 工厂。"""

        loop_name = normalize_loop_name(spec.name)
        if loop_name in self._loops and not override:
            raise ConfigError(
                "Agent loop already registered.",
                details={"loop": loop_name},
            )
        self._loops[loop_name] = LoopRegistration(spec=spec, factory=factory)

    def has(self, loop_name: str) -> bool:
        """判断 Loop 是否已经注册。"""

        return normalize_loop_name(loop_name) in self._loops

    def get_spec(self, loop_name: str) -> LoopSpec:
        """返回 Loop 能力描述。"""

        normalized = normalize_loop_name(loop_name)
        registration = self._loops.get(normalized)
        if registration is None:
            raise unknown_loop_error(normalized, self.list_loop_names())
        return registration.spec

    def list_specs(self) -> List[LoopSpec]:
        """返回全部 Loop 能力描述。"""

        return [registration.spec for registration in self._loops.values()]

    def list_loop_names(self) -> List[str]:
        """返回全部已注册 Loop 名称。"""

        return sorted(self._loops)

    def create(
        self,
        loop_name: str,
        context: LoopFactoryContext,
    ) -> BaseAgentLoop:
        """根据名称和依赖创建 Loop 实例。"""

        normalized = normalize_loop_name(loop_name)
        registration = self._loops.get(normalized)
        if registration is None:
            raise unknown_loop_error(normalized, self.list_loop_names())
        return registration.factory(context)


def normalize_loop_name(loop_name: str) -> str:
    """统一 Loop 名称格式。"""

    if not loop_name or not loop_name.strip():
        raise ConfigError("Agent loop name cannot be empty.")
    return loop_name.strip().lower().replace("-", "_")


def unknown_loop_error(loop_name: str, available: List[str]) -> ConfigError:
    """构造未知 Loop 配置错误。"""

    return ConfigError(
        "Unknown Agent loop.",
        details={
            "loop": loop_name,
            "available_loops": available,
        },
    )


def create_react_loop(context: LoopFactoryContext) -> ReactLoop:
    """创建 ReAct Loop。"""

    if context.tool_executor is None:
        raise ConfigError(
            "ReactLoop requires a ToolExecutor.",
            details={"loop": "react"},
        )
    return ReactLoop(
        llm=context.llm,
        tool_executor=context.tool_executor,
        context_builder=context.context_builder,
        llm_config=context.llm_config,
    )


def create_plan_and_solve_loop(context: LoopFactoryContext) -> PlanAndSolveLoop:
    """创建 Plan-and-Solve Loop。"""

    return PlanAndSolveLoop(
        llm=context.llm,
        context_builder=context.context_builder,
        llm_config=context.llm_config,
        tool_executor=context.tool_executor,
    )


def create_reflection_loop(context: LoopFactoryContext) -> ReflectionLoop:
    """创建 Reflection Loop。"""

    return ReflectionLoop(
        llm=context.llm,
        context_builder=context.context_builder,
        llm_config=context.llm_config,
    )


def create_default_loop_registry() -> LoopRegistry:
    """创建 V2 默认 Agent Loop 注册表。"""

    registry = LoopRegistry()
    registry.register(
        spec=LoopSpec(
            name="react",
            description="ReAct loop with tool use.",
            requires_tools=True,
            supports_reflection=False,
            supports_planning=False,
        ),
        factory=create_react_loop,
    )
    registry.register(
        spec=LoopSpec(
            name="plan_and_solve",
            description="Plan first, execute each step with optional tool use, then synthesize a final answer.",
            requires_tools=False,
            supports_reflection=False,
            supports_planning=True,
            metadata={"supports_optional_tools": True},
        ),
        factory=create_plan_and_solve_loop,
    )
    registry.register(
        spec=LoopSpec(
            name="reflection",
            description="Answer, critique, and revise until accepted or limit reached.",
            requires_tools=False,
            supports_reflection=True,
            supports_planning=False,
        ),
        factory=create_reflection_loop,
    )
    return registry


DEFAULT_LOOP_REGISTRY = create_default_loop_registry()


def create_loop(
    loop_name: str,
    *,
    llm: BaseLLM,
    llm_config: Optional[LLMConfig] = None,
    context_builder: Optional[ContextBuilder] = None,
    tool_executor: Optional[ToolExecutor] = None,
    registry: Optional[LoopRegistry] = None,
) -> BaseAgentLoop:
    """使用默认依赖对象创建指定 Loop。"""

    selected_registry = registry or DEFAULT_LOOP_REGISTRY
    return selected_registry.create(
        loop_name,
        LoopFactoryContext(
            llm=llm,
            llm_config=llm_config,
            context_builder=context_builder,
            tool_executor=tool_executor,
        ),
    )


__all__ = [
    "DEFAULT_LOOP_REGISTRY",
    "LoopFactory",
    "LoopFactoryContext",
    "LoopRegistration",
    "LoopRegistry",
    "LoopSpec",
    "create_default_loop_registry",
    "create_loop",
    "create_plan_and_solve_loop",
    "create_react_loop",
    "create_reflection_loop",
    "normalize_loop_name",
    "unknown_loop_error",
]
