"""Skill 能力接入框架。

Skill 是一组工具能力的封装。它本身不改变工具执行协议，而是通过 SkillToolProvider
把 Skill 内的工具适配成普通 BaseTool，再接入现有 ToolDiscoveryManager 和 ToolRegistry。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.tool import ToolCall, ToolResult
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from tools.base import BaseTool, ToolExecutionContext, ToolSpec
from tools.discovery import ToolProvider, dedupe_tools_by_name
from tools.registry import normalize_tool_name


class SkillSpec(SerializableModel):
    """Skill 能力描述。"""

    name: str
    description: str
    version: Optional[str] = None
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    tool_specs: Tuple[ToolSpec, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Skill 描述。"""

        normalize_skill_name(self.name)
        if not self.description or not self.description.strip():
            raise HarnessValidationError(
                "SkillSpec requires non-empty description.",
                details={"skill_name": self.name},
            )
        invalid_tags = [tag for tag in self.tags if not tag or not tag.strip()]
        if invalid_tags:
            raise HarnessValidationError(
                "SkillSpec tags cannot contain empty values.",
                details={"skill_name": self.name, "invalid_tags": invalid_tags},
            )


class BaseSkill(ABC):
    """所有 Skill 必须遵守的统一接口。"""

    name: str
    description: str
    version: Optional[str] = None
    tags: Tuple[str, ...] = ()
    metadata: Optional[Dict[str, Any]] = None

    def spec(self) -> SkillSpec:
        """返回 Skill 能力描述。"""

        return SkillSpec(
            name=self.name,
            description=self.description,
            version=self.version,
            tags=tuple(self.tags),
            tool_specs=tuple(tool.spec() for tool in self.list_tools()),
            metadata=dict(self.metadata or {}),
        )

    @abstractmethod
    def list_tools(self) -> List[BaseTool]:
        """返回 Skill 暴露的工具实例。"""


class StaticSkill(BaseSkill):
    """基于固定工具列表的 Skill。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tools: Iterable[BaseTool],
        version: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = normalize_skill_name(name)
        self.description = description
        self.version = version
        self.tags = tuple(tags or ())
        self.metadata = metadata or {}
        self._tools = list(tools)
        for tool in self._tools:
            validate_skill_tool(tool, skill_name=self.name)
        self.spec()

    def list_tools(self) -> List[BaseTool]:
        """返回固定工具列表。"""

        return list(self._tools)


class SkillToolAdapter(BaseTool):
    """把 Skill 内部工具适配成可注册的普通工具。"""

    def __init__(
        self,
        *,
        skill: BaseSkill,
        tool: BaseTool,
        namespace_tools: bool = True,
        namespace_separator: str = ".",
    ) -> None:
        self.skill = skill
        self.wrapped_tool = tool
        self.original_tool_name = normalize_tool_name(tool.name)
        self.name = build_skill_tool_name(
            skill,
            tool,
            namespace_tools=namespace_tools,
            namespace_separator=namespace_separator,
        )
        self.description = build_skill_tool_description(skill, tool)
        self.input_model = tool.input_model
        self.is_read_only = tool.is_read_only
        self.source = "skill"
        self.version = tool.version or skill.version
        self.tags = merge_tags(("skill", skill.name), skill.tags, tool.tags)
        self.metadata = build_skill_tool_metadata(skill, tool)

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """委托给 Skill 内部工具执行。"""

        return await self.wrapped_tool.execute(tool_call, arguments, context)


class SkillRegistry:
    """管理 Skill 实例的注册表。"""

    def __init__(self, skills: Optional[Iterable[BaseSkill]] = None) -> None:
        self._skills: Dict[str, BaseSkill] = {}
        for skill in skills or ():
            self.register(skill)

    def register(self, skill: BaseSkill, *, override: bool = False) -> None:
        """注册一个 Skill。"""

        self._validate_skill(skill)
        skill_name = normalize_skill_name(skill.name)
        if skill_name in self._skills and not override:
            raise HarnessValidationError(
                "Skill already registered.",
                details={"skill_name": skill_name},
            )
        self._skills[skill_name] = skill

    def register_many(
        self,
        skills: Iterable[BaseSkill],
        *,
        override: bool = False,
    ) -> List[str]:
        """批量注册 Skill，并返回成功注册的名称。"""

        registered_names: List[str] = []
        for skill in skills:
            self.register(skill, override=override)
            registered_names.append(normalize_skill_name(skill.name))
        return registered_names

    def get(self, skill_name: str) -> BaseSkill:
        """按名称获取 Skill。"""

        normalized = normalize_skill_name(skill_name)
        skill = self._skills.get(normalized)
        if skill is None:
            raise HarnessValidationError(
                "Skill not found.",
                details={"skill_name": normalized},
            )
        return skill

    def find(self, skill_name: str) -> Optional[BaseSkill]:
        """按名称查找 Skill；不存在时返回 None。"""

        return self._skills.get(normalize_skill_name(skill_name))

    def has(self, skill_name: str) -> bool:
        """判断 Skill 是否已注册。"""

        return normalize_skill_name(skill_name) in self._skills

    def list_skills(self) -> List[BaseSkill]:
        """返回所有 Skill。"""

        return list(self._skills.values())

    def list_specs(self) -> List[SkillSpec]:
        """返回所有 Skill 描述。"""

        return [skill.spec() for skill in self.list_skills()]

    def clear(self) -> None:
        """清空注册表。"""

        self._skills.clear()

    def _validate_skill(self, skill: BaseSkill) -> None:
        """校验注册对象是否符合 Skill 接口。"""

        if not isinstance(skill, BaseSkill):
            raise HarnessValidationError(
                "SkillRegistry can only register BaseSkill instances.",
                details={"actual_type": skill.__class__.__name__},
            )
        skill.spec()


class SkillToolProvider(ToolProvider):
    """把 SkillRegistry 中的工具暴露给 ToolDiscoveryManager。"""

    def __init__(
        self,
        skills: Optional[Iterable[BaseSkill]] = None,
        *,
        registry: Optional[SkillRegistry] = None,
        namespace_tools: bool = True,
        namespace_separator: str = ".",
        name: str = "skill",
        source: str = "skill",
        description: str = "Skill tool provider.",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.registry = registry or SkillRegistry()
        for skill in skills or ():
            self.registry.register(skill)
        self.namespace_tools = namespace_tools
        self.namespace_separator = namespace_separator
        self.name = name
        self.source = source
        self.description = description
        self.metadata = metadata or {}
        self.spec()

    def discover(self) -> List[BaseTool]:
        """把所有 Skill 暴露的工具转换为可注册工具。"""

        tools: List[BaseTool] = []
        for skill in self.registry.list_skills():
            for tool in skill.list_tools():
                validate_skill_tool(tool, skill_name=skill.name)
                tools.append(
                    SkillToolAdapter(
                        skill=skill,
                        tool=tool,
                        namespace_tools=self.namespace_tools,
                        namespace_separator=self.namespace_separator,
                    )
                )
        return dedupe_tools_by_name(tools)


def normalize_skill_name(skill_name: str) -> str:
    """统一 Skill 名称。"""

    if not skill_name or not skill_name.strip():
        raise HarnessValidationError("Skill name cannot be empty.")
    return skill_name.strip()


def validate_skill_tool(tool: BaseTool, *, skill_name: str) -> None:
    """校验 Skill 暴露的工具。"""

    if not isinstance(tool, BaseTool):
        raise HarnessValidationError(
            "Skill can only expose BaseTool instances.",
            details={"skill_name": skill_name, "actual_type": tool.__class__.__name__},
        )
    tool.spec()


def build_skill_tool_name(
    skill: BaseSkill,
    tool: BaseTool,
    *,
    namespace_tools: bool = True,
    namespace_separator: str = ".",
) -> str:
    """构建 Skill 工具注册名称。"""

    tool_name = normalize_tool_name(tool.name)
    if not namespace_tools:
        return tool_name
    if not namespace_separator:
        raise HarnessValidationError("Skill namespace separator cannot be empty.")
    return f"{normalize_skill_name(skill.name)}{namespace_separator}{tool_name}"


def build_skill_tool_description(skill: BaseSkill, tool: BaseTool) -> str:
    """构建 Skill 工具描述。"""

    return f"[{skill.name}] {tool.description}"


def build_skill_tool_metadata(skill: BaseSkill, tool: BaseTool) -> Dict[str, Any]:
    """构建 Skill 工具元数据。"""

    metadata = dict(tool.metadata or {})
    metadata.update(
        {
            "skill_name": skill.name,
            "skill_version": skill.version,
            "skill_description": skill.description,
            "original_tool_name": tool.name,
            "original_tool_source": tool.source,
        }
    )
    return metadata


def merge_tags(*tag_groups: Iterable[str]) -> Tuple[str, ...]:
    """合并并去重标签。"""

    seen = set()
    result: List[str] = []
    for group in tag_groups:
        for tag in group:
            normalized = tag.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


__all__ = [
    "BaseSkill",
    "SkillRegistry",
    "SkillSpec",
    "SkillToolAdapter",
    "SkillToolProvider",
    "StaticSkill",
    "build_skill_tool_description",
    "build_skill_tool_metadata",
    "build_skill_tool_name",
    "merge_tags",
    "normalize_skill_name",
    "validate_skill_tool",
]
