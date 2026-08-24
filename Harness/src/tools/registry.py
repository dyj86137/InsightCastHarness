"""工具注册表。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from infra.exception import ToolNotFoundError, ValidationError as HarnessValidationError
from tools.base import BaseTool, ToolSpec


class ToolRegistry:
    """管理工具实例的注册表。"""

    def __init__(self, tools: Optional[Iterable[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: BaseTool, *, override: bool = False) -> None:
        """注册一个工具实例。"""

        self._validate_tool(tool)
        tool_name = normalize_tool_name(tool.name)
        if tool_name in self._tools and not override:
            raise HarnessValidationError(
                "Tool already registered.",
                details={"tool_name": tool_name},
            )
        self._tools[tool_name] = tool

    def register_many(
        self,
        tools: Iterable[BaseTool],
        *,
        override: bool = False,
    ) -> List[str]:
        """批量注册工具，并返回成功注册的工具名称。"""

        registered_names: List[str] = []
        for tool in tools:
            self.register(tool, override=override)
            registered_names.append(normalize_tool_name(tool.name))
        return registered_names

    def unregister(self, tool_name: str) -> None:
        """移除一个工具。"""

        normalized = normalize_tool_name(tool_name)
        if normalized not in self._tools:
            raise ToolNotFoundError(normalized)
        del self._tools[normalized]

    def get(self, tool_name: str) -> BaseTool:
        """按名称获取工具；不存在时抛出 ToolNotFoundError。"""

        normalized = normalize_tool_name(tool_name)
        tool = self._tools.get(normalized)
        if tool is None:
            raise ToolNotFoundError(normalized)
        return tool

    def find(self, tool_name: str) -> Optional[BaseTool]:
        """按名称查找工具；不存在时返回 None。"""

        return self._tools.get(normalize_tool_name(tool_name))

    def has(self, tool_name: str) -> bool:
        """判断工具是否已注册。"""

        return normalize_tool_name(tool_name) in self._tools

    def list_tools(self) -> List[BaseTool]:
        """返回所有工具实例。"""

        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """返回所有已注册工具名称。"""

        return sorted(self._tools)

    def list_specs(self) -> List[ToolSpec]:
        """返回所有工具能力描述。"""

        return [tool.spec() for tool in self.list_tools()]

    def list_by_source(self, source: str) -> List[BaseTool]:
        """按工具来源筛选工具。"""

        normalized_source = normalize_source(source)
        return [
            tool
            for tool in self.list_tools()
            if normalize_source(tool.spec().source) == normalized_source
        ]

    def list_by_tag(self, tag: str) -> List[BaseTool]:
        """按标签筛选工具。"""

        normalized_tag = normalize_tag(tag)
        return [
            tool
            for tool in self.list_tools()
            if normalized_tag in {normalize_tag(item) for item in tool.spec().tags}
        ]

    def list_read_only(self) -> List[BaseTool]:
        """返回只读工具。"""

        return [tool for tool in self.list_tools() if tool.is_read_only]

    def query(
        self,
        *,
        source: Optional[str] = None,
        tag: Optional[str] = None,
        read_only: Optional[bool] = None,
    ) -> List[BaseTool]:
        """按来源、标签和只读属性组合查询工具。"""

        tools = self.list_tools()
        if source is not None:
            normalized_source = normalize_source(source)
            tools = [
                tool
                for tool in tools
                if normalize_source(tool.spec().source) == normalized_source
            ]
        if tag is not None:
            normalized_tag = normalize_tag(tag)
            tools = [
                tool
                for tool in tools
                if normalized_tag in {normalize_tag(item) for item in tool.spec().tags}
            ]
        if read_only is not None:
            tools = [tool for tool in tools if tool.is_read_only == read_only]
        return tools

    def query_specs(
        self,
        *,
        source: Optional[str] = None,
        tag: Optional[str] = None,
        read_only: Optional[bool] = None,
    ) -> List[ToolSpec]:
        """按条件查询工具能力描述。"""

        return [
            tool.spec()
            for tool in self.query(source=source, tag=tag, read_only=read_only)
        ]

    def to_api_schema(self) -> List[Dict[str, object]]:
        """导出可给 LLM Provider 使用的工具 schema。"""

        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "metadata": {
                    "source": spec.source,
                    "version": spec.version,
                    "tags": list(spec.tags),
                    **spec.metadata,
                },
            }
            for spec in self.list_specs()
        ]

    def clear(self) -> None:
        """清空注册表。"""

        self._tools.clear()

    def _validate_tool(self, tool: BaseTool) -> None:
        """校验注册对象是否符合工具接口。"""

        if not isinstance(tool, BaseTool):
            raise HarnessValidationError(
                "ToolRegistry can only register BaseTool instances.",
                details={"actual_type": tool.__class__.__name__},
            )
        normalize_tool_name(tool.name)
        tool.spec()


def normalize_tool_name(tool_name: str) -> str:
    """统一工具名称格式。"""

    if not tool_name or not tool_name.strip():
        raise HarnessValidationError("Tool name cannot be empty.")
    return tool_name.strip()


def normalize_source(source: str) -> str:
    """统一工具来源名称。"""

    if not source or not source.strip():
        raise HarnessValidationError("Tool source cannot be empty.")
    return source.strip().lower()


def normalize_tag(tag: str) -> str:
    """统一工具标签名称。"""

    if not tag or not tag.strip():
        raise HarnessValidationError("Tool tag cannot be empty.")
    return tag.strip().lower()


__all__ = [
    "ToolRegistry",
    "normalize_source",
    "normalize_tag",
    "normalize_tool_name",
]
