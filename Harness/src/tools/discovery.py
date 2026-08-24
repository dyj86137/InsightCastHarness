"""工具发现与批量注册机制。

V2 通过 ToolProvider 统一不同来源的工具：
内置工具、本地模块工具、后续 Skill 工具、MCP 工具都可以先被发现，再注册到 ToolRegistry。
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from pydantic import Field

from infra.exception import HarnessError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from tools.base import BaseTool, ToolSpec
from tools.registry import ToolRegistry, normalize_tool_name


class ToolProviderSpec(SerializableModel):
    """工具来源描述。"""

    name: str
    source: str
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验工具来源描述。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError("ToolProviderSpec requires name.")
        if not self.source or not self.source.strip():
            raise HarnessValidationError(
                "ToolProviderSpec requires source.",
                details={"provider": self.name},
            )


class ToolDiscoveryReport(SerializableModel):
    """一次工具发现和注册的结果报告。"""

    provider_names: Tuple[str, ...] = Field(default_factory=tuple)
    discovered_tools: Tuple[ToolSpec, ...] = Field(default_factory=tuple)
    registered_names: Tuple[str, ...] = Field(default_factory=tuple)
    skipped_names: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[Dict[str, Any], ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    @property
    def discovered_count(self) -> int:
        """返回发现到的工具数量。"""

        return len(self.discovered_tools)

    @property
    def registered_count(self) -> int:
        """返回成功注册的工具数量。"""

        return len(self.registered_names)

    @property
    def error_count(self) -> int:
        """返回错误数量。"""

        return len(self.errors)


class ToolProvider(ABC):
    """所有工具来源适配器的统一接口。"""

    name: str = "provider"
    source: str = "local"
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def spec(self) -> ToolProviderSpec:
        """返回工具来源描述。"""

        return ToolProviderSpec(
            name=self.name,
            source=self.source,
            description=self.description,
            metadata=dict(self.metadata or {}),
        )

    @abstractmethod
    def discover(self) -> List[BaseTool]:
        """发现并返回该来源下的工具实例。"""


class StaticToolProvider(ToolProvider):
    """基于固定工具列表的 Provider。"""

    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        name: str = "static",
        source: str = "static",
        description: str = "Static tool provider.",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools = list(tools)
        self.name = name
        self.source = source
        self.description = description
        self.metadata = metadata or {}

    def discover(self) -> List[BaseTool]:
        """返回固定工具列表。"""

        return list(self._tools)


class BuiltinToolProvider(StaticToolProvider):
    """内置工具 Provider。"""

    def __init__(self) -> None:
        from tools.builtin.calculator import CalculatorTool

        super().__init__(
            [CalculatorTool()],
            name="builtin",
            source="builtin",
            description="Built-in myHarness tools.",
        )


class ModuleToolProvider(ToolProvider):
    """从 Python 模块中发现工具实例或工具类。"""

    def __init__(
        self,
        module_names: Iterable[str],
        *,
        name: str = "module",
        source: str = "module",
        description: str = "Python module tool provider.",
        include_instances: bool = True,
        include_classes: bool = True,
        include_imported_classes: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.module_names = tuple(module_names)
        self.name = name
        self.source = source
        self.description = description
        self.include_instances = include_instances
        self.include_classes = include_classes
        self.include_imported_classes = include_imported_classes
        self.metadata = metadata or {}

    def discover(self) -> List[BaseTool]:
        """导入模块并发现其中的工具。"""

        tools: List[BaseTool] = []
        for module_name in self.module_names:
            module = importlib.import_module(module_name)
            tools.extend(
                discover_tools_from_module(
                    module,
                    include_instances=self.include_instances,
                    include_classes=self.include_classes,
                    include_imported_classes=self.include_imported_classes,
                )
            )
        return dedupe_tools_by_name(tools)


class ToolDiscoveryManager:
    """工具发现与注册编排器。"""

    def __init__(self, providers: Optional[Iterable[ToolProvider]] = None) -> None:
        self._providers: List[ToolProvider] = []
        for provider in providers or ():
            self.add_provider(provider)

    def add_provider(self, provider: ToolProvider) -> None:
        """注册一个工具来源 Provider。"""

        if not isinstance(provider, ToolProvider):
            raise HarnessValidationError(
                "ToolDiscoveryManager can only register ToolProvider instances.",
                details={"actual_type": provider.__class__.__name__},
            )
        provider.spec()
        self._providers.append(provider)

    def list_providers(self) -> List[ToolProvider]:
        """返回所有 Provider。"""

        return list(self._providers)

    def list_provider_specs(self) -> List[ToolProviderSpec]:
        """返回所有 Provider 描述。"""

        return [provider.spec() for provider in self._providers]

    def discover_tools(self, *, continue_on_error: bool = True) -> List[BaseTool]:
        """从所有 Provider 发现工具。"""

        tools: List[BaseTool] = []
        for provider in self._providers:
            try:
                tools.extend(provider.discover())
            except Exception:
                if not continue_on_error:
                    raise
        return dedupe_tools_by_name(tools)

    def register_all(
        self,
        registry: ToolRegistry,
        *,
        override: bool = False,
        continue_on_error: bool = True,
    ) -> ToolDiscoveryReport:
        """发现所有 Provider 的工具并注册到 ToolRegistry。"""

        provider_names: List[str] = []
        discovered_specs: List[ToolSpec] = []
        registered_names: List[str] = []
        skipped_names: List[str] = []
        errors: List[Dict[str, Any]] = []

        for provider in self._providers:
            provider_names.append(provider.name)
            try:
                tools = provider.discover()
            except Exception as exc:
                errors.append(error_to_payload(exc, provider_name=provider.name))
                if not continue_on_error:
                    raise
                continue

            for tool in tools:
                try:
                    spec = tool.spec()
                    discovered_specs.append(spec)
                    if registry.has(tool.name) and not override:
                        skipped_names.append(normalize_tool_name(tool.name))
                        continue
                    registry.register(tool, override=override)
                    registered_names.append(normalize_tool_name(tool.name))
                except Exception as exc:
                    errors.append(
                        error_to_payload(
                            exc,
                            provider_name=provider.name,
                            tool_name=getattr(tool, "name", None),
                        )
                    )
                    if not continue_on_error:
                        raise

        return ToolDiscoveryReport(
            provider_names=tuple(provider_names),
            discovered_tools=tuple(discovered_specs),
            registered_names=tuple(registered_names),
            skipped_names=tuple(skipped_names),
            errors=tuple(errors),
            metadata={"provider_count": len(provider_names)},
        )


def discover_tools_from_module(
    module: ModuleType,
    *,
    include_instances: bool = True,
    include_classes: bool = True,
    include_imported_classes: bool = False,
) -> List[BaseTool]:
    """从模块对象中发现 BaseTool 实例或无参工具类。"""

    tools: List[BaseTool] = []
    for value in vars(module).values():
        if include_instances and isinstance(value, BaseTool):
            tools.append(value)
            continue

        if not include_classes:
            continue
        if not inspect.isclass(value):
            continue
        if value is BaseTool or not issubclass(value, BaseTool):
            continue
        if not include_imported_classes and value.__module__ != module.__name__:
            continue
        if inspect.isabstract(value):
            continue
        tools.append(instantiate_tool_class(value))

    return dedupe_tools_by_name(tools)


def instantiate_tool_class(tool_cls: Type[BaseTool]) -> BaseTool:
    """实例化无参工具类。"""

    try:
        return tool_cls()
    except TypeError as exc:
        raise HarnessValidationError(
            "Discovered tool class must be constructible without arguments.",
            details={"tool_class": tool_cls.__name__},
            cause=exc,
        ) from exc


def dedupe_tools_by_name(tools: Iterable[BaseTool]) -> List[BaseTool]:
    """按工具名称去重，保留第一次发现的工具。"""

    seen = set()
    result: List[BaseTool] = []
    for tool in tools:
        name = normalize_tool_name(tool.name)
        if name in seen:
            continue
        seen.add(name)
        result.append(tool)
    return result


def error_to_payload(
    error: BaseException,
    *,
    provider_name: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """把发现阶段异常转换为结构化错误。"""

    if isinstance(error, HarnessError):
        payload = error.to_dict()
    else:
        payload = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
    if provider_name is not None:
        payload["provider_name"] = provider_name
    if tool_name is not None:
        payload["tool_name"] = tool_name
    return payload


def create_default_discovery() -> ToolDiscoveryManager:
    """创建默认工具发现管理器。"""

    return ToolDiscoveryManager([BuiltinToolProvider()])


def discover_builtin_tools() -> List[BaseTool]:
    """发现 V2 内置工具。"""

    return BuiltinToolProvider().discover()


__all__ = [
    "BuiltinToolProvider",
    "ModuleToolProvider",
    "StaticToolProvider",
    "ToolDiscoveryManager",
    "ToolDiscoveryReport",
    "ToolProvider",
    "ToolProviderSpec",
    "create_default_discovery",
    "dedupe_tools_by_name",
    "discover_builtin_tools",
    "discover_tools_from_module",
    "error_to_payload",
    "instantiate_tool_class",
]
