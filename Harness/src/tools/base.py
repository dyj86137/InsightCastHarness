"""工具系统基础抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, Field

from core.tool import ToolCall, ToolResult
from infra.exception import ToolValidationError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


class EmptyToolInput(BaseModel):
    """无参数工具的默认输入模型。"""

    pass


class ToolSpec(SerializableModel):
    """工具能力描述。"""

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    is_read_only: bool = False
    source: str = "local"
    version: Optional[str] = None
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验工具描述的基础规则。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError("ToolSpec requires non-empty name.")
        if not self.description or not self.description.strip():
            raise HarnessValidationError(
                "ToolSpec requires non-empty description.",
                details={"tool_name": self.name},
            )
        if not self.source or not self.source.strip():
            raise HarnessValidationError(
                "ToolSpec requires non-empty source.",
                details={"tool_name": self.name},
            )
        invalid_tags = [tag for tag in self.tags if not tag or not tag.strip()]
        if invalid_tags:
            raise HarnessValidationError(
                "ToolSpec tags cannot contain empty values.",
                details={"tool_name": self.name, "invalid_tags": invalid_tags},
            )


class ToolExecutionContext(SerializableModel):
    """工具执行时的共享上下文。"""

    run_id: Optional[str] = None
    session_id: Optional[str] = None
    step: Optional[int] = None
    cwd: Path = Field(default_factory=Path.cwd)
    allow_mutation: bool = False
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验工具执行上下文。"""

        if self.step is not None and self.step < 0:
            raise HarnessValidationError(
                "ToolExecutionContext step cannot be negative.",
                details={"step": self.step},
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise HarnessValidationError(
                "ToolExecutionContext timeout_seconds must be positive.",
                details={"timeout_seconds": self.timeout_seconds},
            )


def _model_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


class BaseTool(ABC):
    """所有工具实现必须遵守的统一接口。"""

    name: str
    description: str
    input_model: Type[BaseModel] = EmptyToolInput
    is_read_only: bool = False
    source: str = "local"
    version: Optional[str] = None
    tags: Tuple[str, ...] = ()
    metadata: Optional[Dict[str, Any]] = None

    def spec(self) -> ToolSpec:
        """返回工具能力描述。"""

        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=_model_json_schema(self.input_model),
            is_read_only=self.is_read_only,
            source=self.source,
            version=self.version,
            tags=tuple(self.tags),
            metadata=dict(self.metadata or {}),
        )

    def validate_arguments(self, arguments: Dict[str, Any]) -> BaseModel:
        """使用工具输入模型校验参数。"""

        try:
            return self.input_model(**arguments)
        except Exception as exc:
            raise ToolValidationError(
                "Tool arguments validation failed.",
                details={
                    "tool_name": self.name,
                    "arguments": arguments,
                },
                cause=exc,
            ) from exc

    async def run(self, tool_call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        """校验参数并执行工具，返回标准 ToolResult。"""

        parsed_arguments = self.validate_arguments(tool_call.arguments)
        return await self.execute(tool_call, parsed_arguments, context)

    @abstractmethod
    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行工具。"""


__all__ = [
    "BaseTool",
    "EmptyToolInput",
    "ToolExecutionContext",
    "ToolSpec",
]
