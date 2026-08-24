"""MCP 工具接入框架。

本模块不直接绑定某个 MCP SDK，而是先定义 myHarness 内部需要的最小 MCP 客户端接口。
后续接入官方 MCP SDK、stdio transport、HTTP transport 时，只需要实现 BaseMCPClient。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - compatibility with Pydantic v1
    ConfigDict = None  # type: ignore[assignment]

from core.tool import ToolCall, ToolResult
from infra.exception import ToolExecutionError, ToolValidationError
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from tools.base import BaseTool, ToolExecutionContext, ToolSpec
from tools.discovery import ToolProvider, dedupe_tools_by_name
from tools.registry import normalize_tool_name


class MCPToolArguments(BaseModel):
    """MCP 工具参数容器。

    MCP 工具的输入 schema 来自远端服务，运行时字段并不固定，所以这里允许额外字段。
    """

    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


class MCPToolSpec(SerializableModel):
    """远端 MCP 工具描述。"""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    is_read_only: bool = True
    version: Optional[str] = None
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 MCP 工具描述。"""

        normalize_tool_name(self.name)
        if not isinstance(self.input_schema, dict):
            raise HarnessValidationError(
                "MCPToolSpec input_schema must be a dict.",
                details={"tool_name": self.name},
            )
        invalid_tags = [tag for tag in self.tags if not tag or not tag.strip()]
        if invalid_tags:
            raise HarnessValidationError(
                "MCPToolSpec tags cannot contain empty values.",
                details={"tool_name": self.name, "invalid_tags": invalid_tags},
            )


class MCPToolResult(SerializableModel):
    """远端 MCP 工具调用结果。"""

    output: str = ""
    data: Optional[Any] = None
    is_error: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class BaseMCPClient(ABC):
    """MCP 客户端最小接口。"""

    server_name: str
    metadata: Optional[Dict[str, Any]] = None

    @abstractmethod
    def list_tools(self) -> List[MCPToolSpec]:
        """列出远端 MCP 服务暴露的工具。"""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolResult:
        """调用远端 MCP 工具。"""


class InMemoryMCPClient(BaseMCPClient):
    """用于测试和本地开发的内存 MCP 客户端。"""

    def __init__(
        self,
        *,
        server_name: str,
        tools: Iterable[MCPToolSpec],
        handlers: Optional[Mapping[str, Callable[[Dict[str, Any]], Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.server_name = normalize_mcp_server_name(server_name)
        self._tools = {normalize_tool_name(tool.name): tool for tool in tools}
        self._handlers = {
            normalize_tool_name(name): handler
            for name, handler in dict(handlers or {}).items()
        }
        self.metadata = metadata or {}

    def list_tools(self) -> List[MCPToolSpec]:
        """列出内存中注册的 MCP 工具。"""

        return list(self._tools.values())

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolResult:
        """调用内存 handler，模拟远端 MCP 工具执行。"""

        del context
        normalized = normalize_tool_name(tool_name)
        if normalized not in self._tools:
            raise ToolExecutionError(
                "MCP tool not found.",
                details={"server_name": self.server_name, "mcp_tool_name": normalized},
            )

        handler = self._handlers.get(normalized)
        if handler is None:
            raise ToolExecutionError(
                "MCP tool handler not found.",
                details={"server_name": self.server_name, "mcp_tool_name": normalized},
            )

        result = handler(dict(arguments))
        if inspect.isawaitable(result):
            result = await result
        return normalize_mcp_tool_result(result)


class HttpMCPClient(BaseMCPClient):
    """通过 HTTP JSON-RPC 调用远端 MCP 服务的客户端。"""

    def __init__(
        self,
        *,
        server_name: str,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = 60.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.server_name = normalize_mcp_server_name(server_name)
        self.endpoint = endpoint.strip()
        if not self.endpoint:
            raise HarnessValidationError("HttpMCPClient endpoint cannot be empty.")
        self.headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.metadata = metadata or {}
        self._request_id = 0

    def list_tools(self) -> List[MCPToolSpec]:
        """列出远端 MCP 服务暴露的工具。"""

        response = self._request_sync("tools/list", {})
        tools = response.get("tools") if isinstance(response, Mapping) else None
        if not isinstance(tools, list):
            raise ToolExecutionError(
                "MCP tools/list response missing tools.",
                details={"server_name": self.server_name},
            )
        return [mcp_tool_spec_from_payload(tool) for tool in tools]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> MCPToolResult:
        """调用远端 MCP 工具。"""

        del context
        response = await asyncio.to_thread(
            self._request_sync,
            "tools/call",
            {
                "name": tool_name,
                "arguments": dict(arguments),
            },
        )
        return mcp_tool_result_from_call_response(response)

    def _request_sync(self, method: str, params: Mapping[str, Any]) -> Any:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": dict(params),
        }
        raw = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        request = Request(
            self.endpoint,
            data=raw,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
        message = parse_mcp_http_response(body, content_type=content_type)
        if not isinstance(message, Mapping):
            raise ToolExecutionError(
                "MCP HTTP response root must be an object.",
                details={"server_name": self.server_name},
            )
        if message.get("error"):
            raise ToolExecutionError(
                "MCP JSON-RPC returned an error.",
                details={
                    "server_name": self.server_name,
                    "method": method,
                    "error": message.get("error"),
                },
            )
        return message.get("result")


class MCPToolAdapter(BaseTool):
    """把远端 MCP 工具适配成 myHarness 标准工具。"""

    def __init__(
        self,
        *,
        client: BaseMCPClient,
        mcp_tool: MCPToolSpec,
        namespace_tools: bool = True,
        namespace_separator: str = ".",
    ) -> None:
        self.client = client
        self.mcp_tool = mcp_tool
        self.remote_tool_name = normalize_tool_name(mcp_tool.name)
        self.name = build_mcp_tool_name(
            client,
            mcp_tool,
            namespace_tools=namespace_tools,
            namespace_separator=namespace_separator,
        )
        self.description = build_mcp_tool_description(client, mcp_tool)
        self.input_model = MCPToolArguments
        self.is_read_only = mcp_tool.is_read_only
        self.source = "mcp"
        self.version = mcp_tool.version
        self.tags = merge_mcp_tags(("mcp", client.server_name), mcp_tool.tags)
        self.metadata = build_mcp_tool_metadata(client, mcp_tool)

    def spec(self) -> ToolSpec:
        """返回 MCP 工具能力描述。"""

        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=dict(self.mcp_tool.input_schema),
            is_read_only=self.is_read_only,
            source=self.source,
            version=self.version,
            tags=tuple(self.tags),
            metadata=dict(self.metadata or {}),
        )

    def validate_arguments(self, arguments: Dict[str, Any]) -> BaseModel:
        """按 MCP input_schema 的基础子集校验参数。"""

        validate_mcp_arguments(
            self.mcp_tool.input_schema,
            arguments,
            tool_name=self.name,
        )
        return MCPToolArguments(**arguments)

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """调用远端 MCP 工具并转换成标准 ToolResult。"""

        argument_payload = model_to_arguments(arguments)
        try:
            mcp_result = await self.client.call_tool(
                self.remote_tool_name,
                argument_payload,
                context,
            )
        except ToolValidationError:
            raise
        except ToolExecutionError as exc:
            raise ToolExecutionError(
                "MCP tool execution failed.",
                details={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "mcp_server_name": self.client.server_name,
                    "mcp_tool_name": self.remote_tool_name,
                },
                retryable=exc.retryable,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                "MCP tool execution failed.",
                details={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "mcp_server_name": self.client.server_name,
                    "mcp_tool_name": self.remote_tool_name,
                },
                retryable=True,
                cause=exc,
            ) from exc

        metadata = build_mcp_result_metadata(self.client, self.mcp_tool, mcp_result)
        if mcp_result.is_error:
            failure = ToolResult.failure(
                tool_call=tool_call,
                error=ToolExecutionError(
                    "MCP tool returned an error result.",
                    details={
                        "mcp_server_name": self.client.server_name,
                        "mcp_tool_name": self.remote_tool_name,
                        "mcp_result": mcp_result.to_dict(exclude_none=True),
                    },
                    retryable=bool(mcp_result.metadata.get("retryable", False)),
                ),
                output=mcp_result.output,
                metadata=metadata,
            )
            return failure.clone(data=mcp_result.data)

        return ToolResult.success(
            tool_call=tool_call,
            output=mcp_result.output,
            data=mcp_result.data,
            metadata=metadata,
        )


class MCPToolProvider(ToolProvider):
    """从 MCP 客户端发现工具的 Provider。"""

    def __init__(
        self,
        client: BaseMCPClient,
        *,
        namespace_tools: bool = True,
        namespace_separator: str = ".",
        name: Optional[str] = None,
        source: str = "mcp",
        description: str = "MCP tool provider.",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.namespace_tools = namespace_tools
        self.namespace_separator = namespace_separator
        self.name = name or f"mcp:{client.server_name}"
        self.source = source
        self.description = description
        provider_metadata = dict(metadata or {})
        provider_metadata["server_name"] = client.server_name
        provider_metadata.update(client.metadata or {})
        self.metadata = provider_metadata
        self.spec()

    def discover(self) -> List[BaseTool]:
        """从 MCP 客户端发现并适配工具。"""

        tools: List[BaseTool] = []
        for mcp_tool in self.client.list_tools():
            tools.append(
                MCPToolAdapter(
                    client=self.client,
                    mcp_tool=mcp_tool,
                    namespace_tools=self.namespace_tools,
                    namespace_separator=self.namespace_separator,
                )
            )
        return dedupe_tools_by_name(tools)


def normalize_mcp_server_name(server_name: str) -> str:
    """统一 MCP 服务名称。"""

    if not server_name or not server_name.strip():
        raise HarnessValidationError("MCP server name cannot be empty.")
    return server_name.strip()


def build_mcp_tool_name(
    client: BaseMCPClient,
    mcp_tool: MCPToolSpec,
    *,
    namespace_tools: bool = True,
    namespace_separator: str = ".",
) -> str:
    """构建 MCP 工具注册名称。"""

    remote_name = normalize_tool_name(mcp_tool.name)
    if not namespace_tools:
        return remote_name
    if not namespace_separator:
        raise HarnessValidationError("MCP namespace separator cannot be empty.")
    return f"{normalize_mcp_server_name(client.server_name)}{namespace_separator}{remote_name}"


def build_mcp_tool_description(client: BaseMCPClient, mcp_tool: MCPToolSpec) -> str:
    """构建 MCP 工具描述。"""

    description = mcp_tool.description.strip() if mcp_tool.description else ""
    if description:
        return f"[MCP:{client.server_name}] {description}"
    return f"[MCP:{client.server_name}] {mcp_tool.name}"


def build_mcp_tool_metadata(
    client: BaseMCPClient,
    mcp_tool: MCPToolSpec,
) -> Dict[str, Any]:
    """构建 MCP 工具元数据。"""

    metadata = dict(mcp_tool.metadata)
    metadata.update(
        {
            "mcp_server_name": client.server_name,
            "mcp_tool_name": mcp_tool.name,
        }
    )
    return metadata


def build_mcp_result_metadata(
    client: BaseMCPClient,
    mcp_tool: MCPToolSpec,
    result: MCPToolResult,
) -> Dict[str, Any]:
    """构建 MCP 调用结果元数据。"""

    metadata = dict(result.metadata)
    metadata.update(
        {
            "mcp_server_name": client.server_name,
            "mcp_tool_name": mcp_tool.name,
        }
    )
    return metadata


def validate_mcp_arguments(
    input_schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    tool_name: str,
) -> None:
    """基于 JSON Schema 的基础子集校验 MCP 工具参数。"""

    if not input_schema:
        return
    schema_type = input_schema.get("type")
    if schema_type not in (None, "object"):
        raise ToolValidationError(
            "MCP tool input_schema root type must be object.",
            details={"tool_name": tool_name, "schema_type": schema_type},
        )

    required_fields = input_schema.get("required", ())
    if not isinstance(required_fields, (list, tuple)):
        raise ToolValidationError(
            "MCP tool input_schema required must be a list.",
            details={"tool_name": tool_name},
        )
    for field_name in required_fields:
        if field_name not in arguments:
            raise ToolValidationError(
                "MCP tool required argument is missing.",
                details={"tool_name": tool_name, "argument_name": field_name},
            )

    properties = input_schema.get("properties", {})
    if properties is None:
        return
    if not isinstance(properties, Mapping):
        raise ToolValidationError(
            "MCP tool input_schema properties must be a mapping.",
            details={"tool_name": tool_name},
        )
    for field_name, field_schema in properties.items():
        if field_name not in arguments:
            continue
        if not isinstance(field_schema, Mapping):
            continue
        expected_type = field_schema.get("type")
        if expected_type is None:
            continue
        if not is_json_schema_type(arguments[field_name], expected_type):
            raise ToolValidationError(
                "MCP tool argument type mismatch.",
                details={
                    "tool_name": tool_name,
                    "argument_name": field_name,
                    "expected_type": expected_type,
                    "actual_type": arguments[field_name].__class__.__name__,
                },
            )


def is_json_schema_type(value: Any, expected_type: Any) -> bool:
    """判断值是否符合常见 JSON Schema type。"""

    if isinstance(expected_type, (list, tuple)):
        return any(is_json_schema_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True


def model_to_arguments(arguments: BaseModel) -> Dict[str, Any]:
    """把 MCPToolArguments 转换成普通参数字典。"""

    if hasattr(arguments, "model_dump"):
        return dict(arguments.model_dump())
    return dict(arguments.dict())


def normalize_mcp_tool_result(value: Any) -> MCPToolResult:
    """把不同形式的 MCP 返回值转换成 MCPToolResult。"""

    if isinstance(value, MCPToolResult):
        return value
    if isinstance(value, str):
        return MCPToolResult(output=value)
    if isinstance(value, Mapping):
        if any(key in value for key in ("output", "data", "is_error", "metadata")):
            return MCPToolResult(
                output=str(value.get("output", "")),
                data=value.get("data"),
                is_error=bool(value.get("is_error", False)),
                metadata=dict(value.get("metadata") or {}),
            )
        return MCPToolResult(output=format_mcp_output(value), data=dict(value))
    if isinstance(value, list):
        return MCPToolResult(output=format_mcp_output(value), data=value)
    return MCPToolResult(output=str(value), data=value)


def mcp_tool_spec_from_payload(payload: Any) -> MCPToolSpec:
    """把 MCP tools/list 返回项转换成 MCPToolSpec。"""

    if not isinstance(payload, Mapping):
        raise ToolExecutionError(
            "MCP tool spec payload must be an object.",
            details={"payload_type": type(payload).__name__},
        )
    input_schema = (
        payload.get("inputSchema")
        or payload.get("input_schema")
        or payload.get("schema")
        or {}
    )
    annotations = payload.get("annotations") or {}
    is_read_only = True
    if isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is False:
        is_read_only = False
    return MCPToolSpec(
        name=str(payload.get("name") or ""),
        description=str(payload.get("description") or ""),
        input_schema=dict(input_schema) if isinstance(input_schema, Mapping) else {},
        is_read_only=is_read_only,
        metadata={"raw_mcp_tool": dict(payload)},
    )


def mcp_tool_result_from_call_response(response: Any) -> MCPToolResult:
    """把 MCP tools/call result 转换成 MCPToolResult。"""

    if not isinstance(response, Mapping):
        return normalize_mcp_tool_result(response)
    is_error = bool(response.get("isError") or response.get("is_error"))
    content = response.get("content")
    if content is not None:
        output = format_mcp_output(content)
        data = {
            key: value
            for key, value in response.items()
            if key not in ("isError", "is_error")
        }
        return MCPToolResult(
            output=output,
            data=data,
            is_error=is_error,
            metadata={"response_format": "mcp_content"},
        )
    result = response.get("result", response)
    converted = normalize_mcp_tool_result(result)
    return converted.clone(is_error=is_error or converted.is_error)


def parse_mcp_http_response(body: str, *, content_type: str = "") -> Any:
    """解析 HTTP MCP 的 JSON 或 SSE 响应。"""

    text = body.strip()
    if not text:
        raise ToolExecutionError("MCP HTTP response body is empty.")
    if "text/event-stream" in content_type.lower() or text.startswith("event:"):
        return parse_mcp_sse_response(text)
    return json.loads(text)


def parse_mcp_sse_response(text: str) -> Any:
    """解析简单 SSE 响应，取最后一条 data 作为 JSON-RPC message。"""

    data_lines: List[str] = []
    messages: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if data_lines:
                messages.append("\n".join(data_lines))
                data_lines = []
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
    if data_lines:
        messages.append("\n".join(data_lines))
    if not messages:
        raise ToolExecutionError("MCP SSE response did not contain data lines.")
    return json.loads(messages[-1])


def format_mcp_output(value: Any) -> str:
    """把 MCP content 或普通值格式化成给 LLM 的文本。"""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if value.get("type") == "text" and "text" in value:
            return str(value["text"])
        if "output" in value:
            return str(value["output"])
        return str(dict(value))
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = format_mcp_output(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(value)


def merge_mcp_tags(*tag_groups: Iterable[str]) -> Tuple[str, ...]:
    """合并并去重 MCP 工具标签。"""

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
    "BaseMCPClient",
    "HttpMCPClient",
    "InMemoryMCPClient",
    "MCPToolAdapter",
    "MCPToolArguments",
    "MCPToolProvider",
    "MCPToolResult",
    "MCPToolSpec",
    "build_mcp_result_metadata",
    "build_mcp_tool_description",
    "build_mcp_tool_metadata",
    "build_mcp_tool_name",
    "format_mcp_output",
    "is_json_schema_type",
    "merge_mcp_tags",
    "mcp_tool_result_from_call_response",
    "mcp_tool_spec_from_payload",
    "model_to_arguments",
    "normalize_mcp_server_name",
    "normalize_mcp_tool_result",
    "parse_mcp_http_response",
    "parse_mcp_sse_response",
    "validate_mcp_arguments",
]
