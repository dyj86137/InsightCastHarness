"""InsightCast 发现 IO 的 myHarness 工具。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode

from core.tool import ToolCall, ToolResult
from pydantic import BaseModel, Field
from tools.base import BaseTool, ToolExecutionContext

from insightcast.sources.discovery import JsonFetcher, fetch_json_url
from insightcast.tools.harness import (
    TracingToolExecutor,
    current_tool_execution_context,
    run_tool_sync,
)


FETCH_JSON_TOOL_NAME = "insightcast_fetch_json"


class FetchJsonInput(BaseModel):
    """获取 JSON payload 的参数。"""

    url: str
    params: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: Optional[float] = None


class FetchJsonTool(BaseTool):
    """通过现有 InsightCast 发现 fetcher 获取 JSON。"""

    name = FETCH_JSON_TOOL_NAME
    description = "为 InsightCast 发现来源获取 JSON payload。"
    input_model = FetchJsonInput
    is_read_only = True
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "discovery", "network")
    metadata = {"component": "discovery"}

    def __init__(self, fetch_json: Optional[JsonFetcher] = None) -> None:
        self.fetch_json = fetch_json or fetch_json_url

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = _coerce_fetch_json_input(arguments)
        timeout_seconds = (
            args.timeout_seconds
            if args.timeout_seconds is not None
            else context.timeout_seconds
        )
        payload = await asyncio.to_thread(
            _call_fetch_json,
            self.fetch_json,
            args.url,
            args.params,
            args.headers,
            timeout_seconds,
        )
        return ToolResult.success(
            tool_call=tool_call,
            output="已获取 JSON payload。",
            data=payload,
            metadata={
                "url": args.url,
                "request_url": _request_url_for_trace(args.url, args.params),
                "param_keys": sorted(str(key) for key in args.params),
            },
        )


class ToolJsonFetcher:
    """由 myHarness ToolExecutor 支撑的同步 JsonFetcher 适配器。"""

    def __init__(
        self,
        executor: TracingToolExecutor,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executor = executor
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        tool_call = ToolCall.create(
            name=FETCH_JSON_TOOL_NAME,
            arguments={
                "url": url,
                "params": dict(params),
                "headers": dict(headers),
                "timeout_seconds": self.timeout_seconds,
            },
            metadata={"adapter": "json_fetcher"},
        )
        context = current_tool_execution_context(
            allow_mutation=False,
            timeout_seconds=self.timeout_seconds,
            metadata={"tool_kind": "json_fetcher"},
        )
        result = run_tool_sync(self.executor.execute(tool_call, context))
        if result.is_error:
            raise RuntimeError(_tool_error_message(result))
        if not isinstance(result.data, Mapping):
            raise RuntimeError("Fetch JSON tool returned non-mapping data")
        return result.data


def _coerce_fetch_json_input(arguments: BaseModel) -> FetchJsonInput:
    if isinstance(arguments, FetchJsonInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return FetchJsonInput(**arguments.model_dump())
    return FetchJsonInput(**arguments.dict())


def _call_fetch_json(
    fetch_json: JsonFetcher,
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: Optional[float],
) -> Mapping[str, Any]:
    if timeout_seconds is not None and _accepts_timeout_seconds(fetch_json):
        return fetch_json(
            url,
            params,
            headers,
            timeout_seconds=timeout_seconds,  # type: ignore[call-arg]
        )
    return fetch_json(url, params, headers)


def _accepts_timeout_seconds(fetch_json: JsonFetcher) -> bool:
    try:
        signature = inspect.signature(fetch_json)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if "timeout_seconds" in parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _request_url_for_trace(url: str, params: Mapping[str, Any]) -> str:
    query = urlencode(
        {
            str(key): _redact_trace_param(str(key), value)
            for key, value in params.items()
            if value is not None
        },
        doseq=True,
    )
    return f"{url}?{query}" if query else url


def _redact_trace_param(key: str, value: Any) -> Any:
    lowered = key.strip().lower().replace("-", "_")
    sensitive_keys = {
        "key",
        "api_key",
        "apikey",
        "access_key",
        "secret",
        "client_secret",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "cookie",
        "authorization",
    }
    if lowered in sensitive_keys or lowered.endswith("_token") or lowered.endswith("_secret"):
        return "[redacted]"
    return value


def _tool_error_message(result: ToolResult) -> str:
    error = result.error or {}
    return str(error.get("message") or result.output or "Tool call failed")


__all__ = [
    "FETCH_JSON_TOOL_NAME",
    "FetchJsonInput",
    "FetchJsonTool",
    "ToolJsonFetcher",
]
