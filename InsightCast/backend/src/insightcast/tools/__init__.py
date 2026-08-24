"""myHarness tool adapters."""

from __future__ import annotations

import os
from typing import Mapping, Optional

from tools.mcp import HttpMCPClient, MCPToolProvider

from insightcast.sources.discovery import JsonFetcher
from insightcast.tools.discovery import (
    FETCH_JSON_TOOL_NAME,
    FetchJsonInput,
    FetchJsonTool,
    ToolJsonFetcher,
)
from insightcast.tools.harness import (
    ToolTraceContext,
    TracingToolExecutor,
    current_tool_execution_context,
    run_tool_sync,
    tool_trace_scope,
)
from insightcast.tools.markdown import (
    WRITE_MARKDOWN_BRIEF_TOOL_NAME,
    ToolMarkdownBriefWriter,
    WriteMarkdownBriefInput,
    WriteMarkdownBriefTool,
)
from insightcast.tools.transcript import (
    DEFAULT_LEMONADE_MCP_TOOL_NAME,
    DEFAULT_LEMONADE_TRANSCRIBE_MODEL,
    DOWNLOAD_AUDIO_TOOL_NAME,
    FETCH_PLATFORM_CAPTION_TOOL_NAME,
    LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME,
    DownloadAudioInput,
    DownloadAudioTool,
    FetchPlatformCaptionInput,
    FetchPlatformCaptionTool,
    LemonadeTranscribeAudioInput,
    LemonadeTranscribeAudioTool,
    ToolAudioDownloader,
    ToolAudioTranscriber,
    ToolPlatformCaptionFetcher,
)


def create_insightcast_tool_executor(
    *,
    fetch_json: Optional[JsonFetcher] = None,
    env: Optional[Mapping[str, str]] = None,
) -> TracingToolExecutor:
    """Create the default myHarness ToolExecutor for InsightCast V1."""

    executor = TracingToolExecutor.from_tools(
        (
            FetchJsonTool(fetch_json=fetch_json),
            WriteMarkdownBriefTool(),
            FetchPlatformCaptionTool(env=env),
            DownloadAudioTool(env=env),
        )
    )
    register_lemonade_tools(executor, env=env)
    return executor


def register_lemonade_tools(
    executor: TracingToolExecutor,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Register Lemonade MCP tools when explicitly enabled."""

    source = os.environ if env is None else env
    if not _env_flag(source, "INSIGHTCAST_LEMONADE_MCP_ENABLED", False):
        return
    endpoint = _env_get(
        source,
        "INSIGHTCAST_LEMONADE_MCP_ENDPOINT",
        "http://127.0.0.1:8000/mcp",
    )
    server_name = _env_get(source, "INSIGHTCAST_LEMONADE_MCP_SERVER_NAME", "lemonade")
    client = HttpMCPClient(
        server_name=server_name,
        endpoint=endpoint,
        timeout_seconds=float(_env_get(source, "INSIGHTCAST_LEMONADE_MCP_TIMEOUT", "120")),
        metadata={"provider": "lemonade"},
    )
    for tool in MCPToolProvider(client=client).discover():
        executor.register(tool)
    executor.register(
        LemonadeTranscribeAudioTool(
            executor,
            mcp_tool_name=_env_get(
                source,
                "INSIGHTCAST_LEMONADE_MCP_TRANSCRIBE_TOOL",
                DEFAULT_LEMONADE_MCP_TOOL_NAME,
            ),
            model=_env_get(
                source,
                "INSIGHTCAST_LEMONADE_TRANSCRIBE_MODEL",
                DEFAULT_LEMONADE_TRANSCRIBE_MODEL,
            ),
            response_format=_env_get(
                source,
                "INSIGHTCAST_LEMONADE_RESPONSE_FORMAT",
                "verbose_json",
            ),
            timeout_seconds=float(_env_get(source, "INSIGHTCAST_LEMONADE_TRANSCRIBE_TIMEOUT", "600")),
        )
    )


def _env_get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


__all__ = [
    "DOWNLOAD_AUDIO_TOOL_NAME",
    "FETCH_JSON_TOOL_NAME",
    "FETCH_PLATFORM_CAPTION_TOOL_NAME",
    "LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME",
    "WRITE_MARKDOWN_BRIEF_TOOL_NAME",
    "DEFAULT_LEMONADE_MCP_TOOL_NAME",
    "DEFAULT_LEMONADE_TRANSCRIBE_MODEL",
    "DownloadAudioInput",
    "DownloadAudioTool",
    "FetchJsonInput",
    "FetchJsonTool",
    "FetchPlatformCaptionInput",
    "FetchPlatformCaptionTool",
    "LemonadeTranscribeAudioInput",
    "LemonadeTranscribeAudioTool",
    "ToolAudioDownloader",
    "ToolAudioTranscriber",
    "ToolJsonFetcher",
    "ToolMarkdownBriefWriter",
    "ToolPlatformCaptionFetcher",
    "ToolTraceContext",
    "TracingToolExecutor",
    "WriteMarkdownBriefInput",
    "WriteMarkdownBriefTool",
    "create_insightcast_tool_executor",
    "current_tool_execution_context",
    "register_lemonade_tools",
    "run_tool_sync",
    "tool_trace_scope",
]
