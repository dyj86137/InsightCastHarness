"""InsightCast Markdown 输出的 myHarness 工具。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from core.tool import ToolCall, ToolResult
from pydantic import BaseModel
from tools.base import BaseTool, ToolExecutionContext

from insightcast.domain.models import DailyBrief
from insightcast.tools.harness import (
    TracingToolExecutor,
    current_tool_execution_context,
    run_tool_sync,
)


WRITE_MARKDOWN_BRIEF_TOOL_NAME = "insightcast_write_markdown_brief"


class WriteMarkdownBriefInput(BaseModel):
    """将生成的 Markdown 日报写入磁盘的参数。"""

    markdown: str
    brief_date: date
    output_dir: Optional[str] = None
    filename: Optional[str] = None


class WriteMarkdownBriefTool(BaseTool):
    """将生成的 InsightCast Markdown 日报写入项目工作区。"""

    name = WRITE_MARKDOWN_BRIEF_TOOL_NAME
    description = "将生成的 InsightCast Markdown 日报写入本地文件。"
    input_model = WriteMarkdownBriefInput
    is_read_only = False
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "brief", "filesystem")
    metadata = {"component": "markdown_brief"}

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        from insightcast.briefs.markdown import (
            default_markdown_brief_filename_for_date,
            write_markdown_file,
        )

        args = _coerce_write_markdown_input(arguments)
        output_path = write_markdown_file(
            args.markdown,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            filename=args.filename
            or default_markdown_brief_filename_for_date(args.brief_date),
        )
        return ToolResult.success(
            tool_call=tool_call,
            output=f"Wrote Markdown brief: {output_path}",
            data={"markdown_path": str(output_path)},
            metadata={
                "markdown_path": str(output_path),
                "filename": output_path.name,
            },
        )


class ToolMarkdownBriefWriter:
    """由 myHarness ToolExecutor 支撑的同步 Markdown 写入适配器。"""

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
        markdown: str,
        brief: DailyBrief,
        output_dir: Optional[Path],
    ) -> Path:
        result = run_tool_sync(
            self._execute(markdown, brief, output_dir)
        )
        return self._path_from_result(result)

    async def write_async(
        self,
        markdown: str,
        brief: DailyBrief,
        output_dir: Optional[Path],
    ) -> Path:
        result = await self._execute(markdown, brief, output_dir)
        return self._path_from_result(result)

    async def _execute(
        self,
        markdown: str,
        brief: DailyBrief,
        output_dir: Optional[Path],
    ) -> ToolResult:
        tool_call = ToolCall.create(
            name=WRITE_MARKDOWN_BRIEF_TOOL_NAME,
            arguments={
                "markdown": markdown,
                "brief_date": brief.brief_date.isoformat(),
                "output_dir": str(output_dir) if output_dir is not None else None,
            },
            metadata={"adapter": "markdown_brief_writer", "brief_id": brief.id},
        )
        context = current_tool_execution_context(
            allow_mutation=True,
            timeout_seconds=self.timeout_seconds,
            metadata={"tool_kind": "markdown_brief_writer"},
        )
        return await self.executor.execute(tool_call, context)

    def _path_from_result(self, result: ToolResult) -> Path:
        if result.is_error:
            raise RuntimeError(_tool_error_message(result))
        if not isinstance(result.data, dict) or not result.data.get("markdown_path"):
            raise RuntimeError("Markdown brief tool returned no markdown_path")
        return Path(str(result.data["markdown_path"]))


def _coerce_write_markdown_input(arguments: BaseModel) -> WriteMarkdownBriefInput:
    if isinstance(arguments, WriteMarkdownBriefInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return WriteMarkdownBriefInput(**arguments.model_dump())
    return WriteMarkdownBriefInput(**arguments.dict())


def _tool_error_message(result: ToolResult) -> str:
    error = result.error or {}
    return str(error.get("message") or result.output or "Tool call failed")


__all__ = [
    "WRITE_MARKDOWN_BRIEF_TOOL_NAME",
    "ToolMarkdownBriefWriter",
    "WriteMarkdownBriefInput",
    "WriteMarkdownBriefTool",
]
