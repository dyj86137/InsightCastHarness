"""基于 ReAct 的 Markdown 日报撰写与校验 Agent。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from context.builder import ContextBuilder
from core.message import Message
from core.state import AgentState
from core.tool import ToolCall, ToolResult
from infra.config import ContextConfig, LLMConfig, ToolConfig
from llm import BaseLLM
from tools.base import BaseTool, ToolExecutionContext
from tools.executor import ToolExecutor

from insightcast.agent.llm_trace import InsightCastReactLoop
from insightcast.briefs import (
    MarkdownBriefInputItem,
    build_markdown_brief_prompt,
)
from insightcast.domain.enums import BriefSection
from insightcast.domain.models import DailyBrief, DomainModel
from insightcast.harness_events import (
    HarnessLoopContext,
    current_harness_event_recorder,
)
from insightcast.tools import WRITE_MARKDOWN_BRIEF_TOOL_NAME, WriteMarkdownBriefTool


VALIDATE_MARKDOWN_BRIEF_TOOL_NAME = "insightcast_validate_markdown_brief"


BRIEF_WRITING_REACT_SYSTEM_PROMPT = """你是 InsightCast 的 Markdown 日报撰写 Agent。

你必须严格使用 ReAct 格式。

可用工具：
- insightcast_validate_markdown_brief：在写入文件前校验 Markdown 草稿。
- insightcast_write_markdown_brief：写入最终 Markdown 文件。

必须遵循的流程：
1. 基于输入的结构化条目起草 Markdown 日报。
2. 使用完整 Markdown 草稿调用 insightcast_validate_markdown_brief。
3. 如果校验结果包含问题，先修改草稿，然后再次校验。
4. 只有校验通过后，才能调用 insightcast_write_markdown_brief。
5. 最后用 Final Answer 返回已写入文件的 markdown_path。

写作规则：
- 输出简洁的中文商业情报日报。
- 使用清晰的 Markdown 标题层级。
- 优先呈现 Must Read 和 Worth Watching 内容。
- 避免重复观点和模板化废话。
- 不要编造转录稿中不存在的细节。
- 不要输出原文证据片段、segment_id、审计字段或单独的可引用短句栏目；证据只保存在独立审计文件中。
"""


class BriefWritingAgentResult(DomainModel):
    """ReAct 日报撰写 Agent 返回的结果。"""

    markdown_path: str
    final_answer: str = ""


class ValidateMarkdownBriefInput(BaseModel):
    """Markdown 日报草稿校验参数。"""

    markdown: str


class ValidateMarkdownBriefTool(BaseTool):
    """校验生成的 Markdown 日报草稿。"""

    name = VALIDATE_MARKDOWN_BRIEF_TOOL_NAME
    description = "在写入前校验 InsightCast Markdown 日报草稿。"
    input_model = ValidateMarkdownBriefInput
    is_read_only = True
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "brief", "validation")
    metadata = {"component": "markdown_brief"}

    def __init__(
        self,
        *,
        brief: DailyBrief,
        items: Sequence[MarkdownBriefInputItem],
    ) -> None:
        self.brief = brief
        self.items = tuple(items)

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = _coerce_validate_input(arguments)
        report = validate_markdown_brief(
            args.markdown,
            brief=self.brief,
            items=self.items,
        )
        if report["passed"]:
            output = "校验通过。"
        else:
            output = "校验失败：" + "；".join(report["issues"])
        if report["warnings"]:
            output += " 警告：" + "；".join(report["warnings"])
        return ToolResult.success(
            tool_call=tool_call,
            output=output,
            data=report,
            metadata={
                "passed": report["passed"],
                "issue_count": len(report["issues"]),
                "warning_count": len(report["warnings"]),
            },
        )


class BriefWritingAgent:
    """使用 myHarness ReactLoop 起草、校验、修改并写入日报。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        timeout_seconds: Optional[float] = None,
        max_iterations: int = 8,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.env = env

    async def write_brief_async(
        self,
        *,
        brief: DailyBrief,
        items: Sequence[MarkdownBriefInputItem],
        output_dir: Optional[Path],
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> BriefWritingAgentResult:
        recorder = current_harness_event_recorder()
        active_run_id = (
            recorder.run_id
            if recorder is not None and recorder.run_id is not None
            else run_id
        )
        active_session_id = (
            recorder.session_id
            if recorder is not None and recorder.session_id is not None
            else session_id
        )
        start_step = recorder.step if recorder is not None and recorder.step is not None else 0

        loop = InsightCastReactLoop(
            llm=self.llm,
            tool_executor=self._create_tool_executor(brief=brief, items=items, output_dir=output_dir),
            context_builder=ContextBuilder(
                ContextConfig(max_messages=30, max_input_tokens=12000),
                system_message=Message.system(BRIEF_WRITING_REACT_SYSTEM_PROMPT),
            ),
            llm_config=LLMConfig(
                provider=self.llm.provider_name,
                model=self.model or self.llm.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds or 60.0,
            ),
            trace_metadata={
                "stage": "markdown_brief",
                "task": "insightcast_brief_writing_agent",
                "agent": "brief_writing_agent",
                "brief_id": brief.id,
                "brief_date": brief.brief_date.isoformat(),
            },
            trace_env=self.env,
        )
        state = AgentState(
            run_id=active_run_id or brief.id,
            session_id=active_session_id,
            step=start_step,
            max_steps=start_step + self.max_iterations,
            messages=(
                Message.user(
                    build_brief_writing_agent_prompt(
                        brief=brief,
                        items=items,
                        output_dir=output_dir,
                    )
                ),
            ),
            metadata={
                "stage": "markdown_brief",
                "agent": "brief_writing_agent",
                "brief_id": brief.id,
                "brief_date": brief.brief_date.isoformat(),
            },
        ).mark_running()
        final_state = await loop.run(state, HarnessLoopContext(recorder))
        markdown_path = _extract_markdown_path(final_state)
        if markdown_path is None:
            raise RuntimeError("BriefWritingAgent finished without writing a Markdown file.")
        if not Path(markdown_path).exists():
            raise RuntimeError(f"BriefWritingAgent returned missing Markdown file: {markdown_path}")
        return BriefWritingAgentResult(
            markdown_path=markdown_path,
            final_answer=final_state.final_output or "",
        )

    def _create_tool_executor(
        self,
        *,
        brief: DailyBrief,
        items: Sequence[MarkdownBriefInputItem],
        output_dir: Optional[Path],
    ) -> ToolExecutor:
        del output_dir
        return ToolExecutor.from_tools(
            (
                ValidateMarkdownBriefTool(brief=brief, items=items),
                WriteMarkdownBriefTool(),
            ),
            config=ToolConfig(allow_mutating_tools=True),
        )


def build_brief_writing_agent_prompt(
    *,
    brief: DailyBrief,
    items: Sequence[MarkdownBriefInputItem],
    output_dir: Optional[Path],
) -> str:
    return (
        "撰写今天的 InsightCast Markdown 日报。\n"
        f"brief_date: {brief.brief_date.isoformat()}\n"
        f"output_dir: {str(output_dir) if output_dir is not None else ''}\n\n"
        "调用写入工具时，必须在 Action Input 中包含 brief_date 和 output_dir。\n\n"
        f"{build_markdown_brief_prompt(brief, items)}"
    )


def validate_markdown_brief(
    markdown: str,
    *,
    brief: DailyBrief,
    items: Sequence[MarkdownBriefInputItem],
) -> Dict[str, Any]:
    text = markdown.strip()
    issues = []
    warnings = []
    if not text:
        issues.append("Markdown 为空")
    if not text.startswith("#"):
        issues.append("Markdown 必须以标题开头")
    if "Must Read" not in text and _count_items(items, BriefSection.MUST_READ):
        issues.append("缺少 Must Read 板块")
    if "Worth Watching" not in text and _count_items(items, BriefSection.WORTH_WATCHING):
        warnings.append("缺少 Worth Watching 板块")

    pushed_items = [
        item
        for item in items
        if item.brief_item.section != BriefSection.ARCHIVED
    ]
    if pushed_items and not any(str(item.interview.url) in text for item in pushed_items):
        issues.append("没有找到任何已推送条目的来源 URL")

    duplicate_lines = _duplicate_markdown_lines(text)
    if duplicate_lines:
        warnings.append(
            "可能存在重复观点："
            + ", ".join(duplicate_lines[:3])
        )

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "brief_id": brief.id,
        "brief_date": brief.brief_date.isoformat(),
        "item_count": len(items),
    }


def _coerce_validate_input(arguments: BaseModel) -> ValidateMarkdownBriefInput:
    if isinstance(arguments, ValidateMarkdownBriefInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return ValidateMarkdownBriefInput(**arguments.model_dump())
    return ValidateMarkdownBriefInput(**arguments.dict())


def _count_items(
    items: Sequence[MarkdownBriefInputItem],
    section: BriefSection,
) -> int:
    return len([item for item in items if item.brief_item.section == section])


def _duplicate_markdown_lines(markdown: str) -> Tuple[str, ...]:
    seen = set()
    duplicates = []
    for line in markdown.splitlines():
        normalized = _normalize_markdown_line(line)
        if not normalized:
            continue
        if normalized in seen:
            duplicates.append(normalized)
            continue
        seen.add(normalized)
    return tuple(duplicates)


def _normalize_markdown_line(line: str) -> str:
    text = line.strip().lower()
    if not text.startswith(("-", "*")):
        return ""
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return text.strip()


def _extract_markdown_path(state: AgentState) -> Optional[str]:
    pattern = re.compile(r"Wrote Markdown brief:\s*(.+)", flags=re.IGNORECASE)
    for message in reversed(state.messages):
        if message.name != WRITE_MARKDOWN_BRIEF_TOOL_NAME:
            continue
        match = pattern.search(message.content)
        if match:
            return match.group(1).strip()
    final_output = state.final_output or ""
    match = re.search(r"markdown_path\s*[:=]\s*(.+)", final_output, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


__all__ = [
    "BRIEF_WRITING_REACT_SYSTEM_PROMPT",
    "BriefWritingAgent",
    "BriefWritingAgentResult",
    "VALIDATE_MARKDOWN_BRIEF_TOOL_NAME",
    "ValidateMarkdownBriefInput",
    "ValidateMarkdownBriefTool",
    "build_brief_writing_agent_prompt",
    "validate_markdown_brief",
]
