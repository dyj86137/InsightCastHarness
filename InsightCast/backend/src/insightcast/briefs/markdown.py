"""Markdown daily brief generation and local file output."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from core.message import Message
from infra.config import LLMConfig
from llm import BaseLLM, ChatRequest, create_llm

from insightcast.domain.enums import BriefSection
from insightcast.domain.models import DailyBrief, DailyBriefItem, Interview, InterviewSummary


MARKDOWN_BRIEF_SYSTEM_PROMPT = """你是一名商业情报日报编辑，负责把行业领袖访谈整理成中文 Markdown 日报。

只输出 Markdown 正文，不要输出 JSON、代码块包裹或解释性前言。

写作要求：
- 使用清晰的 Markdown 标题层级。
- 优先展开 Must Read 和 Worth Watching 条目。
- Archived 条目只用于理解当天候选池，除非特别有必要，不展开正文，最多说明归档数量。
- 每条推送内容应包含原始链接、人物/来源、摘要、关注机会、观点变化和 transcript 状态。
- 摘要直接使用完整摘要字段：完整 transcript 的摘要目标为 300～500 个中文字符，最多 600 个字符；partial 或仅有标题/简介的内容可缩短为 150～300 个字符，并明确说明证据不足。不要添加结构化摘要中没有的事实、数字、因果关系或结果判断。
- “关注机会”只写结构化摘要中已有、且有 transcript 事实前提支持的机会方向；允许有限度的业务推导，但不要新增受益人群、行业、产品、收益、替代关系或时间周期，不要把推导写成确定事实。
- 原文证据和声明推导只保存在独立审计文件中，日报正文不要输出证据片段、segment_id 或审计字段。
- 保持商业情报视角，但证据边界优先于表达完整度；可以提炼判断，不得超出结构化摘要和 transcript 证据。
- 不输出完整 transcript，不输出长篇逐字稿。
- 只输出本要求列出的栏目，不添加额外栏目。
- 如果 transcript 只是 partial 或 metadata，请明确降低确定性，不要编造访谈细节。
"""


class MarkdownBriefGenerator(ABC):
    """Generates Markdown content for a DailyBrief."""

    @abstractmethod
    async def generate_async(
        self,
        brief: DailyBrief,
        items: Sequence["MarkdownBriefInputItem"],
    ) -> str:
        """Generate Markdown for a brief and its source items."""


class DirectMarkdownBriefGenerator(MarkdownBriefGenerator):
    """直接渲染结构化摘要，不再次调用 LLM 改写内容。"""

    async def generate_async(
        self,
        brief: DailyBrief,
        items: Sequence["MarkdownBriefInputItem"],
    ) -> str:
        return render_markdown_brief(brief, items)


class MarkdownBriefInputItem:
    """Source data used to render one brief item."""

    def __init__(
        self,
        *,
        brief_item: DailyBriefItem,
        interview: Interview,
        summary: InterviewSummary,
    ) -> None:
        self.brief_item = brief_item
        self.interview = interview
        self.summary = summary


class LLMMarkdownBriefGenerator(MarkdownBriefGenerator):
    """Uses myHarness BaseLLM to generate a Markdown daily brief."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 5000,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        from insightcast.agent.llm_trace import ensure_traced_llm

        self.llm = ensure_traced_llm(llm)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    async def generate_async(
        self,
        brief: DailyBrief,
        items: Sequence[MarkdownBriefInputItem],
    ) -> str:
        request = ChatRequest.create(
            messages=(
                Message.system(MARKDOWN_BRIEF_SYSTEM_PROMPT),
                Message.user(build_markdown_brief_prompt(brief, items)),
            ),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            metadata={
                "task": "insightcast_markdown_daily_brief",
                "brief_id": brief.id,
                "brief_date": brief.brief_date.isoformat(),
            },
        )
        response = await self.llm.chat(request)
        markdown = response.text.strip()
        if not markdown:
            raise ValueError("Markdown brief LLM returned empty content")
        return markdown


def build_markdown_brief_prompt(
    brief: DailyBrief,
    items: Sequence[MarkdownBriefInputItem],
) -> str:
    """Build a compact JSON payload for Markdown brief generation."""

    payload = {
        "brief": {
            "id": brief.id,
            "date": brief.brief_date.isoformat(),
            "title": brief.title,
            "push_channels": [channel.value for channel in brief.push_channels],
            "sections": _section_counts(brief.items),
        },
        "items": [_item_payload(item) for item in items],
    }
    return (
        "请基于下面的结构化访谈摘要生成一份 Markdown 日报。"
        "Must Read 和 Worth Watching 是需要推送的内容；Archived 只做归档上下文。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def render_markdown_brief(
    brief: DailyBrief,
    items: Sequence[MarkdownBriefInputItem],
) -> str:
    """将结构化摘要字段原样写入 Markdown 日报。"""

    section_titles = {
        BriefSection.MUST_READ: "Must Read",
        BriefSection.WORTH_WATCHING: "Worth Watching",
        BriefSection.ARCHIVED: "Archived",
    }
    lines = [f"# {brief.title}", ""]
    for section in (
        BriefSection.MUST_READ,
        BriefSection.WORTH_WATCHING,
        BriefSection.ARCHIVED,
    ):
        lines.extend((f"## {section_titles[section]}", ""))
        section_items = [
            item for item in items if item.brief_item.section == section
        ]
        if not section_items:
            lines.extend(("- 无", ""))
            continue
        for item in section_items:
            lines.extend(_render_markdown_item(item))
    return "\n".join(lines).rstrip()


def _render_markdown_item(item: MarkdownBriefInputItem) -> Tuple[str, ...]:
    interview = item.interview
    summary = item.summary
    person_names = "、".join(interview.person_names) or "未识别人物"
    opportunities = summary.potential_opportunities
    lines = [
        f"### {item.brief_item.rank}. {interview.title}",
        f"- 原始链接：<{interview.url}>",
        f"- 人物 / 来源：{person_names} / {interview.source_name}",
        f"- 摘要：{summary.summary}",
        "- 关注机会：",
    ]
    if opportunities:
        lines.extend(f"  - {opportunity}" for opportunity in opportunities)
    else:
        lines.append("  - 无")
    lines.extend(
        (
            f"- 观点变化：{summary.novelty_assessment or '暂无'}",
            f"- transcript 状态：{interview.transcript_status.value}",
            "",
        )
    )
    return tuple(lines)


def write_markdown_brief(
    markdown: str,
    *,
    brief: DailyBrief,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """Write Markdown brief content to a local project file."""

    return write_markdown_file(
        markdown,
        output_dir=output_dir,
        filename=filename or default_markdown_brief_filename(brief),
    )


def write_markdown_file(
    markdown: str,
    *,
    output_dir: Optional[Path] = None,
    filename: str,
) -> Path:
    """Write Markdown content to a local project file."""

    text = markdown.strip()
    if not text:
        raise ValueError("Markdown brief content cannot be empty")

    root = output_dir or default_markdown_brief_dir()
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / filename
    output_path.write_text(text + "\n", encoding="utf-8")
    return output_path


def default_markdown_brief_dir() -> Path:
    """Return the default local output directory for Markdown briefs."""

    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / "data" / "briefs"


def default_markdown_brief_filename(brief: DailyBrief) -> str:
    """Return a stable Markdown filename for a brief date."""

    return default_markdown_brief_filename_for_date(brief.brief_date)


def default_markdown_brief_filename_for_date(brief_date: date) -> str:
    """Return a stable Markdown filename for a brief date."""

    return f"{brief_date.isoformat()}-insightcast-brief.md"


def create_brief_llm_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "INSIGHTCAST_BRIEF_LLM",
) -> BaseLLM:
    """Create the LLM used for Markdown brief writing from environment variables."""

    source = os.environ if env is None else env
    config = LLMConfig(
        provider=_env_get(source, f"{prefix}_PROVIDER", "openai"),
        model=_env_get(source, f"{prefix}_MODEL", "gpt-4.1-mini"),
        api_key=(
            source.get(f"{prefix}_API_KEY")
            or source.get("OPENAI_API_KEY")
            or source.get("MYHARNESS_LLM_API_KEY")
        ),
        base_url=source.get(f"{prefix}_BASE_URL") or source.get("MYHARNESS_LLM_BASE_URL"),
        timeout_seconds=float(_env_get(source, f"{prefix}_TIMEOUT_SECONDS", "60")),
        temperature=float(_env_get(source, f"{prefix}_TEMPERATURE", "0.2")),
        max_tokens=_optional_int(source.get(f"{prefix}_MAX_TOKENS")),
    )
    return create_llm(config)


def _item_payload(item: MarkdownBriefInputItem) -> Dict[str, Any]:
    interview = item.interview
    summary = item.summary
    brief_item = item.brief_item
    return {
        "section": brief_item.section.value,
        "rank": brief_item.rank,
        "score": brief_item.score,
        "reason": brief_item.reason,
        "interview": {
            "id": interview.id,
            "title": interview.title,
            "url": str(interview.url),
            "source_name": interview.source_name,
            "person_names": list(interview.person_names),
            "industries": [industry.value for industry in interview.industries],
            "published_at": interview.published_at.isoformat()
            if interview.published_at
            else None,
            "duration_seconds": interview.duration_seconds,
            "transcript_status": interview.transcript_status.value,
            "importance_score": interview.importance_score,
            "novelty_score": interview.novelty_score,
            "relevance_score": interview.relevance_score,
        },
        "summary": {
            "id": summary.id,
            "summary": summary.summary,
            "key_points": list(summary.key_points),
            "potential_opportunities": list(summary.potential_opportunities),
            "industry_judgements": list(summary.industry_judgements),
            "mentioned_companies": list(summary.mentioned_companies),
            "mentioned_products": list(summary.mentioned_products),
            "audience": list(summary.audience),
            "novelty_assessment": summary.novelty_assessment,
            "insights": [
                {
                    "statement": insight.statement,
                    "category": insight.category,
                    "confidence": insight.confidence,
                    "mentioned_companies": list(insight.mentioned_companies),
                    "mentioned_products": list(insight.mentioned_products),
                }
                for insight in summary.insights[:5]
            ],
        },
    }


def _section_counts(items: Tuple[DailyBriefItem, ...]) -> Dict[str, int]:
    counts = {section.value: 0 for section in BriefSection}
    for item in items:
        counts[item.section.value] = counts.get(item.section.value, 0) + 1
    return counts


def _env_get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


__all__ = [
    "DirectMarkdownBriefGenerator",
    "LLMMarkdownBriefGenerator",
    "MARKDOWN_BRIEF_SYSTEM_PROMPT",
    "MarkdownBriefGenerator",
    "MarkdownBriefInputItem",
    "build_markdown_brief_prompt",
    "create_brief_llm_from_env",
    "default_markdown_brief_dir",
    "default_markdown_brief_filename",
    "default_markdown_brief_filename_for_date",
    "write_markdown_brief",
    "write_markdown_file",
    "render_markdown_brief",
]
