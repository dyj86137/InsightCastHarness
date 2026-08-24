"""Daily brief generation."""
"""Daily brief generation and local push helpers."""

from insightcast.briefs.markdown import (
    DirectMarkdownBriefGenerator,
    LLMMarkdownBriefGenerator,
    MARKDOWN_BRIEF_SYSTEM_PROMPT,
    MarkdownBriefGenerator,
    MarkdownBriefInputItem,
    build_markdown_brief_prompt,
    create_brief_llm_from_env,
    default_markdown_brief_dir,
    default_markdown_brief_filename,
    default_markdown_brief_filename_for_date,
    render_markdown_brief,
    write_markdown_brief,
    write_markdown_file,
)


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
    "render_markdown_brief",
    "write_markdown_brief",
    "write_markdown_file",
]
