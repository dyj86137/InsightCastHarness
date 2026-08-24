from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM

from insightcast.agent import BriefWritingAgent, MarkdownBriefRunner
from insightcast.briefs import DirectMarkdownBriefGenerator, LLMMarkdownBriefGenerator
from insightcast.domain.enums import (
    BriefSection,
    BriefStatus,
    ContentFormat,
    Industry,
    InterviewStatus,
    InterviewType,
    TranscriptStatus,
)
from insightcast.domain.models import (
    DailyBrief,
    DailyBriefItem,
    ExtractedInsight,
    Interview,
    InterviewSummary,
)
from insightcast.storage.repositories import InsightCastRepositories


class MarkdownBriefRunnerTests(unittest.TestCase):
    def test_direct_markdown_generator_reuses_structured_summary_fields(self) -> None:
        generator = DirectMarkdownBriefGenerator()
        interview = make_interview(id="interview_direct")
        summary = make_summary(interview_id=interview.id)
        brief_item = make_brief_item(interview, summary, section=BriefSection.MUST_READ)
        brief = DailyBrief(
            brief_date=date(2026, 7, 12),
            title="InsightCast Daily Brief - 2026-07-12",
            items=(brief_item,),
        )

        markdown = asyncio.run(
            generator.generate_async(
                brief,
                [make_input_item(brief_item, interview, summary)],
            )
        )

        self.assertIn(f"- 摘要：{summary.summary}", markdown)
        for opportunity in summary.potential_opportunities:
            self.assertIn(f"  - {opportunity}", markdown)

    def test_llm_markdown_generator_uses_structured_brief_payload(self) -> None:
        llm = FakeLLM(responses=["# InsightCast Daily Brief\n\n## Must Read\n\n- NVIDIA"])
        generator = LLMMarkdownBriefGenerator(llm)
        interview = make_interview(id="interview_prompt")
        summary = make_summary(interview_id=interview.id)
        brief_item = make_brief_item(interview, summary, section=BriefSection.MUST_READ)
        brief = DailyBrief(
            brief_date=date(2026, 7, 12),
            title="InsightCast Daily Brief - 2026-07-12",
            items=(brief_item,),
        )

        markdown = asyncio.run(
            generator.generate_async(
                brief,
                [(make_input_item(brief_item, interview, summary))],
            )
        )

        self.assertIn("# InsightCast Daily Brief", markdown)
        self.assertIn("Jensen Huang interview on AI infrastructure", llm.last_request.messages[1].content)
        self.assertIn('"section": "must_read"', llm.last_request.messages[1].content)
        self.assertIn('"transcript_status": "partial"', llm.last_request.messages[1].content)
        self.assertIn('"summary": {', llm.last_request.messages[1].content)
        self.assertIn('"potential_opportunities"', llm.last_request.messages[1].content)
        self.assertIn("关注机会", llm.last_request.messages[0].content)
        self.assertNotIn('"strategic_signals"', llm.last_request.messages[1].content)
        self.assertNotIn('"notable_quotes"', llm.last_request.messages[1].content)
        self.assertNotIn("可引用短句", llm.last_request.messages[1].content)
        self.assertNotIn("访谈材料强调数据中心需求", llm.last_request.messages[1].content)
        self.assertEqual(llm.last_request.max_tokens, 5000)

    def test_runner_writes_markdown_brief_and_updates_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repositories = InsightCastRepositories(root_dir=root / "storage")
            for index in range(9):
                interview = make_interview(
                    id=f"interview_{index}",
                    importance_score=round(0.95 - index * 0.05, 2),
                )
                repositories.interviews.create(interview)
                repositories.interview_summaries.create(
                    make_summary(interview_id=interview.id, id=f"summary_{index}")
                )

            llm = FakeLLM(
                responses=[
                    "# InsightCast Daily Brief\n\n"
                    "## Must Read\n\n"
                    "- [Jensen Huang interview](https://example.com/interviews/interview_0)\n"
                ]
            )
            runner = MarkdownBriefRunner(
                repositories,
                LLMMarkdownBriefGenerator(llm),
                output_dir=root / "briefs",
                brief_date=date(2026, 7, 12),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.item_count, 9)
            self.assertEqual(result.must_read_count, 3)
            self.assertEqual(result.worth_watching_count, 5)
            self.assertEqual(result.archived_count, 1)
            self.assertEqual(result.pushed_count, 8)
            self.assertTrue(Path(result.markdown_path).exists())
            self.assertIn("Must Read", Path(result.markdown_path).read_text(encoding="utf-8"))

            brief = repositories.daily_briefs.find_by_date(date(2026, 7, 12))
            self.assertIsNotNone(brief)
            self.assertEqual(brief.status, BriefStatus.SENT)
            self.assertEqual(brief.markdown_path, result.markdown_path)
            self.assertEqual(brief.items[0].section, BriefSection.MUST_READ)
            self.assertEqual(brief.items[3].section, BriefSection.WORTH_WATCHING)
            self.assertEqual(brief.items[8].section, BriefSection.ARCHIVED)

            self.assertEqual(
                repositories.interviews.require("interview_0").status,
                InterviewStatus.PUSHED,
            )
            self.assertIsNotNone(repositories.interviews.require("interview_0").pushed_at)
            self.assertEqual(
                repositories.interviews.require("interview_8").status,
                InterviewStatus.ARCHIVED,
            )
            self.assertIsNone(repositories.interviews.require("interview_8").pushed_at)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.metadata["stage"], "markdown_brief")
            self.assertEqual(run.pushed_count, 8)

    def test_runner_can_use_react_brief_writing_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repositories = InsightCastRepositories(root_dir=root / "storage")
            interview = make_interview(id="interview_agent")
            repositories.interviews.create(interview)
            repositories.interview_summaries.create(
                make_summary(interview_id=interview.id, id="summary_agent")
            )
            markdown = (
                "# InsightCast Daily Brief\n\n"
                "## Must Read\n\n"
                "- [Jensen Huang interview](https://example.com/interviews/interview_agent)\n"
            )
            agent_llm = FakeLLM(
                responses=[
                    (
                        "Thought: I should validate the draft before writing.\n"
                        "Action: insightcast_validate_markdown_brief\n"
                        f"Action Input: {{\"markdown\": {json_string(markdown)}}}"
                    ),
                    (
                        "Thought: Validation passed, now write the file.\n"
                        "Action: insightcast_write_markdown_brief\n"
                        "Action Input: {"
                        f"\"markdown\": {json_string(markdown)}, "
                        "\"brief_date\": \"2026-07-12\", "
                        f"\"output_dir\": {json_string(str(root / 'briefs'))}"
                        "}"
                    ),
                    "Final Answer: markdown_path: done",
                ]
            )
            runner = MarkdownBriefRunner(
                repositories,
                LLMMarkdownBriefGenerator(
                    FakeLLM(responses=[RuntimeError("fallback should not run")])
                ),
                output_dir=root / "briefs",
                brief_date=date(2026, 7, 12),
                brief_agent=BriefWritingAgent(agent_llm),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertTrue(Path(result.markdown_path).exists())
            self.assertIn("Must Read", Path(result.markdown_path).read_text(encoding="utf-8"))
            self.assertEqual(len(agent_llm.requests), 3)

    def test_runner_falls_back_when_react_brief_agent_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repositories = InsightCastRepositories(root_dir=root / "storage")
            interview = make_interview(id="interview_fallback")
            repositories.interviews.create(interview)
            repositories.interview_summaries.create(
                make_summary(interview_id=interview.id, id="summary_fallback")
            )
            fallback_markdown = (
                "# InsightCast Daily Brief\n\n"
                "## Must Read\n\n"
                "- [Fallback](https://example.com/interviews/interview_fallback)\n"
            )
            runner = MarkdownBriefRunner(
                repositories,
                LLMMarkdownBriefGenerator(FakeLLM(responses=[fallback_markdown])),
                output_dir=root / "briefs",
                brief_date=date(2026, 7, 12),
                brief_agent=BriefWritingAgent(FakeLLM(responses=["not react format"])),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "partial")
            self.assertEqual(result.errors[0].error_type, "BriefWritingAgentFailed")
            self.assertTrue(Path(result.markdown_path).exists())
            self.assertIn("Fallback", Path(result.markdown_path).read_text(encoding="utf-8"))


def make_input_item(
    brief_item: DailyBriefItem,
    interview: Interview,
    summary: InterviewSummary,
):
    from insightcast.briefs import MarkdownBriefInputItem

    return MarkdownBriefInputItem(
        brief_item=brief_item,
        interview=interview,
        summary=summary,
    )


def make_interview(
    *,
    id: str,
    importance_score: float = 0.9,
) -> Interview:
    return Interview(
        id=id,
        title="Jensen Huang interview on AI infrastructure",
        url=f"https://example.com/interviews/{id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.PUSH_READY,
        source_name="YouTube Search",
        person_names=("Jensen Huang",),
        industries=(Industry.AI,),
        published_at=datetime(2026, 7, 11, 8, 0, 0),
        duration_seconds=3600,
        transcript_status=TranscriptStatus.PARTIAL,
        importance_score=importance_score,
        novelty_score=0.7,
        relevance_score=0.8,
    )


def make_summary(
    *,
    interview_id: str,
    id: str = "summary_prompt",
) -> InterviewSummary:
    return InterviewSummary(
        id=id,
        interview_id=interview_id,
        summary=(
            "Jensen Huang 继续强调 AI 基础设施需求。"
            "他认为企业推理负载会扩大。"
            "数据中心仍是 NVIDIA 的战略重点。"
            "供应链可能继续受益。"
            "该访谈适合关注 AI 投资的人阅读。"
        ),
        key_points=("GPU 需求仍然强劲",),
        potential_opportunities=("NVIDIA 继续强调数据中心扩张",),
        industry_judgements=("AI 基础设施仍处于高景气阶段",),
        mentioned_companies=("NVIDIA",),
        mentioned_products=("GPU",),
        novelty_assessment="暂无历史观点可比，仅基于本次材料判断。",
        insights=(
            ExtractedInsight(
                statement="企业 AI 推理需求正在扩大",
                category="demand",
                confidence=0.8,
                evidence="访谈材料强调数据中心需求",
            ),
        ),
    )


def make_brief_item(
    interview: Interview,
    summary: InterviewSummary,
    *,
    section: BriefSection,
) -> DailyBriefItem:
    return DailyBriefItem(
        interview_id=interview.id,
        summary_id=summary.id,
        section=section,
        rank=1,
        title=interview.title,
        url=interview.url,
        person_names=interview.person_names,
        industries=interview.industries,
        score=interview.importance_score,
    )


def json_string(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
