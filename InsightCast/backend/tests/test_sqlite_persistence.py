from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM

from insightcast.agent import MarkdownBriefRunner
from insightcast.briefs import LLMMarkdownBriefGenerator
from insightcast.domain.enums import (
    CandidateStatus,
    ContentFormat,
    Industry,
    InterviewStatus,
    InterviewType,
    SourceType,
    TranscriptStatus,
)
from insightcast.domain.models import CandidateItem, Interview, InterviewSummary, Person
from insightcast.storage import DuplicateRecordError, SQLiteCollectionStore
from insightcast.storage.repositories import InsightCastRepositories


class SQLitePersistenceTests(unittest.TestCase):
    def test_legacy_summary_fields_are_migrated_on_read(self) -> None:
        summary = InterviewSummary.from_dict(
            {
                "id": "summary_legacy",
                "interview_id": "interview_legacy",
                "five_sentence_summary": "旧摘要内容",
                "strategic_signals": ["旧战略信号"],
                "potential_impact": "旧潜在影响",
                "why_it_matters": "旧重要性说明",
                "notable_quotes": ["旧引用"],
            }
        )

        self.assertEqual(summary.summary, "旧摘要内容")
        self.assertEqual(summary.potential_opportunities, ("旧战略信号",))
        serialized = summary.to_dict()
        self.assertNotIn("five_sentence_summary", serialized)
        self.assertNotIn("strategic_signals", serialized)
        self.assertNotIn("potential_impact", serialized)
        self.assertNotIn("why_it_matters", serialized)
        self.assertNotIn("notable_quotes", serialized)

    def test_sqlite_repositories_persist_records_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "insightcast.sqlite3"
            repositories = InsightCastRepositories(
                root_dir=database_path,
                backend="sqlite",
            )
            repositories.people.create(
                Person(
                    id="person_jensen",
                    name="Jensen Huang",
                    aliases=("黄仁勋",),
                    companies=("NVIDIA",),
                    industries=(Industry.AI,),
                    importance=0.95,
                )
            )
            repositories.candidates.create(make_candidate("candidate_one"))

            reloaded = InsightCastRepositories(
                root_dir=database_path,
                backend="sqlite",
            )

            self.assertEqual(reloaded.people.require("person_jensen").name, "Jensen Huang")
            self.assertIsNotNone(reloaded.people.find_by_name("黄仁勋"))
            self.assertEqual(reloaded.candidates.count(), 1)
            self.assertEqual(
                reloaded.candidates.list_by_status(CandidateStatus.DISCOVERED)[0].id,
                "candidate_one",
            )

    def test_sqlite_store_respects_create_duplicate_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteCollectionStore(
                "test_records",
                database_path=Path(tmp) / "store.sqlite3",
            )
            store.save_record({"id": "record_one", "value": 1}, overwrite=False)

            with self.assertRaises(DuplicateRecordError):
                store.save_record({"id": "record_one", "value": 2}, overwrite=False)

            self.assertEqual(store.require_record("record_one")["value"], 1)

    def test_markdown_push_state_survives_restart_and_prevents_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_path = root / "insightcast.sqlite3"
            repositories = InsightCastRepositories(
                root_dir=database_path,
                backend="sqlite",
            )
            for index in range(2):
                interview = make_interview(f"interview_{index}", 0.9 - index * 0.1)
                repositories.interviews.create(interview)
                repositories.interview_summaries.create(
                    make_summary(interview_id=interview.id, id=f"summary_{index}")
                )

            first_llm = FakeLLM(responses=["# InsightCast Daily Brief\n\n## Must Read\n\n- NVIDIA"])
            first_result = MarkdownBriefRunner(
                repositories,
                LLMMarkdownBriefGenerator(first_llm),
                output_dir=root / "briefs",
                brief_date=date(2026, 7, 12),
            ).run_once()

            reloaded = InsightCastRepositories(
                root_dir=database_path,
                backend="sqlite",
            )
            second_llm = FakeLLM(responses=[RuntimeError("LLM should not be called")])
            second_result = MarkdownBriefRunner(
                reloaded,
                LLMMarkdownBriefGenerator(second_llm),
                output_dir=root / "briefs",
                brief_date=date(2026, 7, 12),
            ).run_once()

            self.assertEqual(first_result.pushed_count, 2)
            self.assertEqual(second_result.item_count, 0)
            self.assertEqual(second_result.pushed_count, 0)
            self.assertEqual(second_llm.requests, [])
            self.assertEqual(
                reloaded.interviews.require("interview_0").status,
                InterviewStatus.PUSHED,
            )
            self.assertIsNotNone(reloaded.daily_briefs.find_by_date(date(2026, 7, 12)))


def make_candidate(candidate_id: str) -> CandidateItem:
    return CandidateItem(
        id=candidate_id,
        source_id="source_youtube",
        source_name="YouTube Search",
        source_type=SourceType.YOUTUBE,
        platform_item_id=candidate_id,
        title="Jensen Huang interview",
        url=f"https://example.com/watch/{candidate_id}",
        format=ContentFormat.VIDEO,
        status=CandidateStatus.DISCOVERED,
    )


def make_interview(interview_id: str, importance_score: float) -> Interview:
    return Interview(
        id=interview_id,
        title="Jensen Huang interview on AI infrastructure",
        url=f"https://example.com/interviews/{interview_id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.PUSH_READY,
        source_name="YouTube Search",
        person_names=("Jensen Huang",),
        industries=(Industry.AI,),
        transcript_status=TranscriptStatus.PARTIAL,
        importance_score=importance_score,
        novelty_score=0.7,
        relevance_score=0.8,
    )


def make_summary(*, interview_id: str, id: str) -> InterviewSummary:
    return InterviewSummary(
        id=id,
        interview_id=interview_id,
        summary=(
            "Jensen Huang 继续强调 AI 基础设施需求。"
            "企业推理负载可能扩大。"
            "数据中心仍是 NVIDIA 的战略重点。"
            "供应链可能继续受益。"
            "该访谈适合关注 AI 投资的人阅读。"
        ),
        key_points=("GPU 需求仍然强劲",),
        potential_opportunities=("NVIDIA 继续强调数据中心扩张",),
        industry_judgements=("AI 基础设施仍处于高景气阶段",),
        mentioned_companies=("NVIDIA",),
    )


if __name__ == "__main__":
    unittest.main()
