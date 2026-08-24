from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from memory import MemoryKind

from insightcast.domain.enums import (
    ContentFormat,
    Industry,
    InterviewStatus,
    InterviewType,
    TranscriptStatus,
)
from insightcast.domain.models import Interview, InterviewSummary
from insightcast.memory import (
    PersonHistoryMemoryProvider,
    normalize_memory_tag,
    person_memory_tag,
    summary_to_memory_record,
)
from insightcast.storage.repositories import InsightCastRepositories


class PersonHistoryMemoryTests(unittest.TestCase):
    def test_normalize_memory_tag_preserves_chinese_names(self) -> None:
        self.assertEqual(normalize_memory_tag("雷军"), "雷军")
        self.assertEqual(person_memory_tag("雷军"), "person:雷军")
        self.assertEqual(normalize_memory_tag("Lei Jun / 雷军"), "lei_jun_雷军")

    def test_summary_to_memory_record_uses_myharness_memory_model(self) -> None:
        interview = make_interview(
            id="interview_old",
            title="Jensen Huang on enterprise AI inference",
            person_names=("Jensen Huang",),
            published_at=datetime(2026, 7, 1, 12, 0, 0),
        )
        summary = make_summary(
            interview_id=interview.id,
            summary_text="Jensen Huang 重点谈企业 AI 推理需求。",
            novelty="相比此前，更强调推理侧需求。",
        )

        record = summary_to_memory_record(interview, summary)

        self.assertEqual(record.kind, MemoryKind.LONG_TERM)
        self.assertIn(person_memory_tag("Jensen Huang"), record.tags)
        self.assertIn("关注机会", record.content)
        self.assertEqual(record.metadata["interview_id"], interview.id)

    def test_person_history_provider_builds_same_person_injection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            now = datetime(2026, 7, 12, 12, 0, 0)
            current = make_interview(
                id="interview_current",
                title="Jensen Huang latest AI infrastructure interview",
                person_names=("Jensen Huang",),
                published_at=now,
            )
            old = make_interview(
                id="interview_old",
                title="Jensen Huang prior data center interview",
                person_names=("Jensen Huang",),
                published_at=now - timedelta(days=30),
            )
            other = make_interview(
                id="interview_other",
                title="Other CEO interview",
                person_names=("Other CEO",),
                published_at=now - timedelta(days=1),
            )
            repositories.interviews.create(current)
            repositories.interviews.create(old)
            repositories.interviews.create(other)
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=old.id,
                    summary_text="Jensen Huang 此前强调数据中心 GPU 供给。",
                    novelty="此前主要强调训练和数据中心供给。",
                )
            )
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=other.id,
                    summary_text="Other CEO 讨论零售业务。",
                    novelty="无关历史观点。",
                )
            )

            provider = PersonHistoryMemoryProvider(repositories, top_k=2)
            injection = asyncio.run(provider.build_summary_memory(current))

            self.assertEqual(len(injection.results), 1)
            self.assertIn("人物历史 memory", injection.content)
            self.assertIn("数据中心 GPU 供给", injection.content)
            self.assertNotIn("零售业务", injection.content)
            self.assertNotIn("interview_current", injection.content)
            self.assertEqual(
                injection.metadata["provider"],
                "insightcast_person_history",
            )

    def test_person_history_provider_matches_chinese_person_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            current = make_interview(
                id="interview_current",
                title="雷军最新访谈：小米汽车交付进展",
                person_names=("雷军",),
            )
            old = make_interview(
                id="interview_old",
                title="雷军谈小米汽车产品定位",
                person_names=("雷军",),
            )
            other = make_interview(
                id="interview_other",
                title="李想谈智能电动车",
                person_names=("李想",),
            )
            repositories.interviews.create(current)
            repositories.interviews.create(old)
            repositories.interviews.create(other)
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=old.id,
                    summary_text="雷军此前强调小米汽车要围绕用户体验建立差异化。",
                    novelty="此前主要强调产品定位和用户体验。",
                )
            )
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=other.id,
                    summary_text="李想讨论增程路线和家庭用户。",
                    novelty="无关历史观点。",
                )
            )

            provider = PersonHistoryMemoryProvider(repositories, top_k=2)
            injection = asyncio.run(provider.build_summary_memory(current))

            self.assertEqual(len(injection.results), 1)
            self.assertIn("用户体验", injection.content)
            self.assertNotIn("增程路线", injection.content)


def make_interview(
    *,
    id: str,
    title: str,
    person_names: tuple[str, ...],
    published_at: datetime | None = None,
) -> Interview:
    return Interview(
        id=id,
        title=title,
        url=f"https://example.com/{id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.SUMMARY_READY,
        person_names=person_names,
        industries=(Industry.AI,),
        published_at=published_at,
        transcript_status=TranscriptStatus.PARTIAL,
        importance_score=0.8,
    )


def make_summary(
    *,
    interview_id: str,
    summary_text: str,
    novelty: str,
) -> InterviewSummary:
    return InterviewSummary(
        interview_id=interview_id,
        summary=summary_text,
        key_points=(summary_text,),
        potential_opportunities=("NVIDIA 继续强调数据中心扩张",),
        industry_judgements=("AI 基础设施仍处于高景气阶段",),
        mentioned_companies=("NVIDIA",),
        mentioned_products=("GPU",),
        novelty_assessment=novelty,
    )


if __name__ == "__main__":
    unittest.main()
