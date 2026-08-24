from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.agent import InterviewRankingRunner
from insightcast.domain.enums import (
    ContentFormat,
    Industry,
    InterviewStatus,
    InterviewType,
    SourceType,
    TranscriptStatus,
)
from insightcast.domain.models import (
    ExtractedInsight,
    Interview,
    InterviewSummary,
    Person,
    Source,
    UserInterest,
)
from insightcast.storage.repositories import InsightCastRepositories


class RankingRunnerTests(unittest.TestCase):
    def test_ranks_interviews_and_writes_scores(self) -> None:
        now = datetime(2026, 7, 12, 12, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.people.create(
                Person(
                    id="person_jensen",
                    name="Jensen Huang",
                    companies=("NVIDIA",),
                    industries=(Industry.AI,),
                    importance=0.95,
                )
            )
            repositories.people.create(
                Person(
                    id="person_minor",
                    name="Minor Person",
                    companies=("SmallCo",),
                    industries=(Industry.RETAIL,),
                    importance=0.3,
                )
            )
            repositories.sources.create(
                Source(
                    id="source_youtube",
                    name="YouTube Search",
                    type=SourceType.YOUTUBE,
                    authority=0.8,
                    industries=(Industry.AI,),
                )
            )
            repositories.sources.create(
                Source(
                    id="source_low",
                    name="Low Authority Source",
                    type=SourceType.WEBSITE,
                    authority=0.2,
                    industries=(Industry.RETAIL,),
                )
            )
            repositories.user_interests.create(
                UserInterest(
                    id="interest_ai",
                    industries=(Industry.AI,),
                    people=("person_jensen",),
                    companies=("NVIDIA",),
                    keywords=("AI infrastructure", "GPU"),
                )
            )

            high = make_interview(
                id="interview_high",
                title="Jensen Huang interview on AI infrastructure",
                source_id="source_youtube",
                person_ids=("person_jensen",),
                person_names=("Jensen Huang",),
                industries=(Industry.AI,),
                published_at=now - timedelta(days=1),
                duration_seconds=3600,
                transcript_status=TranscriptStatus.PARTIAL,
            )
            low = make_interview(
                id="interview_low",
                title="Minor retail interview",
                source_id="source_low",
                person_ids=("person_minor",),
                person_names=("Minor Person",),
                industries=(Industry.RETAIL,),
                published_at=now - timedelta(days=100),
                duration_seconds=300,
                transcript_status=TranscriptStatus.UNAVAILABLE,
            )
            repositories.interviews.create(high)
            repositories.interviews.create(low)
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=high.id,
                    summary_text="Jensen Huang 讨论 AI infrastructure 和 GPU 需求。",
                    companies=("NVIDIA",),
                    novelty="相比过往表述，本次更强调企业推理需求变化。",
                    insights=(
                        ExtractedInsight(
                            statement="企业 AI 推理需求正在扩大",
                            category="demand",
                            confidence=0.8,
                            mentioned_companies=("NVIDIA",),
                            mentioned_products=("GPU",),
                        ),
                    ),
                )
            )
            repositories.interview_summaries.create(
                make_summary(
                    interview_id=low.id,
                    summary_text="零售业务常规访谈。",
                    companies=("SmallCo",),
                    novelty="暂无历史观点可比。",
                )
            )

            result = InterviewRankingRunner(repositories, now=now).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.processed_count, 2)
            self.assertEqual(result.ranked_interview_ids[0], "interview_high")

            high_saved = repositories.interviews.require("interview_high")
            low_saved = repositories.interviews.require("interview_low")
            self.assertEqual(high_saved.status, InterviewStatus.PUSH_READY)
            self.assertEqual(low_saved.status, InterviewStatus.PUSH_READY)
            self.assertGreater(high_saved.importance_score, low_saved.importance_score)
            self.assertGreater(high_saved.novelty_score, low_saved.novelty_score)
            self.assertGreater(high_saved.relevance_score, low_saved.relevance_score)
            self.assertIn("score_breakdown", high_saved.metadata)
            self.assertEqual(high_saved.metadata["score_breakdown"]["person_importance"], 0.95)

            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "finished")
            self.assertEqual(run.metadata["stage"], "ranking")

    def test_missing_summary_marks_interview_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.interviews.create(
                make_interview(
                    id="interview_missing_summary",
                    title="Missing summary",
                    source_id="source_youtube",
                    person_ids=(),
                    person_names=(),
                    industries=(),
                )
            )

            result = InterviewRankingRunner(repositories).run_once()

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.processed_count, 0)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(
                repositories.interviews.require("interview_missing_summary").status,
                InterviewStatus.FAILED,
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.error, "1 ranking errors")


def make_interview(
    *,
    id: str,
    title: str,
    source_id: str,
    person_ids: tuple[str, ...],
    person_names: tuple[str, ...],
    industries: tuple[Industry, ...],
    published_at: datetime | None = None,
    duration_seconds: int | None = None,
    transcript_status: TranscriptStatus = TranscriptStatus.PARTIAL,
) -> Interview:
    return Interview(
        id=id,
        title=title,
        url=f"https://example.com/{id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.SUMMARY_READY,
        source_id=source_id,
        person_ids=person_ids,
        person_names=person_names,
        industries=industries,
        published_at=published_at,
        duration_seconds=duration_seconds,
        transcript_status=transcript_status,
    )


def make_summary(
    *,
    interview_id: str,
    summary_text: str,
    companies: tuple[str, ...],
    novelty: str,
    insights: tuple[ExtractedInsight, ...] = (),
) -> InterviewSummary:
    return InterviewSummary(
        interview_id=interview_id,
        summary=summary_text,
        key_points=(summary_text,),
        potential_opportunities=("关注机会",),
        industry_judgements=("行业判断",),
        mentioned_companies=companies,
        mentioned_products=("GPU",) if "NVIDIA" in companies else (),
        novelty_assessment=novelty,
        insights=insights,
    )


if __name__ == "__main__":
    unittest.main()
