from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.agent import InterviewPromotionRunner
from insightcast.domain.enums import (
    CandidateStatus,
    ContentFormat,
    InterviewType,
    SourceType,
)
from insightcast.domain.models import CandidateItem, InterviewDecision
from insightcast.storage.repositories import InsightCastRepositories


class InterviewPromotionRunnerTests(unittest.TestCase):
    def test_promotes_accepted_candidate_to_interview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_a",
                title="Jensen Huang full interview",
                platform_item_id="video_a",
                detected_person_names=("Jensen Huang",),
            )
            repositories.candidates.create(candidate)
            repositories.interview_decisions.save(make_decision(candidate.id))

            result = InterviewPromotionRunner(repositories).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.processed_count, 1)
            self.assertEqual(result.created_count, 1)
            self.assertEqual(result.merged_count, 0)
            self.assertEqual(repositories.interviews.count(), 1)
            self.assertEqual(
                repositories.candidates.require(candidate.id).status,
                CandidateStatus.ARCHIVED,
            )

            interview = repositories.interviews.list()[0]
            self.assertEqual(interview.title, "Jensen Huang full interview")
            self.assertEqual(interview.candidate_ids, ("candidate_a",))
            self.assertEqual(interview.type, InterviewType.INTERVIEW)
            self.assertEqual(interview.format, ContentFormat.VIDEO)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "finished")
            self.assertEqual(run.metadata["stage"], "interview_promotion")

    def test_merges_duplicate_candidates_into_existing_interview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            first = make_candidate(
                id="candidate_a",
                title="Sam Altman full interview",
                platform_item_id="video_same",
                url="https://www.youtube.com/watch?v=video_same&utm_source=newsletter",
                detected_person_names=("Sam Altman",),
            )
            second = make_candidate(
                id="candidate_b",
                title="Sam Altman full interview",
                platform_item_id="video_same",
                url="https://www.youtube.com/watch?v=video_same",
                detected_person_names=("Sam Altman",),
            )
            repositories.candidates.create(first)
            repositories.candidates.create(second)
            repositories.interview_decisions.save(make_decision(first.id))
            repositories.interview_decisions.save(make_decision(second.id))

            result = InterviewPromotionRunner(repositories).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.created_count, 1)
            self.assertEqual(result.merged_count, 1)
            self.assertEqual(repositories.interviews.count(), 1)
            self.assertEqual(
                repositories.candidates.require(first.id).status,
                CandidateStatus.ARCHIVED,
            )
            self.assertEqual(
                repositories.candidates.require(second.id).status,
                CandidateStatus.DUPLICATE,
            )

            interview = repositories.interviews.list()[0]
            self.assertEqual(interview.candidate_ids, ("candidate_a", "candidate_b"))
            self.assertEqual(interview.metadata["duplicate_candidate_ids"], ["candidate_b"])

    def test_missing_decision_marks_candidate_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_missing_decision",
                title="Accepted but missing decision",
                platform_item_id="video_missing",
            )
            repositories.candidates.create(candidate)

            result = InterviewPromotionRunner(repositories).run_once()

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.processed_count, 0)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(repositories.interviews.count(), 0)
            self.assertEqual(
                repositories.candidates.require(candidate.id).status,
                CandidateStatus.FAILED,
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "failed")
            self.assertEqual(run.error, "1 interview promotion errors")


def make_candidate(
    *,
    id: str,
    title: str,
    platform_item_id: str,
    url: str | None = None,
    detected_person_names: tuple[str, ...] = (),
) -> CandidateItem:
    return CandidateItem(
        id=id,
        source_id="source_youtube",
        source_name="YouTube Search",
        source_type=SourceType.YOUTUBE,
        platform_item_id=platform_item_id,
        title=title,
        url=url or f"https://www.youtube.com/watch?v={platform_item_id}",
        format=ContentFormat.VIDEO,
        detected_person_names=detected_person_names,
        status=CandidateStatus.ACCEPTED,
    )


def make_decision(candidate_id: str) -> InterviewDecision:
    return InterviewDecision(
        candidate_id=candidate_id,
        is_qualifying_content=True,
        target_person_present=True,
        content_type=InterviewType.INTERVIEW,
        is_short_clip_or_commentary=False,
        confidence=0.9,
        reason="LLM judged this as a real interview.",
    )


if __name__ == "__main__":
    unittest.main()
