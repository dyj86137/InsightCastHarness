from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM

from insightcast.agent import ClassificationRunner, LLMInterviewClassifier
from insightcast.domain.enums import CandidateStatus, ContentFormat, InterviewType, SourceType
from insightcast.domain.models import CandidateItem
from insightcast.storage.repositories import InsightCastRepositories


class InterviewClassifierTests(unittest.TestCase):
    def test_llm_classifier_returns_interview_decision(self) -> None:
        llm = FakeLLM(
            responses=[
                """
                {
                  "is_qualifying_content": true,
                  "target_person_present": true,
                  "content_type": "podcast_interview",
                  "is_short_clip_or_commentary": false,
                  "confidence": 0.91,
                  "reason": "The title and description describe a long-form interview with the target person."
                }
                """
            ]
        )
        classifier = LLMInterviewClassifier(llm)
        candidate = make_candidate(
            id="candidate_sam",
            title="Sam Altman long-form podcast interview",
            description="A conversation with OpenAI CEO Sam Altman.",
            detected_person_names=("Sam Altman",),
            query_text="Sam Altman interview",
        )

        decision = asyncio.run(classifier.classify_async(candidate))

        self.assertEqual(decision.candidate_id, "candidate_sam")
        self.assertTrue(decision.is_qualifying_content)
        self.assertTrue(decision.target_person_present)
        self.assertEqual(decision.content_type, InterviewType.PODCAST_INTERVIEW)
        self.assertEqual(decision.confidence, 0.91)
        self.assertEqual(decision.metadata["provider"], "fake")
        self.assertIn("Sam Altman long-form podcast interview", llm.last_request.messages[1].content)
        self.assertIn('"format": "video"', llm.last_request.messages[1].content)

    def test_classification_runner_persists_decisions_and_candidate_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.candidates.create(
                make_candidate(
                    id="candidate_accept",
                    title="Jensen Huang interview",
                    description="A long-form conversation.",
                    detected_person_names=("Jensen Huang",),
                    query_text="Jensen Huang interview",
                )
            )
            repositories.candidates.create(
                make_candidate(
                    id="candidate_reject",
                    title="AI news recap about Sam Altman",
                    description="News commentary with clips.",
                    detected_person_names=("Sam Altman",),
                    query_text="Sam Altman interview",
                )
            )
            llm = FakeLLM(
                responses=[
                    """
                    {
                      "is_qualifying_content": true,
                      "target_person_present": true,
                      "content_type": "interview",
                      "is_short_clip_or_commentary": false,
                      "confidence": 0.88,
                      "reason": "The metadata indicates a real interview."
                    }
                    """,
                    """
                    {
                      "is_qualifying_content": false,
                      "target_person_present": false,
                      "content_type": "news_clip",
                      "is_short_clip_or_commentary": true,
                      "confidence": 0.82,
                      "reason": "This is a news recap and commentary, not the target person's interview."
                    }
                    """,
                ]
            )
            runner = ClassificationRunner(
                repositories,
                LLMInterviewClassifier(llm),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.processed_count, 2)
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 1)
            self.assertEqual(repositories.interview_decisions.count(), 2)
            self.assertEqual(
                repositories.candidates.require("candidate_accept").status,
                CandidateStatus.ACCEPTED,
            )
            self.assertEqual(
                repositories.candidates.require("candidate_reject").status,
                CandidateStatus.REJECTED,
            )

    def test_classification_runner_can_preserve_duplicate_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            first = make_candidate(
                id="candidate_duplicate_first",
                title="Jensen Huang interview first occurrence",
                detected_person_names=("Jensen Huang",),
                query_text="Jensen Huang interview",
            )
            second = make_candidate(
                id="candidate_duplicate_second",
                title="Jensen Huang interview second occurrence",
                detected_person_names=("Jensen Huang",),
                query_text="Jensen Huang interview",
            ).clone(
                platform_item_id=first.platform_item_id,
                url=first.url,
            )
            repositories.candidates.create(first)
            repositories.candidates.create(second)
            accepted_response = """
            {
              "is_qualifying_content": true,
              "target_person_present": true,
              "content_type": "interview",
              "is_short_clip_or_commentary": false,
              "confidence": 0.9,
              "reason": "Both occurrences are evaluated independently."
            }
            """

            result = ClassificationRunner(
                repositories,
                LLMInterviewClassifier(FakeLLM(responses=[accepted_response, accepted_response])),
                deduplicate=False,
            ).run_once()

            self.assertEqual(result.processed_count, 2)
            self.assertEqual(result.accepted_count, 2)
            self.assertEqual(result.duplicate_count, 0)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "finished")
            self.assertEqual(run.metadata["stage"], "classification")

    def test_classification_runner_keeps_low_confidence_positive_as_classified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.candidates.create(
                make_candidate(
                    id="candidate_uncertain",
                    title="Possible Jensen Huang interview",
                    description="The metadata is ambiguous.",
                    detected_person_names=("Jensen Huang",),
                    query_text="Jensen Huang interview",
                )
            )
            llm = FakeLLM(
                responses=[
                    """
                    {
                      "is_qualifying_content": true,
                      "target_person_present": true,
                      "content_type": "interview",
                      "is_short_clip_or_commentary": false,
                      "confidence": 0.61,
                      "reason": "It may be an interview, but metadata is not strong enough."
                    }
                    """,
                ]
            )
            runner = ClassificationRunner(
                repositories,
                LLMInterviewClassifier(llm),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.accepted_count, 0)
            self.assertEqual(result.classified_count, 1)
            self.assertEqual(result.rejected_count, 0)
            self.assertEqual(
                repositories.candidates.require("candidate_uncertain").status,
                CandidateStatus.CLASSIFIED,
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.metadata["classified_count"], 1)

    def test_classification_runner_marks_cross_platform_original_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.candidates.create(
                make_candidate(
                    id="candidate_original_mirror",
                    title="Original video mirror",
                    description="Original: https://www.youtube.com/watch?v=video123",
                    query_text="Sam Altman interview",
                )
            )
            repositories.candidates.create(
                make_candidate(
                    id="candidate_duplicate_mirror",
                    title="Same video, translated upload",
                    description="Source: https://youtu.be/video123",
                    query_text="Sam Altman interview",
                )
            )
            llm = FakeLLM(
                responses=[
                    """
                    {
                      "is_qualifying_content": true,
                      "target_person_present": true,
                      "content_type": "interview",
                      "is_short_clip_or_commentary": false,
                      "confidence": 0.9,
                      "reason": "The metadata indicates a real interview."
                    }
                    """
                ]
            )
            runner = ClassificationRunner(
                repositories,
                LLMInterviewClassifier(llm),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.processed_count, 1)
            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.duplicate_count, 1)
            self.assertEqual(repositories.interview_decisions.count(), 1)
            candidate_ids = (
                "candidate_original_mirror",
                "candidate_duplicate_mirror",
            )
            accepted_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if repositories.candidates.require(candidate_id).status
                == CandidateStatus.ACCEPTED
            ]
            duplicate_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if repositories.candidates.require(candidate_id).status
                == CandidateStatus.DUPLICATE
            ]
            self.assertEqual(len(accepted_ids), 1)
            self.assertEqual(len(duplicate_ids), 1)
            self.assertEqual(result.duplicate_candidate_ids, tuple(duplicate_ids))
            duplicate = repositories.candidates.require(duplicate_ids[0])
            self.assertEqual(
                duplicate.raw_metadata["insightcast_duplicate_of"],
                accepted_ids[0],
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.discovered_count, 2)
            self.assertEqual(run.metadata["duplicate_count"], 1)

    def test_classification_runner_records_partial_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            repositories.candidates.create(
                make_candidate(id="candidate_a_fail", title="Broken candidate")
            )
            repositories.candidates.create(
                make_candidate(id="candidate_z_accept", title="Real interview")
            )
            llm = FakeLLM(
                responses=[
                    RuntimeError("simulated llm failure"),
                    """
                    {
                      "is_qualifying_content": true,
                      "target_person_present": true,
                      "content_type": "interview",
                      "is_short_clip_or_commentary": false,
                      "confidence": 0.9,
                      "reason": "Looks like a real interview."
                    }
                    """,
                ]
            )
            runner = ClassificationRunner(
                repositories,
                LLMInterviewClassifier(llm),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "partial")
            self.assertEqual(result.processed_count, 1)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(
                repositories.candidates.require("candidate_a_fail").status,
                CandidateStatus.FAILED,
            )
            self.assertEqual(
                repositories.candidates.require("candidate_z_accept").status,
                CandidateStatus.ACCEPTED,
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "partial")
            self.assertEqual(run.error, "1 classification errors")


def make_candidate(
    *,
    id: str,
    title: str,
    description: str | None = None,
    detected_person_names: tuple[str, ...] = (),
    query_text: str | None = None,
) -> CandidateItem:
    return CandidateItem(
        id=id,
        source_id="source_youtube",
        source_name="YouTube Search",
        source_type=SourceType.YOUTUBE,
        platform_item_id=id,
        title=title,
        description=description,
        url=f"https://www.youtube.com/watch?v={id}",
        format=ContentFormat.VIDEO,
        detected_person_names=detected_person_names,
        query_text=query_text,
        status=CandidateStatus.DISCOVERED,
    )


if __name__ == "__main__":
    unittest.main()
