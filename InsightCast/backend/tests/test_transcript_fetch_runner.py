from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.agent import TranscriptFetchRunner
from insightcast.domain.enums import (
    CandidateStatus,
    ContentFormat,
    InterviewStatus,
    InterviewType,
    SourceType,
    TranscriptSource,
    TranscriptStatus,
)
from insightcast.domain.models import CandidateItem, Interview
from insightcast.storage.repositories import InsightCastRepositories
from insightcast.transcript import (
    AudioDownloadResult,
    AudioTranscriptionResult,
    AudioTranscriptionTranscriptProvider,
    CandidateMetadataTranscriptProvider,
    CompositeTranscriptProvider,
    PlatformCaptionFetchResult,
    PlatformCaptionTranscriptProvider,
)


class TranscriptFetchRunnerTests(unittest.TestCase):
    def test_fetches_partial_transcript_from_candidate_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_a",
                title="Jensen Huang interview",
                description="A long-form conversation about AI infrastructure.",
            )
            interview = make_interview(
                id="interview_a",
                title="Jensen Huang interview",
                candidate_ids=(candidate.id,),
            )
            repositories.candidates.create(candidate)
            repositories.interviews.create(interview)

            result = TranscriptFetchRunner(repositories).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.partial_count, 1)
            self.assertEqual(result.unavailable_count, 0)
            transcript = repositories.transcripts.find_by_interview_id(interview.id)
            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.status, TranscriptStatus.PARTIAL)
            self.assertEqual(transcript.source, TranscriptSource.SOURCE_METADATA)
            self.assertIn("A long-form conversation", transcript.text)
            updated_interview = repositories.interviews.require(interview.id)
            self.assertEqual(updated_interview.transcript_status, TranscriptStatus.PARTIAL)
            self.assertEqual(updated_interview.status, InterviewStatus.TRANSCRIPT_READY)

    def test_marks_transcript_unavailable_when_only_title_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_title_only",
                title="Title only video",
                description=None,
            )
            interview = make_interview(
                id="interview_title_only",
                title="Title only video",
                candidate_ids=(candidate.id,),
            )
            repositories.candidates.create(candidate)
            repositories.interviews.create(interview)

            result = TranscriptFetchRunner(repositories).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.partial_count, 0)
            self.assertEqual(result.unavailable_count, 1)
            transcript = repositories.transcripts.find_by_interview_id(interview.id)
            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.status, TranscriptStatus.UNAVAILABLE)
            self.assertEqual(transcript.source, TranscriptSource.UNKNOWN)
            updated_interview = repositories.interviews.require(interview.id)
            self.assertEqual(updated_interview.transcript_status, TranscriptStatus.UNAVAILABLE)
            self.assertEqual(updated_interview.status, InterviewStatus.NEW)

    def test_fetching_again_updates_existing_transcript_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_repeat",
                title="Repeat interview",
                description="Initial description.",
            )
            interview = make_interview(
                id="interview_repeat",
                title="Repeat interview",
                candidate_ids=(candidate.id,),
            )
            repositories.candidates.create(candidate)
            repositories.interviews.create(interview)

            first_result = TranscriptFetchRunner(repositories).run_once()
            transcript = repositories.transcripts.find_by_interview_id(interview.id)
            repositories.interviews.update(
                interview.id,
                transcript_status=TranscriptStatus.UNAVAILABLE,
                status=InterviewStatus.NEW,
            )
            repositories.candidates.update(
                candidate.id,
                description="Updated description.",
            )
            second_result = TranscriptFetchRunner(repositories).run_once()
            updated = repositories.transcripts.find_by_interview_id(interview.id)

            self.assertEqual(first_result.partial_count, 1)
            self.assertEqual(second_result.partial_count, 1)
            self.assertEqual(updated.id, transcript.id)
            self.assertIn("Updated description.", updated.text)

    def test_limit_restricts_processed_interviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            for index in range(2):
                candidate = make_candidate(
                    id=f"candidate_limit_{index}",
                    title=f"Limited interview {index}",
                    description="A usable description.",
                )
                interview = make_interview(
                    id=f"interview_limit_{index}",
                    title=f"Limited interview {index}",
                    candidate_ids=(candidate.id,),
                )
                repositories.candidates.create(candidate)
                repositories.interviews.create(interview)

            result = TranscriptFetchRunner(repositories, limit=1).run_once()

            self.assertEqual(result.processed_count, 1)
            self.assertEqual(result.partial_count, 1)
            transcripts = repositories.transcripts.list()
            self.assertEqual(len(transcripts), 1)

    def test_fetches_ready_transcript_from_platform_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_caption",
                title="Caption interview",
                description=None,
            )
            interview = make_interview(
                id="interview_caption",
                title="Caption interview",
                candidate_ids=(candidate.id,),
            )
            repositories.candidates.create(candidate)
            repositories.interviews.create(interview)
            provider = CompositeTranscriptProvider(
                (
                    PlatformCaptionTranscriptProvider(
                        fetcher=lambda candidate: PlatformCaptionFetchResult(
                            candidate_id=candidate.id,
                            text="Full platform caption text.",
                            source=TranscriptSource.PLATFORM_CAPTION,
                            language="en",
                        )
                    ),
                    CandidateMetadataTranscriptProvider(),
                )
            )

            result = TranscriptFetchRunner(repositories, provider=provider).run_once()

            self.assertEqual(result.ready_count, 1)
            transcript = repositories.transcripts.find_by_interview_id(interview.id)
            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.status, TranscriptStatus.READY)
            self.assertEqual(transcript.source, TranscriptSource.PLATFORM_CAPTION)
            self.assertIn("Full platform caption", transcript.text)

    def test_uses_audio_transcription_after_missing_platform_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_audio",
                title="Audio interview",
                description=None,
            )
            interview = make_interview(
                id="interview_audio",
                title="Audio interview",
                candidate_ids=(candidate.id,),
            )
            repositories.candidates.create(candidate)
            repositories.interviews.create(interview)
            provider = CompositeTranscriptProvider(
                (
                    PlatformCaptionTranscriptProvider(fetcher=lambda candidate: None),
                    AudioTranscriptionTranscriptProvider(
                        downloader=lambda candidate: AudioDownloadResult(
                            candidate_id=candidate.id,
                            audio_path=str(Path(tmp) / "audio.m4a"),
                        ),
                        transcriber=lambda download, interview, candidate: AudioTranscriptionResult(
                            text="Audio transcription text.",
                            language="en",
                        ),
                    ),
                    CandidateMetadataTranscriptProvider(),
                )
            )

            result = TranscriptFetchRunner(repositories, provider=provider).run_once()

            self.assertEqual(result.ready_count, 1)
            transcript = repositories.transcripts.find_by_interview_id(interview.id)
            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.status, TranscriptStatus.READY)
            self.assertEqual(transcript.source, TranscriptSource.AUDIO_TRANSCRIPTION)
            self.assertIn("Audio transcription", transcript.text)


def make_candidate(
    *,
    id: str,
    title: str,
    description: str | None,
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
        status=CandidateStatus.ARCHIVED,
    )


def make_interview(
    *,
    id: str,
    title: str,
    candidate_ids: tuple[str, ...],
) -> Interview:
    return Interview(
        id=id,
        title=title,
        url=f"https://www.youtube.com/watch?v={id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.NEW,
        candidate_ids=candidate_ids,
        person_names=("Jensen Huang",),
        transcript_status=TranscriptStatus.UNAVAILABLE,
    )


if __name__ == "__main__":
    unittest.main()
