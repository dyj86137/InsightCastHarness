from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.domain.enums import TranscriptSource, TranscriptStatus
from insightcast.domain.models import Interview, Transcript
from insightcast.evidence import (
    EvidenceClaim,
    EvidenceSpan,
    append_interview_evidence,
    locate_text,
    transcript_source_segments,
)


class EvidenceTests(unittest.TestCase):
    def test_locate_text_matches_exact_excerpt(self) -> None:
        self.assertEqual(locate_text("AI demand is growing.", "demand is growing"), (3, 20))

    def test_locate_text_matches_whitespace_normalized_excerpt(self) -> None:
        source = "AI demand\n\nis growing."
        start, end = locate_text(source, "AI demand is growing.")

        self.assertEqual((start, end), (0, len(source)))
        self.assertEqual(source[start:end], source)

    def test_locate_text_returns_empty_position_when_excerpt_is_absent(self) -> None:
        self.assertEqual(locate_text("Only the title is available.", "missing claim"), (None, None))

    def test_segment_ids_use_matching_source_ranges(self) -> None:
        transcript = Transcript(
            id="transcript_segments",
            interview_id="interview_segments",
            status=TranscriptStatus.READY,
            source=TranscriptSource.AUDIO_TRANSCRIPTION,
            text="First statement.\n\nSecond statement.",
            segments=("First statement.", "Second statement."),
        )

        segments = transcript_source_segments(transcript, transcript.text)

        self.assertEqual(
            [segment["segment_id"] for segment in segments],
            ["transcript_segment_0001", "transcript_segment_0002"],
        )
        self.assertEqual(
            transcript.text[segments[1]["start_char"] : segments[1]["end_char"]],
            "Second statement.",
        )

    def test_unmatched_segment_is_not_assigned_a_false_range(self) -> None:
        transcript = Transcript(
            id="transcript_unmatched_segment",
            interview_id="interview_unmatched_segment",
            status=TranscriptStatus.READY,
            source=TranscriptSource.AUDIO_TRANSCRIPTION,
            text="First statement.\n\nSecond statement.",
            segments=("Missing statement.", "Second statement."),
        )

        segments = transcript_source_segments(transcript, transcript.text)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["segment_id"], "transcript_segment_0002")
        self.assertEqual(
            transcript.text[segments[0]["start_char"] : segments[0]["end_char"]],
            "Second statement.",
        )

    def test_append_materializes_segment_id_and_unmatched_position(self) -> None:
        interview = Interview(
            id="interview_evidence",
            title="AI interview",
            url="https://example.com/interview-evidence",
        )
        transcript = Transcript(
            id="transcript_evidence",
            interview_id=interview.id,
            status=TranscriptStatus.READY,
            source=TranscriptSource.AUDIO_TRANSCRIPTION,
            text="AI demand is growing.",
            segments=("AI demand is growing.",),
        )
        claims = (
            EvidenceClaim(
                field="summary",
                claim="AI demand is growing.",
                evidence=(EvidenceSpan(text="AI demand is growing.", segment_id="wrong_id"),),
            ),
            EvidenceClaim(
                field="potential_opportunities",
                claim="Unsupported opportunity.",
                evidence=(EvidenceSpan(text="No such source text."),),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            append_interview_evidence(
                path,
                run_id="run_evidence",
                interview=interview,
                transcript=transcript,
                claims=claims,
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            record["claims"][0]["evidence"][0]["segment_id"],
            "transcript_segment_0001",
        )
        self.assertEqual(
            record["claims"][0]["evidence"][0]["text"],
            transcript.text[0:len("AI demand is growing.")],
        )
        self.assertEqual(record["claims"][1]["evidence"], [])

    def test_materialized_evidence_uses_source_slice_not_model_text(self) -> None:
        interview = Interview(
            id="interview_source_slice",
            title="AI interview",
            url="https://example.com/interview-source-slice",
        )
        transcript = Transcript(
            id="transcript_source_slice",
            interview_id=interview.id,
            status=TranscriptStatus.READY,
            source=TranscriptSource.AUDIO_TRANSCRIPTION,
            text="First statement.\n\nSecond statement.",
            segments=("First statement.", "Second statement."),
        )
        claims = (
            EvidenceClaim(
                field="summary",
                claim="First statement.",
                evidence=(EvidenceSpan(text="First statement."),),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            append_interview_evidence(
                path,
                run_id="run_source_slice",
                interview=interview,
                transcript=transcript,
                claims=claims,
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        span = record["claims"][0]["evidence"][0]
        self.assertEqual(span["text"], transcript.text[0:len("First statement.")])
        self.assertEqual(
            span["text"],
            transcript.text[span["start_char"] : span["end_char"]],
        )


if __name__ == "__main__":
    unittest.main()
