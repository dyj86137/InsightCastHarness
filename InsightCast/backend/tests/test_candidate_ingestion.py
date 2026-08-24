from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.domain.enums import CandidateStatus, SourceType
from insightcast.domain.models import CandidateItem
from insightcast.sources import (
    candidate_video_identity_keys,
    ingest_candidates,
    normalize_candidate_url,
    video_identity_from_url,
)
from insightcast.storage.repositories import CandidateRepository


class CandidateIngestionTests(unittest.TestCase):
    def test_ingests_new_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = CandidateRepository(root_dir=Path(tmp) / "storage")
            candidate = make_candidate(
                id="candidate_one",
                platform_item_id="abc123",
                title="Sam Altman interview",
                url="https://www.youtube.com/watch?v=abc123",
            )

            result = ingest_candidates(repository, [candidate])

            self.assertEqual(result.discovered_count, 1)
            self.assertEqual(result.created_count, 1)
            self.assertEqual(result.updated_count, 0)
            self.assertEqual(result.duplicate_count, 0)
            self.assertEqual(repository.count(), 1)
            self.assertEqual(repository.require("candidate_one").title, "Sam Altman interview")

    def test_updates_duplicate_by_platform_id_and_preserves_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = CandidateRepository(root_dir=Path(tmp) / "storage")
            existing = make_candidate(
                id="candidate_existing",
                platform_item_id="abc123",
                title="Old title",
                url="https://www.youtube.com/watch?v=abc123",
                status=CandidateStatus.ACCEPTED,
                raw_metadata={"old": True},
            )
            repository.create(existing)

            incoming = make_candidate(
                id="candidate_incoming",
                platform_item_id="abc123",
                title="New title",
                url="https://www.youtube.com/watch?v=abc123",
                detected_person_names=("Sam Altman",),
                raw_metadata={"new": True},
            )
            result = ingest_candidates(repository, [incoming])
            saved = repository.require("candidate_existing")

            self.assertEqual(result.created_count, 0)
            self.assertEqual(result.updated_count, 1)
            self.assertEqual(result.duplicate_count, 1)
            self.assertEqual(repository.count(), 1)
            self.assertEqual(saved.id, "candidate_existing")
            self.assertEqual(saved.title, "New title")
            self.assertEqual(saved.status, CandidateStatus.ACCEPTED)
            self.assertEqual(saved.detected_person_names, ("Sam Altman",))
            self.assertEqual(saved.raw_metadata, {"old": True, "new": True})

    def test_dedupes_by_normalized_url_within_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = CandidateRepository(root_dir=Path(tmp) / "storage")
            first = make_candidate(
                id="candidate_first",
                platform_item_id=None,
                title="First",
                url="https://www.youtube.com/watch?v=abc123&utm_source=newsletter#top",
            )
            second = make_candidate(
                id="candidate_second",
                platform_item_id=None,
                title="Second",
                url="https://www.youtube.com/watch?v=abc123",
            )

            result = ingest_candidates(repository, [first, second])

            self.assertEqual(result.created_count, 1)
            self.assertEqual(result.updated_count, 1)
            self.assertEqual(result.duplicate_count, 1)
            self.assertEqual(repository.count(), 1)
            self.assertEqual(repository.require("candidate_first").title, "Second")

    def test_normalizes_tracking_url_parts(self) -> None:
        normalized = normalize_candidate_url(
            "https://WWW.YouTube.com/watch?utm_source=x&v=abc123&utm_campaign=y#frag"
        )

        self.assertEqual(normalized, "https://www.youtube.com/watch?v=abc123")

    def test_extracts_same_youtube_identity_from_watch_and_short_urls(self) -> None:
        first = make_candidate(
            id="candidate_bilibili_one",
            platform_item_id="BVone",
            title="Bilibili mirror one",
            url="https://www.bilibili.com/video/BVone",
            description="原始视频：https://www.youtube.com/watch?v=video123",
        )
        second = make_candidate(
            id="candidate_bilibili_two",
            platform_item_id="BVtwo",
            title="Bilibili mirror two",
            url="https://www.bilibili.com/video/BVtwo",
            description="Source: https://youtu.be/video123?t=10",
        )

        self.assertEqual(
            video_identity_from_url("https://www.youtube.com/watch?v=video123"),
            "youtube:video:video123",
        )
        self.assertEqual(
            video_identity_from_url("https://youtu.be/video123?t=10"),
            "youtube:video:video123",
        )
        self.assertIn(
            "youtube:video:video123",
            candidate_video_identity_keys(first),
        )
        self.assertEqual(
            set(candidate_video_identity_keys(first))
            & set(candidate_video_identity_keys(second)),
            {"youtube:video:video123"},
        )


def make_candidate(
    *,
    id: str,
    platform_item_id: str | None,
    title: str,
    url: str,
    description: str | None = None,
    status: CandidateStatus = CandidateStatus.DISCOVERED,
    detected_person_names: tuple[str, ...] = (),
    raw_metadata: dict[str, object] | None = None,
) -> CandidateItem:
    return CandidateItem(
        id=id,
        source_id="source_youtube",
        source_name="YouTube Search",
        source_type=SourceType.YOUTUBE,
        platform_item_id=platform_item_id,
        title=title,
        description=description,
        url=url,
        status=status,
        detected_person_names=detected_person_names,
        raw_metadata=raw_metadata or {},
    )


if __name__ == "__main__":
    unittest.main()
