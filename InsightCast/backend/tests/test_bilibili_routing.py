from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.domain.enums import ContentFormat, SourceType
from insightcast.domain.models import CandidateItem, SearchQuery, Source
from insightcast.integrations.bilibili import BilibiliBackendRouter
from insightcast.sources.discovery import BilibiliDiscoverySource
from insightcast.transcript import download_candidate_audio, fetch_platform_caption


class BilibiliRoutingTests(unittest.TestCase):
    def test_bilibili_caption_uses_bili_cli_without_importing_yt_dlp(self) -> None:
        candidate = make_bilibili_candidate()
        payload = {
            "ok": True,
            "data": {
                "subtitle": {
                    "subtitles": [
                        {
                            "lan": "zh-Hans",
                            "body": [{"content": "Bilibili caption text."}],
                        }
                    ]
                }
            },
        }

        def run(command, **kwargs):
            self.assertEqual(command[:3], ["bili", "video", "BV1xx411c7mD"])
            self.assertIn("--subtitle", command)
            self.assertIn("--json", command)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with patch("insightcast.integrations.bilibili.shutil.which", return_value="bili"), patch(
            "insightcast.integrations.bilibili.subprocess.run", side_effect=run
        ), patch(
            "insightcast.transcript.fetcher.import_yt_dlp",
            side_effect=AssertionError("Bilibili must not import yt-dlp"),
        ):
            result = fetch_platform_caption(
                candidate,
                preferred_languages=("zh-Hans",),
                env={"INSIGHTCAST_BILIBILI_BACKENDS": "bili-cli,bilibili-api"},
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.text, "Bilibili caption text.")
        self.assertEqual(result.metadata["backend"], "bili-cli")

    def test_bilibili_cli_search_falls_back_to_public_api(self) -> None:
        calls = []

        def fetch_json(url, params, headers):
            calls.append((url, dict(params), dict(headers)))
            return {
                "code": 0,
                "data": {
                    "result": [
                        {
                            "bvid": "BV1xx411c7mD",
                            "title": "Fallback result",
                            "arcurl": "https://www.bilibili.com/video/BV1xx411c7mD",
                        }
                    ]
                },
            }

        def run(command, timeout):
            return subprocess.CompletedProcess(command, 1, "", "HTTP 412")

        router = BilibiliBackendRouter(
            env={"INSIGHTCAST_BILIBILI_BACKENDS": "bili-cli,bilibili-api"},
            fetch_json=fetch_json,
            command_runner=run,
        )
        with patch("insightcast.integrations.bilibili.shutil.which", return_value="bili"):
            result = router.search("黄仁勋 访谈", max_results=5)

        self.assertEqual(result.backend, "bilibili-api")
        self.assertEqual(result.items[0]["bvid"], "BV1xx411c7mD")
        self.assertEqual(calls[0][1]["order"], "totalrank")

    def test_bilibili_discovery_records_cli_backend(self) -> None:
        source = Source(
            id="source_bilibili",
            name="Bilibili Search",
            type=SourceType.BILIBILI,
            metadata={"max_results": 5},
        )
        query = SearchQuery(
            id="query_bilibili",
            text="黄仁勋 访谈",
            source_id=source.id,
            source_type=source.type,
        )
        payload = {
            "ok": True,
            "data": {
                "items": [
                    {
                        "bvid": "BV1xx411c7mD",
                        "title": "Jensen Huang interview",
                        "url": "https://www.bilibili.com/video/BV1xx411c7mD",
                    }
                ]
            },
        }

        def run(command, **kwargs):
            self.assertEqual(command[0:2], ["bili", "search"])
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with patch("insightcast.integrations.bilibili.shutil.which", return_value="bili"), patch(
            "insightcast.integrations.bilibili.subprocess.run", side_effect=run
        ):
            candidates = BilibiliDiscoverySource(
                source,
                env={"INSIGHTCAST_BILIBILI_BACKENDS": "bili-cli"},
                fetch_json=lambda url, params, headers: {},
            ).discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].raw_metadata["insightcast_bilibili_backend"],
            "bili-cli",
        )

    def test_bilibili_audio_uses_bili_cli_and_creates_candidate_scoped_file(self) -> None:
        candidate = make_bilibili_candidate()

        def run(command, **kwargs):
            output_dir = Path(command[command.index("-o") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "audio.m4a").write_bytes(b"audio")
            self.assertIn("--no-split", command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "insightcast.integrations.bilibili.shutil.which", return_value="bili"
        ), patch(
            "insightcast.integrations.bilibili.subprocess.run", side_effect=run
        ), patch(
            "insightcast.transcript.fetcher.import_yt_dlp",
            side_effect=AssertionError("Bilibili must not use yt-dlp audio"),
        ):
            result = download_candidate_audio(
                candidate,
                output_dir=Path(tmp),
                env={"INSIGHTCAST_BILIBILI_BACKENDS": "bili-cli,bilibili-api"},
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.metadata["backend"], "bili-cli")
        self.assertTrue(Path(result.audio_path).exists())
        self.assertIn("BV1xx411c7mD", result.audio_path)

    def test_bilibili_audio_returns_none_when_cli_missing_instead_of_using_yt_dlp(self) -> None:
        candidate = make_bilibili_candidate()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "insightcast.integrations.bilibili.shutil.which", return_value=None
        ), patch(
            "insightcast.transcript.fetcher.import_yt_dlp",
            side_effect=AssertionError("Bilibili must not use yt-dlp audio"),
        ):
            result = download_candidate_audio(
                candidate,
                output_dir=Path(tmp),
                env={"INSIGHTCAST_BILIBILI_BACKENDS": "bili-cli,bilibili-api"},
            )

        self.assertIsNone(result)


def make_bilibili_candidate() -> CandidateItem:
    return CandidateItem(
        id="candidate_bilibili",
        source_id="source_bilibili",
        source_name="Bilibili Search",
        source_type=SourceType.BILIBILI,
        platform_item_id="BV1xx411c7mD",
        title="Jensen Huang interview",
        description="A Bilibili interview.",
        url="https://www.bilibili.com/video/BV1xx411c7mD",
        format=ContentFormat.VIDEO,
    )


if __name__ == "__main__":
    unittest.main()
