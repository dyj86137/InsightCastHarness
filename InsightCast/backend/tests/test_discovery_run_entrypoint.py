from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.agent import DiscoveryRunResult
from insightcast.run.discovery_run import format_result, main, run_discovery
from insightcast.storage.repositories import InsightCastRepositories


class DiscoveryRunEntrypointTests(unittest.TestCase):
    def test_run_discovery_loads_config_and_writes_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "storage"
            result = run_discovery(
                config_path=BACKEND_ROOT / "configs" / "local.example.json",
                storage_dir=storage_dir,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=successful_fetch_json,
            )

            repositories = InsightCastRepositories(root_dir=storage_dir)
            self.assertEqual(result.status, "finished")
            self.assertGreater(result.query_count, 0)
            self.assertEqual(repositories.run_records.count(), 1)
            self.assertEqual(repositories.candidates.count(), result.discovered_count)

    def test_main_prints_summary_and_returns_status_code(self) -> None:
        result = DiscoveryRunResult(
            run_id="run_test",
            status="partial",
            query_count=3,
            discovered_count=1,
            created_count=1,
        )

        with patch("insightcast.run.discovery_run.run_discovery", return_value=result):
            with patch("builtins.print") as print_mock:
                code = main(["--config", "example.json"])

        self.assertEqual(code, 2)
        printed = print_mock.call_args[0][0]
        self.assertIn("Discovery run partial: run_test", printed)
        self.assertIn("queries: 3", printed)

    def test_format_result_includes_error_summary(self) -> None:
        result = DiscoveryRunResult(
            run_id="run_test",
            status="finished",
            query_count=1,
            discovered_count=1,
            created_count=1,
        )

        self.assertIn("Discovery run finished: run_test", format_result(result))


def successful_fetch_json(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    if "googleapis" in url:
        video_id = stable_suffix("yt", params["q"])
        return {
            "items": [
                {
                    "id": {"videoId": video_id},
                    "snippet": {
                        "title": f"{params['q']} video",
                        "description": "Interview candidate",
                        "publishedAt": "2026-07-09T12:00:00Z",
                    },
                }
            ]
        }

    bvid = stable_suffix("BV", params["keyword"])
    return {
        "code": 0,
        "data": {
            "result": [
                {
                    "bvid": bvid,
                    "title": f"{params['keyword']} 视频",
                    "description": "候选访谈",
                    "duration": "10:00",
                }
            ]
        },
    }


def stable_suffix(prefix: str, value: object) -> str:
    digest = hex(abs(hash(str(value))))[2:10]
    return f"{prefix}{digest}"


if __name__ == "__main__":
    unittest.main()
