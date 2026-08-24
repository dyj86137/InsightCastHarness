from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM

from insightcast.agent import DiscoveryResearchAgent, DiscoveryRunner
from insightcast.config import apply_config, load_config
from insightcast.storage.repositories import InsightCastRepositories


class DiscoveryRunnerTests(unittest.TestCase):
    def test_run_once_generates_queries_discovers_and_ingests_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = prepare_repositories(Path(tmp))
            runner = DiscoveryRunner(
                repositories,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=successful_fetch_json,
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertGreater(result.query_count, 0)
            self.assertGreater(result.discovered_count, 0)
            self.assertEqual(result.created_count, result.discovered_count)
            self.assertEqual(result.updated_count, 0)
            self.assertEqual(result.errors, ())
            self.assertEqual(repositories.candidates.count(), result.discovered_count)

            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "finished")
            self.assertEqual(run.discovered_count, result.discovered_count)
            self.assertEqual(run.accepted_count, result.created_count)
            self.assertEqual(run.pushed_count, 0)
            self.assertEqual(run.metadata["query_count"], result.query_count)

            second_result = runner.run_once()
            self.assertEqual(second_result.status, "finished")
            self.assertEqual(second_result.created_count, 0)
            self.assertEqual(second_result.updated_count, second_result.discovered_count)
            self.assertEqual(repositories.candidates.count(), result.discovered_count)

    def test_run_once_can_use_react_discovery_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = prepare_repositories(Path(tmp))
            agent_llm = FakeLLM(
                responses=[
                    (
                        "Thought: I should prepare the configured discovery queries.\n"
                        "Action: insightcast_prepare_discovery_queries\n"
                        "Action Input: {}"
                    ),
                    (
                        "Thought: The queries are ready, so I should run discovery.\n"
                        "Action: insightcast_run_discovery_queries\n"
                        "Action Input: {}"
                    ),
                    "Final Answer: discovery finished",
                ]
            )
            runner = DiscoveryRunner(
                repositories,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=successful_fetch_json,
                discovery_agent=DiscoveryResearchAgent(
                    agent_llm,
                    env={"YOUTUBE_API_KEY": "test-key"},
                    fetch_json=successful_fetch_json,
                ),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "finished")
            self.assertGreater(result.query_count, 0)
            self.assertGreater(result.discovered_count, 0)
            self.assertEqual(result.errors, ())
            self.assertEqual(repositories.candidates.count(), result.discovered_count)
            self.assertEqual(len(agent_llm.requests), 3)

    def test_run_once_falls_back_when_react_discovery_agent_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = prepare_repositories(Path(tmp))
            runner = DiscoveryRunner(
                repositories,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=successful_fetch_json,
                discovery_agent=DiscoveryResearchAgent(
                    FakeLLM(responses=["not react format"]),
                    env={"YOUTUBE_API_KEY": "test-key"},
                    fetch_json=successful_fetch_json,
                ),
            )

            result = runner.run_once()

            self.assertEqual(result.status, "partial")
            self.assertGreater(result.discovered_count, 0)
            self.assertEqual(result.errors[0].error_type, "DiscoveryResearchAgentFailed")
            self.assertEqual(repositories.candidates.count(), result.discovered_count)

    def test_run_once_records_partial_failures_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = prepare_repositories(Path(tmp))
            runner = DiscoveryRunner(
                repositories,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=partially_failing_fetch_json,
            )

            result = runner.run_once()

            self.assertEqual(result.status, "partial")
            self.assertGreater(result.discovered_count, 0)
            self.assertGreater(len(result.errors), 0)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "partial")
            self.assertEqual(run.error, f"{len(result.errors)} discovery errors")
            self.assertEqual(len(run.metadata["errors"]), len(result.errors))

    def test_run_once_marks_failed_when_all_queries_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = prepare_repositories(Path(tmp))
            runner = DiscoveryRunner(
                repositories,
                env={"YOUTUBE_API_KEY": "test-key"},
                fetch_json=always_failing_fetch_json,
            )

            result = runner.run_once()

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.discovered_count, 0)
            self.assertEqual(repositories.candidates.count(), 0)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.status, "failed")
            self.assertGreater(len(run.metadata["errors"]), 0)


def prepare_repositories(root: Path) -> InsightCastRepositories:
    config = load_config(BACKEND_ROOT / "configs" / "local.example.json")
    repositories = InsightCastRepositories(root_dir=root / "storage")
    apply_config(config, repositories)
    return repositories


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


def partially_failing_fetch_json(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    if "googleapis" in url:
        raise RuntimeError("simulated youtube failure")
    return successful_fetch_json(url, params, headers)


def always_failing_fetch_json(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    raise RuntimeError("simulated discovery failure")


def stable_suffix(prefix: str, value: object) -> str:
    text = str(value)
    digest = hex(abs(hash(text)))[2:10]
    return f"{prefix}{digest}"


if __name__ == "__main__":
    unittest.main()
