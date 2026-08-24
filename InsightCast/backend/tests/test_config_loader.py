from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.config import apply_config, load_config, load_config_dir, load_env_file
from insightcast.domain.enums import Industry, PushChannel, SourceType
from insightcast.storage.repositories import InsightCastRepositories


class ConfigLoaderTests(unittest.TestCase):
    def test_load_full_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "insightcast.json"
            config_path.write_text(
                json.dumps(
                    {
                        "industries": [
                            {
                                "id": "industry_ai",
                                "name": "AI",
                                "slug": "ai",
                                "keywords": ["LLM"],
                            }
                        ],
                        "people": [
                            {
                                "id": "person_ada",
                                "name": "Ada Lovelace",
                                "aliases": ["Augusta Ada King"],
                                "companies": ["Analytical Engines"],
                                "industries": ["ai"],
                                "importance": 0.8,
                            }
                        ],
                        "sources": [
                            {
                                "id": "source_show",
                                "name": "Example Show",
                                "type": "podcast_rss",
                                "url": "https://example.com/feed",
                                "authority": 0.7,
                            }
                        ],
                        "interests": [
                            {
                                "id": "interest_default",
                                "industries": ["ai"],
                                "people": ["person_ada"],
                                "push_channels": ["markdown"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.industries[0].slug, Industry.AI)
        self.assertEqual(config.people[0].search_names, ("Ada Lovelace", "Augusta Ada King"))
        self.assertEqual(config.sources[0].type, SourceType.PODCAST_RSS)
        self.assertEqual(config.user_interests[0].push_channels, (PushChannel.MARKDOWN,))

    def test_apply_config_upserts_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(BACKEND_ROOT / "configs" / "local.example.json")
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")

            result = apply_config(config, repositories)
            second_result = apply_config(config, repositories)

            self.assertEqual(result.people, len(config.people))
            self.assertEqual(second_result.people, len(config.people))
            self.assertEqual(repositories.people.count(), len(config.people))
            self.assertEqual(repositories.sources.count(), len(config.sources))
            self.assertIsNotNone(repositories.people.find_by_name("黄仁勋"))

    def test_load_split_directory_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "people.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "person_split",
                            "name": "Split Person",
                            "industries": ["ai"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "sources.json").write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "source_split",
                                "name": "Split Source",
                                "type": "rss",
                                "url": "https://example.com/rss",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            config = load_config_dir(root)

        self.assertEqual(config.people[0].id, "person_split")
        self.assertEqual(config.sources[0].id, "source_split")


class EnvFileTests(unittest.TestCase):
    def test_load_env_file_loads_values_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# local integration settings",
                        "INSIGHTCAST_TEST_KEY=value # comment",
                        'INSIGHTCAST_TEST_QUOTED="hello # world"',
                        "INSIGHTCAST_TEST_EMPTY=",
                        "INSIGHTCAST_TEST_EXISTING=from-file",
                        "export INSIGHTCAST_TEST_EXPORTED=yes",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"INSIGHTCAST_TEST_EXISTING": "from-shell"},
                clear=False,
            ):
                loaded = load_env_file(env_path)

                self.assertEqual(os.environ["INSIGHTCAST_TEST_KEY"], "value")
                self.assertEqual(os.environ["INSIGHTCAST_TEST_QUOTED"], "hello # world")
                self.assertEqual(os.environ["INSIGHTCAST_TEST_EMPTY"], "")
                self.assertEqual(os.environ["INSIGHTCAST_TEST_EXISTING"], "from-shell")
                self.assertEqual(os.environ["INSIGHTCAST_TEST_EXPORTED"], "yes")
                self.assertEqual(loaded["INSIGHTCAST_TEST_EXISTING"], "from-shell")


if __name__ == "__main__":
    unittest.main()
