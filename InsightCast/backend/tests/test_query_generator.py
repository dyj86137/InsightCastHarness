from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.config import apply_config, load_config
from insightcast.domain.enums import Industry, SourceType
from insightcast.domain.models import Person, Source, UserInterest
from insightcast.sources import (
    generate_and_save_search_queries,
    generate_search_queries,
    save_search_queries,
)
from insightcast.storage.repositories import InsightCastRepositories, SearchQueryRepository


class QueryGeneratorTests(unittest.TestCase):
    def test_generates_language_specific_platform_queries(self) -> None:
        config = load_config(BACKEND_ROOT / "configs" / "local.example.json")

        queries = generate_search_queries(
            config.people,
            config.sources,
            config.user_interests,
        )

        texts_by_source = {}
        for query in queries:
            texts_by_source.setdefault(query.source_id, set()).add(query.text)

        youtube_texts = texts_by_source["source_youtube_search"]
        bilibili_texts = texts_by_source["source_bilibili_search"]

        self.assertIn("Jensen Huang interview", youtube_texts)
        self.assertIn("NVIDIA CEO interview", youtube_texts)
        self.assertNotIn("黄仁勋 interview", youtube_texts)
        self.assertIn("黄仁勋 访谈", bilibili_texts)
        self.assertIn("Jensen Huang 访谈", bilibili_texts)
        self.assertIn("NVIDIA CEO 访谈", bilibili_texts)

        youtube_query = next(
            query for query in queries if query.text == "Jensen Huang interview"
        )
        self.assertEqual(youtube_query.language, "en")
        self.assertEqual(youtube_query.source_type, SourceType.YOUTUBE)
        self.assertEqual(youtube_query.metadata["search_scope"], "platform")

        bilibili_query = next(query for query in queries if query.text == "黄仁勋 访谈")
        self.assertEqual(bilibili_query.language, "zh")
        self.assertEqual(bilibili_query.source_type, SourceType.BILIBILI)
        self.assertEqual(bilibili_query.metadata["search_scope"], "platform")

    def test_filters_people_by_interest_and_source_industry(self) -> None:
        people = [
            Person(
                id="person_ai",
                name="AI Person",
                companies=("AI Co",),
                title="CEO",
                industries=(Industry.AI,),
            ),
            Person(
                id="person_retail",
                name="Retail Person",
                companies=("Retail Co",),
                title="CEO",
                industries=(Industry.RETAIL,),
            ),
        ]
        sources = [
            Source(
                id="source_ai",
                name="AI Search",
                type=SourceType.YOUTUBE,
                languages=("en",),
                industries=(Industry.AI,),
                metadata={"search_scope": "platform"},
            )
        ]
        interests = [
            UserInterest(
                id="interest_ai",
                industries=(Industry.AI,),
            )
        ]

        queries = generate_search_queries(people, sources, interests)

        self.assertEqual({query.person_id for query in queries}, {"person_ai"})
        self.assertIn("AI Co CEO interview", {query.text for query in queries})
        self.assertNotIn("Retail Co CEO interview", {query.text for query in queries})

    def test_generated_queries_have_stable_ids_and_upsert_cleanly(self) -> None:
        config = load_config(BACKEND_ROOT / "configs" / "local.example.json")
        queries = generate_search_queries(
            config.people,
            config.sources,
            config.user_interests,
        )

        self.assertEqual(len({query.id for query in queries}), len(queries))

        with tempfile.TemporaryDirectory() as tmp:
            repository = SearchQueryRepository(root_dir=Path(tmp) / "storage")

            first_count = save_search_queries(repository, queries)
            second_count = save_search_queries(repository, queries)

            self.assertEqual(first_count, len(queries))
            self.assertEqual(second_count, len(queries))
            self.assertEqual(repository.count(), len(queries))

    def test_generate_and_save_queries_from_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(BACKEND_ROOT / "configs" / "local.example.json")
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            apply_config(config, repositories)

            queries = generate_and_save_search_queries(repositories)
            generated_count = repositories.search_queries.count()
            generate_and_save_search_queries(repositories)

            self.assertGreater(generated_count, 0)
            self.assertEqual(repositories.search_queries.count(), generated_count)
            self.assertEqual(len(queries), generated_count)


if __name__ == "__main__":
    unittest.main()
