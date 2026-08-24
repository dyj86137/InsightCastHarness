from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.domain.enums import SourceType
from insightcast.domain.models import CandidateItem, SearchQuery, Source
from insightcast.sources import (
    BilibiliDiscoverySource,
    DailymotionDiscoverySource,
    PeerTubeDiscoverySource,
    TwitchDiscoverySource,
    VimeoDiscoverySource,
    YouTubeDiscoverySource,
    create_discovery_source,
    discover_candidates,
    parse_iso8601_duration_seconds,
    parse_twitch_duration_seconds,
)
from insightcast.transcript.fetcher import is_supported_platform_candidate


class DiscoverySourceTests(unittest.TestCase):
    def test_youtube_discovery_maps_search_response_to_candidates(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            if url.endswith("/videos"):
                return {
                    "items": [
                        {
                            "id": "abc123",
                            "contentDetails": {"duration": "PT2M40S"},
                        }
                    ]
                }
            return {
                "items": [
                    {
                        "id": {"videoId": "abc123"},
                        "snippet": {
                            "title": "Sam Altman on AI Agents",
                            "description": "A long-form interview.",
                            "publishedAt": "2026-07-09T12:00:00Z",
                            "channelTitle": "Example Channel",
                        },
                    }
                ]
            }

        source = Source(
            id="source_youtube_search",
            name="YouTube Search",
            type=SourceType.YOUTUBE,
            languages=("en",),
            metadata={"search_scope": "platform", "max_results": 5},
        )
        query = SearchQuery(
            id="query_youtube",
            text="Sam Altman interview",
            person_id="person_sam_altman",
            source_id=source.id,
            source_type=source.type,
            language="en",
            metadata={"person_name": "Sam Altman"},
        )

        adapter = YouTubeDiscoverySource(
            source,
            api_key="test-key",
            fetch_json=fetch_json,
        )
        candidates = adapter.discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].platform_item_id, "abc123")
        self.assertEqual(candidates[0].title, "Sam Altman on AI Agents")
        self.assertEqual(str(candidates[0].url), "https://www.youtube.com/watch?v=abc123")
        self.assertEqual(candidates[0].duration_seconds, 160)
        self.assertEqual(candidates[0].query_text, "Sam Altman interview")
        self.assertEqual(candidates[0].detected_person_names, ("Sam Altman",))
        self.assertEqual(calls[0][1]["key"], "test-key")
        self.assertEqual(calls[0][1]["q"], "Sam Altman interview")
        self.assertTrue(calls[1][0].endswith("/videos"))
        self.assertEqual(calls[1][1]["part"], "contentDetails")
        self.assertEqual(calls[1][1]["id"], "abc123")

    def test_parse_iso8601_duration_seconds(self) -> None:
        self.assertEqual(parse_iso8601_duration_seconds("PT2M40S"), 160)
        self.assertEqual(parse_iso8601_duration_seconds("PT1H02M03S"), 3723)
        self.assertEqual(parse_iso8601_duration_seconds("PT45S"), 45)
        self.assertEqual(parse_iso8601_duration_seconds("PT10M"), 600)
        self.assertIsNone(parse_iso8601_duration_seconds(""))
        self.assertIsNone(parse_iso8601_duration_seconds("not-a-duration"))

    def test_bilibili_discovery_maps_search_response_to_candidates(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            return {
                "code": 0,
                "data": {
                    "result": [
                        {
                            "bvid": "BV1xx411c7mD",
                            "title": "<em class=\"keyword\">黄仁勋</em> 访谈",
                            "description": "英伟达 CEO 深度对话",
                            "pubdate": 1783608000,
                            "duration": "12:34",
                            "arcurl": "https://www.bilibili.com/video/BV1xx411c7mD",
                        },
                        {
                            "bvid": "BV1yy411c7mE",
                            "title": "第二条结果",
                            "duration": "01:02:03",
                        },
                    ]
                },
            }

        source = Source(
            id="source_bilibili_search",
            name="Bilibili Search",
            type=SourceType.BILIBILI,
            languages=("zh",),
            metadata={"search_scope": "platform", "max_results": 1, "order": "pubdate"},
        )
        query = SearchQuery(
            id="query_bilibili",
            text="黄仁勋 访谈",
            person_id="person_jensen_huang",
            source_id=source.id,
            source_type=source.type,
            language="zh",
            metadata={"person_name": "Jensen Huang"},
        )

        adapter = BilibiliDiscoverySource(
            source,
            env={"INSIGHTCAST_BILIBILI_COOKIE": "SESSDATA=abc; bili_jct=def"},
            fetch_json=fetch_json,
        )
        candidates = adapter.discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].platform_item_id, "BV1xx411c7mD")
        self.assertEqual(candidates[0].title, "黄仁勋 访谈")
        self.assertEqual(candidates[0].description, "英伟达 CEO 深度对话")
        self.assertEqual(candidates[0].duration_seconds, 754)
        self.assertEqual(candidates[0].source_type, SourceType.BILIBILI)
        self.assertEqual(calls[0][2]["Cookie"], "SESSDATA=abc; bili_jct=def")

    def test_vimeo_discovery_maps_search_response_to_candidates(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            return {
                "data": [
                    {
                        "uri": "/videos/12345",
                        "name": "Founder interview",
                        "description": "A long-form conversation.",
                        "link": "https://vimeo.com/12345",
                        "release_time": "2026-08-01T09:30:00+00:00",
                        "duration": 3872,
                    }
                ]
            }

        source = Source(
            id="source_vimeo",
            name="Vimeo",
            type=SourceType.VIMEO,
            metadata={"max_results": 5},
        )
        query = SearchQuery(
            id="query_vimeo",
            text="Founder interview",
            source_id=source.id,
            source_type=source.type,
        )

        candidates = VimeoDiscoverySource(
            source,
            access_token="test-token",
            fetch_json=fetch_json,
        ).discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].platform_item_id, "12345")
        self.assertEqual(str(candidates[0].url), "https://vimeo.com/12345")
        self.assertEqual(candidates[0].duration_seconds, 3872)
        self.assertEqual(calls[0][0], "https://api.vimeo.com/videos")
        self.assertEqual(calls[0][1]["query"], "Founder interview")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer test-token")

    def test_dailymotion_discovery_maps_search_response_to_candidates(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            return {
                "list": [
                    {
                        "id": "x9abcde",
                        "title": "Founder <b>interview</b>",
                        "description": "Company strategy.",
                        "url": "https://www.dailymotion.com/video/x9abcde",
                        "created_time": 1785600000,
                        "duration": 1240,
                    }
                ]
            }

        source = Source(
            id="source_dailymotion",
            name="Dailymotion",
            type=SourceType.DAILYMOTION,
            metadata={"max_results": 5},
        )
        query = SearchQuery(
            id="query_dailymotion",
            text="founder interview",
            source_id=source.id,
            source_type=source.type,
        )

        candidates = DailymotionDiscoverySource(source, fetch_json=fetch_json).discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].title, "Founder interview")
        self.assertEqual(candidates[0].duration_seconds, 1240)
        self.assertEqual(calls[0][0], "https://api.dailymotion.com/videos")
        self.assertEqual(calls[0][1]["search"], "founder interview")

    def test_twitch_discovery_reads_configured_broadcaster_vods(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            return {
                "data": [
                    {
                        "id": "v123",
                        "title": "Sam Altman interview",
                        "description": "Founder conversation",
                        "url": "https://www.twitch.tv/videos/v123",
                        "published_at": "2026-08-01T10:00:00Z",
                        "duration": "1h2m3s",
                    },
                    {
                        "id": "v124",
                        "title": "Unrelated stream",
                        "url": "https://www.twitch.tv/videos/v124",
                        "duration": "2h",
                    },
                ]
            }

        source = Source(
            id="source_twitch",
            name="Twitch VOD",
            type=SourceType.TWITCH,
            platform_id="98765",
            metadata={"max_results": 5, "query_match_mode": "all"},
        )
        query = SearchQuery(
            id="query_twitch",
            text="Sam Altman interview",
            source_id=source.id,
            source_type=source.type,
        )

        candidates = TwitchDiscoverySource(
            source,
            client_id="test-client",
            access_token="test-token",
            fetch_json=fetch_json,
        ).discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].platform_item_id, "v123")
        self.assertEqual(candidates[0].duration_seconds, 3723)
        self.assertEqual(calls[0][0], "https://api.twitch.tv/helix/videos")
        self.assertEqual(calls[0][1]["user_id"], "98765")
        self.assertEqual(calls[0][2]["Client-ID"], "test-client")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer test-token")

    def test_peertube_discovery_maps_instance_search_response_to_candidates(self) -> None:
        calls = []

        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            calls.append((url, dict(params), dict(headers)))
            return {
                "data": [
                    {
                        "uuid": "e0d30a1a-1234-5678-9abc-def012345678",
                        "name": "Open source founder interview",
                        "description": "Building a company in public.",
                        "url": "https://video.example/videos/watch/e0d30a1a-1234-5678-9abc-def012345678",
                        "publishedAt": "2026-08-01T11:00:00Z",
                        "duration": 860,
                    }
                ]
            }

        source = Source(
            id="source_peertube",
            name="PeerTube",
            type=SourceType.PEERTUBE,
            url="https://video.example",
            metadata={"max_results": 5},
        )
        query = SearchQuery(
            id="query_peertube",
            text="founder interview",
            source_id=source.id,
            source_type=source.type,
        )

        candidates = PeerTubeDiscoverySource(source, fetch_json=fetch_json).discover(query)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].platform_item_id, "e0d30a1a-1234-5678-9abc-def012345678")
        self.assertEqual(candidates[0].duration_seconds, 860)
        self.assertEqual(calls[0][0], "https://video.example/api/v1/search/videos")
        self.assertEqual(calls[0][1]["search"], "founder interview")

    def test_parse_twitch_duration_seconds(self) -> None:
        self.assertEqual(parse_twitch_duration_seconds("1h2m3s"), 3723)
        self.assertEqual(parse_twitch_duration_seconds("45m"), 2700)
        self.assertEqual(parse_twitch_duration_seconds("15s"), 15)
        self.assertIsNone(parse_twitch_duration_seconds(""))
        self.assertIsNone(parse_twitch_duration_seconds("about an hour"))

    def test_new_video_platform_candidates_are_supported_by_ytdlp_fallback(self) -> None:
        candidates = [
            CandidateItem(
                id=f"candidate_{source_type.value}",
                source_type=source_type,
                title=f"{source_type.value} interview",
                url=url,
            )
            for source_type, url in (
                (SourceType.VIMEO, "https://vimeo.com/12345"),
                (SourceType.DAILYMOTION, "https://www.dailymotion.com/video/x9abcde"),
                (SourceType.TWITCH, "https://www.twitch.tv/videos/12345"),
                (SourceType.PEERTUBE, "https://video.example/videos/watch/12345"),
            )
        ]

        self.assertTrue(all(is_supported_platform_candidate(item) for item in candidates))

    def test_discovery_factory_supports_new_video_platforms(self) -> None:
        sources = (
            Source(id="vimeo", name="Vimeo", type=SourceType.VIMEO),
            Source(id="dailymotion", name="Dailymotion", type=SourceType.DAILYMOTION),
            Source(
                id="twitch",
                name="Twitch",
                type=SourceType.TWITCH,
                platform_id="98765",
            ),
            Source(
                id="peertube",
                name="PeerTube",
                type=SourceType.PEERTUBE,
                url="https://video.example",
            ),
        )
        expected_classes = (
            VimeoDiscoverySource,
            DailymotionDiscoverySource,
            TwitchDiscoverySource,
            PeerTubeDiscoverySource,
        )

        adapters = tuple(
            create_discovery_source(
                source,
                env={
                    "VIMEO_ACCESS_TOKEN": "test-vimeo-token",
                    "TWITCH_CLIENT_ID": "test-twitch-client",
                    "TWITCH_ACCESS_TOKEN": "test-twitch-token",
                },
            )
            for source in sources
        )

        self.assertEqual(tuple(type(adapter) for adapter in adapters), expected_classes)

    def test_discover_candidates_dispatches_by_query_source(self) -> None:
        def fetch_json(
            url: str,
            params: Mapping[str, Any],
            headers: Mapping[str, str],
        ) -> Mapping[str, Any]:
            if url.endswith("/videos"):
                return {
                    "items": [
                        {
                            "id": "abc123",
                            "contentDetails": {"duration": "PT1M"},
                        }
                    ]
                }
            if "googleapis" in url:
                return {
                    "items": [
                        {
                            "id": {"videoId": "abc123"},
                            "snippet": {"title": "Video", "publishedAt": "2026-07-09T12:00:00Z"},
                        }
                    ]
                }
            return {
                "code": 0,
                "data": {
                    "result": [
                        {
                            "bvid": "BV1xx411c7mD",
                            "title": "视频",
                            "duration": "1:00",
                        }
                    ]
                },
            }

        youtube = Source(
            id="source_youtube",
            name="YouTube",
            type=SourceType.YOUTUBE,
        )
        bilibili = Source(
            id="source_bilibili",
            name="Bilibili",
            type=SourceType.BILIBILI,
        )
        queries = [
            SearchQuery(
                id="query_youtube",
                text="Sam Altman interview",
                source_id=youtube.id,
                source_type=youtube.type,
            ),
            SearchQuery(
                id="query_bilibili",
                text="黄仁勋 访谈",
                source_id=bilibili.id,
                source_type=bilibili.type,
            ),
        ]

        candidates = discover_candidates(
            queries,
            [youtube, bilibili],
            env={"YOUTUBE_API_KEY": "test-key"},
            fetch_json=fetch_json,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate.source_id for candidate in candidates}, {youtube.id, bilibili.id})


if __name__ == "__main__":
    unittest.main()
