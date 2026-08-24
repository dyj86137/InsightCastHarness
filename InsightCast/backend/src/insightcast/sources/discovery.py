"""真实内容来源发现接口和平台适配器。"""

from __future__ import annotations

import html
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from insightcast.domain.enums import ContentFormat, SourceType
from insightcast.domain.models import CandidateItem, SearchQuery, Source
from insightcast.integrations.bilibili import BilibiliBackendRouter
from insightcast.platform_cookies import resolve_cookie_header


JsonFetcher = Callable[[str, Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]]
DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS = 8.0


class DiscoveryError(Exception):
    """内容发现阶段的基础异常。"""


class UnsupportedSourceError(DiscoveryError):
    """当前 source type 暂无 discovery adapter。"""


class DiscoverySource(ABC):
    """把 SearchQuery 转换成候选内容的来源适配器。"""

    def __init__(
        self,
        source: Source,
        *,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        self.source = source
        self.fetch_json = fetch_json or fetch_json_url

    @abstractmethod
    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        """执行一次真实来源搜索，并返回规范化候选内容。"""

    def _ensure_matching_query(self, query: SearchQuery) -> None:
        if query.source_id and query.source_id != self.source.id:
            raise DiscoveryError(
                "SearchQuery source_id does not match discovery source: "
                f"{query.source_id} != {self.source.id}"
            )
        if query.source_type and query.source_type != self.source.type:
            raise DiscoveryError(
                "SearchQuery source_type does not match discovery source: "
                f"{query.source_type} != {self.source.type}"
            )

    def _max_results(self, default: int = 10, maximum: int = 50) -> int:
        value = self.source.metadata.get("max_results", default)
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = default
        return max(1, min(count, maximum))


class YouTubeDiscoverySource(DiscoverySource):
    """基于 YouTube Data API v3 的公开视频搜索。"""

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(
        self,
        source: Source,
        *,
        api_key: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        super().__init__(source, fetch_json=fetch_json)
        env_mapping = env if env is not None else os.environ
        key_name = str(source.metadata.get("api_key_env") or "YOUTUBE_API_KEY")
        self.api_key = api_key or str(source.metadata.get("api_key") or "") or env_mapping.get(key_name)
        if not self.api_key:
            raise DiscoveryError(
                "YouTube discovery requires an API key. Set YOUTUBE_API_KEY "
                "or source.metadata.api_key_env."
            )

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        params = {
            "part": "snippet",
            "type": "video",
            "q": query.text,
            "maxResults": self._max_results(),
            "order": self._youtube_order(),
            "key": self.api_key,
        }
        if query.language:
            params["relevanceLanguage"] = query.language

        payload = self.fetch_json(self.SEARCH_URL, params, {})
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise DiscoveryError("YouTube search response field `items` must be a list.")

        durations_by_video_id = self._duration_seconds_by_video_id(items)
        candidates = []
        for item in items:
            candidate = self._candidate_from_item(item, query, durations_by_video_id)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _duration_seconds_by_video_id(
        self,
        items: Sequence[Any],
    ) -> Dict[str, int]:
        video_ids = []
        seen = set()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            item_id = item.get("id") or {}
            if not isinstance(item_id, Mapping):
                continue
            video_id = str(item_id.get("videoId") or "").strip()
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            video_ids.append(video_id)
        if not video_ids:
            return {}

        params = {
            "part": "contentDetails",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        try:
            payload = self.fetch_json(self.VIDEOS_URL, params, {})
        except Exception:
            return {}
        video_items = payload.get("items", [])
        if not isinstance(video_items, list):
            return {}

        durations = {}
        for item in video_items:
            if not isinstance(item, Mapping):
                continue
            video_id = str(item.get("id") or "").strip()
            content_details = item.get("contentDetails") or {}
            if not video_id or not isinstance(content_details, Mapping):
                continue
            duration = parse_iso8601_duration_seconds(content_details.get("duration"))
            if duration is not None:
                durations[video_id] = duration
        return durations

    def _youtube_order(self) -> str:
        order = str(self.source.metadata.get("order") or "relevance").strip()
        allowed = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
        return order if order in allowed else "relevance"

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
        durations_by_video_id: Optional[Mapping[str, int]] = None,
    ) -> Optional[CandidateItem]:
        item_id = item.get("id") or {}
        snippet = item.get("snippet") or {}
        if not isinstance(item_id, Mapping) or not isinstance(snippet, Mapping):
            return None
        video_id = str(item_id.get("videoId") or "").strip()
        if not video_id:
            return None

        title = clean_text(str(snippet.get("title") or ""))
        if not title:
            return None
        url = f"https://www.youtube.com/watch?v={video_id}"
        return CandidateItem(
            id=stable_candidate_id(self.source, video_id, url),
            source_id=self.source.id,
            source_name=self.source.name,
            source_type=self.source.type,
            platform_item_id=video_id,
            title=title,
            description=clean_text(str(snippet.get("description") or "")) or None,
            url=url,
            format=ContentFormat.VIDEO,
            published_at=parse_datetime(snippet.get("publishedAt")),
            duration_seconds=(durations_by_video_id or {}).get(video_id),
            detected_person_ids=query_person_ids(query),
            detected_person_names=query_person_names(query),
            query_id=query.id,
            query_text=query.text,
            raw_metadata=dict(item),
        )


class BilibiliDiscoverySource(DiscoverySource):
    """基于 B站公开 Web 搜索接口的视频搜索。"""

    SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"

    def __init__(
        self,
        source: Source,
        *,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        super().__init__(source, fetch_json=fetch_json)
        self.env = os.environ if env is None else env
        try:
            self.cookie_header = resolve_cookie_header(
                self.env,
                cookie=_optional_metadata_str(source, "cookie"),
                cookie_file=_optional_metadata_str(source, "cookie_file"),
                cookie_env=str(
                    source.metadata.get("cookie_env")
                    or "INSIGHTCAST_BILIBILI_COOKIE"
                ),
                cookie_file_env=str(
                    source.metadata.get("cookie_file_env")
                    or "INSIGHTCAST_BILIBILI_COOKIE_FILE"
                ),
                domains=("bilibili.com",),
            )
        except OSError as exc:
            raise DiscoveryError(f"Bilibili Cookie 加载失败：{exc}") from exc
        self.bilibili_router = BilibiliBackendRouter(
            env=self.env,
            cookie_header=self.cookie_header,
            fetch_json=self.fetch_json,
        )

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        try:
            route = self.bilibili_router.search(
                query.text,
                max_results=self._max_results(),
                order=self._bilibili_order(),
                timeout_seconds=_env_float(
                    self.env,
                    "INSIGHTCAST_DISCOVERY_FETCH_TIMEOUT_SECONDS",
                    DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS,
                ),
            )
        except Exception as exc:
            raise DiscoveryError(str(exc)) from exc

        candidates = []
        for item in route.items:
            candidate = self._candidate_from_item(item, query, backend=route.backend)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def _bilibili_order(self) -> str:
        order = str(self.source.metadata.get("order") or "totalrank").strip()
        allowed = {
            "totalrank",
            "click",
            "pubdate",
            "dm",
            "stow",
            "scores",
        }
        return order if order in allowed else "totalrank"

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
        *,
        backend: str = "bilibili-api",
    ) -> Optional[CandidateItem]:
        bvid = str(item.get("bvid") or "").strip()
        aid = str(item.get("aid") or "").strip()
        platform_item_id = bvid or aid
        if not platform_item_id:
            return None

        title = clean_text(str(item.get("title") or ""))
        if not title:
            return None
        raw_url = str(item.get("arcurl") or "").strip()
        if raw_url.startswith("//"):
            raw_url = f"https:{raw_url}"
        url = raw_url or f"https://www.bilibili.com/video/{platform_item_id}"

        raw_metadata = dict(item)
        raw_metadata["insightcast_bilibili_backend"] = backend
        return CandidateItem(
            id=stable_candidate_id(self.source, platform_item_id, url),
            source_id=self.source.id,
            source_name=self.source.name,
            source_type=self.source.type,
            platform_item_id=platform_item_id,
            title=title,
            description=clean_text(str(item.get("description") or "")) or None,
            url=url,
            format=ContentFormat.VIDEO,
            published_at=parse_datetime(item.get("pubdate")),
            duration_seconds=parse_duration_seconds(item.get("duration")),
            detected_person_ids=query_person_ids(query),
            detected_person_names=query_person_names(query),
            query_id=query.id,
            query_text=query.text,
            raw_metadata=raw_metadata,
        )


class VimeoDiscoverySource(DiscoverySource):
    """基于 Vimeo API 的公开视频搜索。"""

    VIDEOS_URL = "https://api.vimeo.com/videos"

    def __init__(
        self,
        source: Source,
        *,
        access_token: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        super().__init__(source, fetch_json=fetch_json)
        env_mapping = env if env is not None else os.environ
        key_name = str(source.metadata.get("access_token_env") or "VIMEO_ACCESS_TOKEN")
        self.access_token = (
            access_token
            or str(source.metadata.get("access_token") or "")
            or env_mapping.get(key_name)
        )
        if not self.access_token:
            raise DiscoveryError(
                "Vimeo discovery requires an access token. Set VIMEO_ACCESS_TOKEN "
                "or source.metadata.access_token_env."
            )

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        payload = self.fetch_json(
            self._videos_url(),
            {
                "query": query.text,
                "per_page": self._max_results(maximum=100),
                "sort": self._vimeo_sort(),
            },
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/vnd.vimeo.*+json;version=3.4",
            },
        )
        items = payload.get("data", [])
        if not isinstance(items, list):
            raise DiscoveryError("Vimeo response field `data` must be a list.")
        return tuple(
            candidate
            for item in items
            if isinstance(item, Mapping)
            for candidate in (self._candidate_from_item(item, query),)
            if candidate is not None
        )

    def _videos_url(self) -> str:
        scope = str(self.source.metadata.get("search_scope") or "platform").strip()
        if scope == "platform":
            return self.VIDEOS_URL
        if scope not in {"user", "channel", "album"}:
            raise DiscoveryError(
                "Vimeo search_scope must be one of: platform, user, channel, album."
            )
        if not self.source.platform_id:
            raise DiscoveryError(
                f"Vimeo {scope} search requires source.platform_id."
            )
        collection = {"user": "users", "channel": "channels", "album": "albums"}[scope]
        return f"https://api.vimeo.com/{collection}/{self.source.platform_id}/videos"

    def _vimeo_sort(self) -> str:
        value = str(self.source.metadata.get("order") or "relevant").strip()
        allowed = {"relevant", "date", "alphabetical", "plays", "likes", "comments"}
        return value if value in allowed else "relevant"

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
    ) -> Optional[CandidateItem]:
        platform_item_id = _vimeo_video_id(item)
        title = clean_text(str(item.get("name") or ""))
        if not platform_item_id or not title:
            return None
        url = str(item.get("link") or f"https://vimeo.com/{platform_item_id}").strip()
        return _video_candidate(
            self.source,
            query,
            platform_item_id=platform_item_id,
            title=title,
            url=url,
            description=clean_text(str(item.get("description") or "")) or None,
            published_at=parse_datetime(
                item.get("release_time") or item.get("created_time")
            ),
            duration_seconds=parse_duration_seconds(item.get("duration")),
            raw_metadata=item,
        )


class DailymotionDiscoverySource(DiscoverySource):
    """基于 Dailymotion Data API 的公开视频搜索。"""

    VIDEOS_URL = "https://api.dailymotion.com/videos"
    FIELDS = "id,title,description,url,created_time,duration,owner.username"

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        params: Dict[str, Any] = {
            "search": query.text,
            "limit": self._max_results(),
            "fields": self.FIELDS,
        }
        order = str(self.source.metadata.get("order") or "").strip()
        if order:
            params["sort"] = order
        payload = self.fetch_json(self._videos_url(), params, {})
        items = payload.get("list", [])
        if not isinstance(items, list):
            raise DiscoveryError("Dailymotion response field `list` must be a list.")
        return tuple(
            candidate
            for item in items
            if isinstance(item, Mapping)
            for candidate in (self._candidate_from_item(item, query),)
            if candidate is not None
        )

    def _videos_url(self) -> str:
        scope = str(self.source.metadata.get("search_scope") or "platform").strip()
        if scope == "platform":
            return self.VIDEOS_URL
        if scope != "user":
            raise DiscoveryError(
                "Dailymotion search_scope must be one of: platform, user."
            )
        if not self.source.platform_id:
            raise DiscoveryError("Dailymotion user search requires source.platform_id.")
        return f"https://api.dailymotion.com/user/{self.source.platform_id}/videos"

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
    ) -> Optional[CandidateItem]:
        platform_item_id = str(item.get("id") or "").strip()
        title = clean_text(str(item.get("title") or ""))
        if not platform_item_id or not title:
            return None
        url = str(
            item.get("url") or f"https://www.dailymotion.com/video/{platform_item_id}"
        ).strip()
        return _video_candidate(
            self.source,
            query,
            platform_item_id=platform_item_id,
            title=title,
            url=url,
            description=clean_text(str(item.get("description") or "")) or None,
            published_at=parse_datetime(item.get("created_time")),
            duration_seconds=parse_duration_seconds(item.get("duration")),
            raw_metadata=item,
        )


class TwitchDiscoverySource(DiscoverySource):
    """基于 Twitch Helix API 按频道发现公开 VOD。"""

    VIDEOS_URL = "https://api.twitch.tv/helix/videos"

    def __init__(
        self,
        source: Source,
        *,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        super().__init__(source, fetch_json=fetch_json)
        env_mapping = env if env is not None else os.environ
        client_id_env = str(source.metadata.get("client_id_env") or "TWITCH_CLIENT_ID")
        access_token_env = str(
            source.metadata.get("access_token_env") or "TWITCH_ACCESS_TOKEN"
        )
        self.client_id = (
            client_id
            or str(source.metadata.get("client_id") or "")
            or env_mapping.get(client_id_env)
        )
        self.access_token = (
            access_token
            or str(source.metadata.get("access_token") or "")
            or env_mapping.get(access_token_env)
        )
        if not self.client_id or not self.access_token:
            raise DiscoveryError(
                "Twitch discovery requires a client ID and access token. Set "
                "TWITCH_CLIENT_ID and TWITCH_ACCESS_TOKEN, or configure source metadata."
            )
        if not source.platform_id:
            raise DiscoveryError(
                "Twitch discovery requires source.platform_id to be a broadcaster ID; "
                "the Helix API has no global VOD keyword-search endpoint."
            )

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        payload = self.fetch_json(
            self.VIDEOS_URL,
            {
                "user_id": self.source.platform_id,
                "first": self._max_results(maximum=100),
                "type": str(self.source.metadata.get("video_type") or "archive"),
            },
            {
                "Client-ID": self.client_id,
                "Authorization": f"Bearer {self.access_token}",
            },
        )
        items = payload.get("data", [])
        if not isinstance(items, list):
            raise DiscoveryError("Twitch response field `data` must be a list.")
        return tuple(
            candidate
            for item in items
            if isinstance(item, Mapping) and _twitch_query_matches(item, query, self.source)
            for candidate in (self._candidate_from_item(item, query),)
            if candidate is not None
        )

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
    ) -> Optional[CandidateItem]:
        platform_item_id = str(item.get("id") or "").strip()
        title = clean_text(str(item.get("title") or ""))
        if not platform_item_id or not title:
            return None
        url = str(item.get("url") or "").strip()
        if not url:
            return None
        return _video_candidate(
            self.source,
            query,
            platform_item_id=platform_item_id,
            title=title,
            url=url,
            description=clean_text(str(item.get("description") or "")) or None,
            published_at=parse_datetime(item.get("published_at") or item.get("created_at")),
            duration_seconds=parse_twitch_duration_seconds(item.get("duration")),
            raw_metadata=item,
        )


class PeerTubeDiscoverySource(DiscoverySource):
    """基于指定 PeerTube 实例 API 的公开视频搜索。"""

    def __init__(
        self,
        source: Source,
        *,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        super().__init__(source, fetch_json=fetch_json)
        self.api_base_url = _peertube_api_base_url(source)

    def discover(self, query: SearchQuery) -> Tuple[CandidateItem, ...]:
        self._ensure_matching_query(query)
        payload = self.fetch_json(
            f"{self.api_base_url}/search/videos",
            {
                "search": query.text,
                "count": self._max_results(),
                "start": 0,
                "sort": str(self.source.metadata.get("order") or "-publishedAt"),
            },
            {},
        )
        items = payload.get("data", [])
        if not isinstance(items, list):
            raise DiscoveryError("PeerTube response field `data` must be a list.")
        return tuple(
            candidate
            for item in items
            if isinstance(item, Mapping)
            for candidate in (self._candidate_from_item(item, query),)
            if candidate is not None
        )

    def _candidate_from_item(
        self,
        item: Mapping[str, Any],
        query: SearchQuery,
    ) -> Optional[CandidateItem]:
        platform_item_id = str(
            item.get("uuid") or item.get("shortUUID") or item.get("id") or ""
        ).strip()
        title = clean_text(str(item.get("name") or ""))
        if not platform_item_id or not title:
            return None
        url = str(item.get("url") or "").strip()
        if not url:
            return None
        return _video_candidate(
            self.source,
            query,
            platform_item_id=platform_item_id,
            title=title,
            url=url,
            description=clean_text(str(item.get("description") or "")) or None,
            published_at=parse_datetime(item.get("publishedAt") or item.get("createdAt")),
            duration_seconds=parse_duration_seconds(item.get("duration")),
            raw_metadata=item,
        )


def create_discovery_source(
    source: Source,
    *,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[JsonFetcher] = None,
) -> DiscoverySource:
    """根据 Source 创建真实 discovery adapter。"""

    if source.type == SourceType.YOUTUBE:
        return YouTubeDiscoverySource(source, env=env, fetch_json=fetch_json)
    if source.type == SourceType.BILIBILI:
        return BilibiliDiscoverySource(source, env=env, fetch_json=fetch_json)
    if source.type == SourceType.VIMEO:
        return VimeoDiscoverySource(source, env=env, fetch_json=fetch_json)
    if source.type == SourceType.DAILYMOTION:
        return DailymotionDiscoverySource(source, fetch_json=fetch_json)
    if source.type == SourceType.TWITCH:
        return TwitchDiscoverySource(source, env=env, fetch_json=fetch_json)
    if source.type == SourceType.PEERTUBE:
        return PeerTubeDiscoverySource(source, fetch_json=fetch_json)
    raise UnsupportedSourceError(f"Unsupported discovery source type: {source.type}")


def discover_candidates(
    queries: Sequence[SearchQuery],
    sources: Sequence[Source],
    *,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[JsonFetcher] = None,
) -> Tuple[CandidateItem, ...]:
    """对一批 SearchQuery 执行真实平台发现。"""

    source_by_id = {source.id: source for source in sources}
    adapters: Dict[str, DiscoverySource] = {}
    candidates = []
    for query in queries:
        if not query.source_id:
            raise DiscoveryError(f"SearchQuery missing source_id: {query.id}")
        source = source_by_id.get(query.source_id)
        if source is None:
            raise DiscoveryError(f"SearchQuery source not found: {query.source_id}")
        adapter = adapters.get(source.id)
        if adapter is None:
            adapter = create_discovery_source(source, env=env, fetch_json=fetch_json)
            adapters[source.id] = adapter
        candidates.extend(adapter.discover(query))
    return tuple(candidates)


def fetch_json_url(
    url: str,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    timeout_seconds: Optional[float] = None,
) -> Mapping[str, Any]:
    """使用标准库发起 GET 请求并解析 JSON。"""

    query = urlencode({key: value for key, value in params.items() if value is not None})
    request_url = f"{url}?{query}" if query else url
    request = Request(request_url, headers=dict(headers))
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else _env_float(
            os.environ,
            "INSIGHTCAST_DISCOVERY_FETCH_TIMEOUT_SECONDS",
            DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS,
        )
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"Discovery response is not valid JSON: {url}") from exc
    if not isinstance(payload, Mapping):
        raise DiscoveryError(f"Discovery response root must be an object: {url}")
    return payload


def stable_candidate_id(source: Source, platform_item_id: str, url: str) -> str:
    raw = "|".join((source.id, platform_item_id, url))
    digest = sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"candidate_{digest}"


def _video_candidate(
    source: Source,
    query: SearchQuery,
    *,
    platform_item_id: str,
    title: str,
    url: str,
    description: Optional[str],
    published_at: Optional[datetime],
    duration_seconds: Optional[int],
    raw_metadata: Mapping[str, Any],
) -> CandidateItem:
    return CandidateItem(
        id=stable_candidate_id(source, platform_item_id, url),
        source_id=source.id,
        source_name=source.name,
        source_type=source.type,
        platform_item_id=platform_item_id,
        title=title,
        description=description,
        url=url,
        format=ContentFormat.VIDEO,
        published_at=published_at,
        duration_seconds=duration_seconds,
        detected_person_ids=query_person_ids(query),
        detected_person_names=query_person_names(query),
        query_id=query.id,
        query_text=query.text,
        raw_metadata=dict(raw_metadata),
    )


def _vimeo_video_id(item: Mapping[str, Any]) -> str:
    uri = str(item.get("uri") or "").strip().rstrip("/")
    if uri.startswith("/videos/"):
        return uri.rsplit("/", 1)[-1]
    link = str(item.get("link") or "").strip().rstrip("/")
    if "/" in link:
        return link.rsplit("/", 1)[-1]
    return ""


def _twitch_query_matches(
    item: Mapping[str, Any],
    query: SearchQuery,
    source: Source,
) -> bool:
    mode = str(source.metadata.get("query_match_mode") or "all").strip().lower()
    if mode == "none":
        return True
    if mode not in {"all", "any"}:
        raise DiscoveryError("Twitch query_match_mode must be one of: all, any, none.")
    terms = [term for term in re.findall(r"[\w-]+", query.text.lower()) if len(term) > 1]
    if not terms:
        return True
    text = " ".join(
        str(item.get(key) or "") for key in ("title", "description", "user_name")
    ).lower()
    return all(term in text for term in terms) if mode == "all" else any(
        term in text for term in terms
    )


def _peertube_api_base_url(source: Source) -> str:
    raw = str(source.metadata.get("api_base_url") or source.url or "").strip().rstrip("/")
    if not raw:
        raise DiscoveryError(
            "PeerTube discovery requires source.url or source.metadata.api_base_url."
        )
    if raw.endswith("/api/v1"):
        return raw
    return f"{raw}/api/v1"


def query_person_ids(query: SearchQuery) -> Tuple[str, ...]:
    if not query.person_id:
        return ()
    return (query.person_id,)


def query_person_names(query: SearchQuery) -> Tuple[str, ...]:
    value = query.metadata.get("person_name")
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_duration_seconds(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = int(value)
        return seconds if seconds >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def parse_twitch_duration_seconds(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return parse_duration_seconds(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)


def parse_iso8601_duration_seconds(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(
        r"P(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?"
        r"(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        text,
    )
    if not match or "T" not in text:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds or 0)


def clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = html.unescape(text)
    return " ".join(text.split()).strip()


def _optional_metadata_str(source: Source, key: str) -> Optional[str]:
    value = source.metadata.get(key)
    if value is None or value == "":
        return None
    return str(value)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DiscoveryError(f"Environment variable {key} must be a float.") from exc
    if parsed <= 0:
        raise DiscoveryError(f"Environment variable {key} must be positive.")
    return parsed


__all__ = [
    "BilibiliDiscoverySource",
    "DailymotionDiscoverySource",
    "DEFAULT_DISCOVERY_FETCH_TIMEOUT_SECONDS",
    "DiscoveryError",
    "DiscoverySource",
    "JsonFetcher",
    "UnsupportedSourceError",
    "PeerTubeDiscoverySource",
    "TwitchDiscoverySource",
    "VimeoDiscoverySource",
    "YouTubeDiscoverySource",
    "clean_text",
    "create_discovery_source",
    "discover_candidates",
    "fetch_json_url",
    "parse_datetime",
    "parse_duration_seconds",
    "parse_iso8601_duration_seconds",
    "parse_twitch_duration_seconds",
    "stable_candidate_id",
]
