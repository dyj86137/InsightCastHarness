"""Content source adapters and query generation."""

from insightcast.sources.candidate_ingestion import (
    CandidateIngestResult,
    candidate_identity_keys,
    candidate_video_identity_keys,
    ingest_candidates,
    merge_candidate,
    normalize_candidate_url,
    video_identity_from_url,
)
from insightcast.sources.discovery import (
    BilibiliDiscoverySource,
    DailymotionDiscoverySource,
    DiscoveryError,
    DiscoverySource,
    PeerTubeDiscoverySource,
    TwitchDiscoverySource,
    UnsupportedSourceError,
    VimeoDiscoverySource,
    YouTubeDiscoverySource,
    create_discovery_source,
    discover_candidates,
    parse_iso8601_duration_seconds,
    parse_twitch_duration_seconds,
)
from insightcast.sources.query_generator import (
    CHINESE_QUERY_SUFFIXES,
    ENGLISH_QUERY_SUFFIXES,
    generate_and_save_search_queries,
    generate_search_queries,
    save_search_queries,
)


__all__ = [
    "BilibiliDiscoverySource",
    "DailymotionDiscoverySource",
    "CandidateIngestResult",
    "CHINESE_QUERY_SUFFIXES",
    "DiscoveryError",
    "DiscoverySource",
    "ENGLISH_QUERY_SUFFIXES",
    "PeerTubeDiscoverySource",
    "TwitchDiscoverySource",
    "UnsupportedSourceError",
    "VimeoDiscoverySource",
    "YouTubeDiscoverySource",
    "create_discovery_source",
    "discover_candidates",
    "generate_and_save_search_queries",
    "generate_search_queries",
    "candidate_identity_keys",
    "candidate_video_identity_keys",
    "ingest_candidates",
    "merge_candidate",
    "normalize_candidate_url",
    "parse_iso8601_duration_seconds",
    "parse_twitch_duration_seconds",
    "save_search_queries",
    "video_identity_from_url",
]
