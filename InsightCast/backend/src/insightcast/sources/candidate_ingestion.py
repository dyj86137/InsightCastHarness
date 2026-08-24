"""发现结果入库和第一层候选去重。"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field

from insightcast.domain.models import (
    CandidateItem,
    DomainModel,
    dedupe_non_empty,
    utc_now,
)
from insightcast.storage.repositories import CandidateRepository


_VIDEO_URL_PATTERN = re.compile(r"https?://[^\s<>\"'，。；：！？、]+", re.IGNORECASE)
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}
_BILIBILI_HOSTS = {"bilibili.com", "m.bilibili.com"}
_VIDEO_METADATA_KEYS = (
    "description",
    "desc",
    "original_url",
    "source_url",
    "webpage_url",
    "url",
    "arcurl",
)


class CandidateIngestResult(DomainModel):
    """候选内容入库后的统计结果。"""

    discovered_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    duplicate_count: int = 0
    candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)


def ingest_candidates(
    repository: CandidateRepository,
    candidates: Iterable[CandidateItem],
    *,
    update_existing: bool = True,
) -> CandidateIngestResult:
    """批量保存候选内容，并按平台 ID 和 URL 做第一层去重。"""

    incoming = tuple(candidates)
    index = CandidateIndex(repository.list())
    created_count = 0
    updated_count = 0
    duplicate_count = 0
    candidate_ids: List[str] = []

    for candidate in incoming:
        existing = index.find(candidate)
        if existing is None:
            saved = repository.create(candidate)
            index.add(saved)
            created_count += 1
            candidate_ids.append(saved.id)
            continue

        duplicate_count += 1
        if update_existing:
            saved = repository.save(merge_candidate(existing, candidate))
            index.add(saved)
            updated_count += 1
            candidate_ids.append(saved.id)
        else:
            candidate_ids.append(existing.id)

    return CandidateIngestResult(
        discovered_count=len(incoming),
        created_count=created_count,
        updated_count=updated_count,
        duplicate_count=duplicate_count,
        candidate_ids=tuple(dedupe_non_empty(candidate_ids)),
    )


class CandidateIndex:
    """候选内容的轻量内存索引。"""

    def __init__(self, candidates: Iterable[CandidateItem]) -> None:
        self._items: Dict[str, CandidateItem] = {}
        for candidate in candidates:
            self.add(candidate)

    def add(self, candidate: CandidateItem) -> None:
        for key in candidate_identity_keys(candidate):
            self._items[key] = candidate

    def find(self, candidate: CandidateItem) -> Optional[CandidateItem]:
        for key in candidate_identity_keys(candidate):
            existing = self._items.get(key)
            if existing is not None:
                return existing
        return None


def merge_candidate(existing: CandidateItem, incoming: CandidateItem) -> CandidateItem:
    """合并重复候选，保留已有审核状态和创建时间。"""

    raw_metadata = dict(existing.raw_metadata)
    raw_metadata.update(incoming.raw_metadata)
    return incoming.clone(
        id=existing.id,
        canonical_url=incoming.canonical_url or existing.canonical_url,
        status=existing.status,
        detected_person_ids=dedupe_non_empty(
            list(existing.detected_person_ids) + list(incoming.detected_person_ids)
        ),
        detected_person_names=dedupe_non_empty(
            list(existing.detected_person_names) + list(incoming.detected_person_names)
        ),
        raw_metadata=raw_metadata,
        created_at=existing.created_at,
        updated_at=utc_now(),
    )


def candidate_identity_keys(candidate: CandidateItem) -> Tuple[str, ...]:
    """返回用于判断候选重复的稳定 identity keys。"""

    keys: List[str] = [f"id:{candidate.id}"]
    if candidate.platform_item_id:
        keys.append(
            "platform:"
            f"{candidate.source_type.value}:"
            f"{candidate.platform_item_id.strip().lower()}"
        )
    for url in (candidate.effective_url, str(candidate.url)):
        normalized = normalize_candidate_url(url)
        if normalized:
            keys.append(f"url:{normalized}")
    return tuple(dedupe_non_empty(keys))


def normalize_candidate_url(url: str) -> str:
    """规范化候选 URL，用于第一层去重。"""

    text = str(url).strip()
    if not text:
        return ""
    parts = urlsplit(text)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    normalized_query = urlencode(sorted(query_items))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            normalized_query,
            "",
        )
    )


def video_identity_from_url(url: str) -> Optional[str]:
    """返回 YouTube/Bilibili 视频的跨平台身份 key。"""

    text = str(url).strip().rstrip(".,;:!?)]}，。；：！？、）】")
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None

    host = (parts.hostname or "").lower().lstrip("www.")
    if host == "youtu.be":
        video_id = parts.path.strip("/").split("/", 1)[0]
        return f"youtube:video:{video_id}" if video_id else None

    if host in _YOUTUBE_HOSTS:
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        video_id = query.get("v")
        if not video_id:
            path_parts = [part for part in parts.path.split("/") if part]
            if len(path_parts) >= 2 and path_parts[0].lower() in {
                "shorts",
                "embed",
                "live",
            }:
                video_id = path_parts[1]
        return f"youtube:video:{video_id}" if video_id else None

    if host in _BILIBILI_HOSTS:
        match = re.search(r"/video/(BV[0-9A-Za-z]+|av[0-9]+)", parts.path)
        if match:
            return f"bilibili:video:{match.group(1)}"

    return None


def candidate_video_identity_keys(candidate: CandidateItem) -> Tuple[str, ...]:
    """返回候选自身及简介中外部视频链接的身份 keys。

    例如 Bilibili 搬运条目的 Bilibili URL 和简介中的 YouTube 原始链接会
    同时返回，分类阶段可以据此发现跨平台重复内容。
    """

    keys: List[str] = []
    urls = [candidate.effective_url, str(candidate.url)]
    texts = []
    if candidate.description:
        texts.append(candidate.description)
    for key in _VIDEO_METADATA_KEYS:
        value = candidate.raw_metadata.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)

    for url in urls:
        identity = video_identity_from_url(url)
        if identity:
            keys.append(identity)
    for text in texts:
        for match in _VIDEO_URL_PATTERN.findall(text):
            identity = video_identity_from_url(match)
            if identity:
                keys.append(identity)
    return tuple(dedupe_non_empty(keys))


__all__ = [
    "CandidateIndex",
    "CandidateIngestResult",
    "candidate_identity_keys",
    "candidate_video_identity_keys",
    "ingest_candidates",
    "merge_candidate",
    "normalize_candidate_url",
    "video_identity_from_url",
]
