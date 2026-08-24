"""Socket.IO keyword subscriptions for finished InsightCast brief items."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import socketio


EVENT_INDUSTRY_UPDATE = "industry_update"
EVENT_SUBSCRIPTION_UPDATED = "subscription_updated"
SUBSCRIBE_KEYWORDS_EVENT = "subscribe_keywords"
KEYWORD_ROOM_PREFIX = "keyword:"
MAX_KEYWORDS_PER_CLIENT = 20
MAX_KEYWORD_LENGTH = 80


logger = logging.getLogger(__name__)


def normalize_keywords(value: object) -> List[str]:
    """Return stable, bounded keyword subscriptions from an untrusted payload."""

    if not isinstance(value, (list, tuple, set)):
        return []

    keywords: List[str] = []
    seen: Set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        keyword = " ".join(raw.split()).casefold()
        if not keyword or len(keyword) > MAX_KEYWORD_LENGTH or keyword in seen:
            continue
        keywords.append(keyword)
        seen.add(keyword)
        if len(keywords) >= MAX_KEYWORDS_PER_CLIENT:
            break
    return keywords


def keyword_room(keyword: str) -> str:
    return f"{KEYWORD_ROOM_PREFIX}{keyword}"


class RealtimeHub:
    """Owns Socket.IO rooms and safely receives publications from pipeline threads."""

    def __init__(self, *, cors_allowed_origins: Sequence[str]) -> None:
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=list(cors_allowed_origins),
            logger=False,
            engineio_logger=False,
        )
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscriptions: Dict[str, Set[str]] = {}
        self._install_event_handlers()

    async def start(self) -> None:
        self._event_loop = asyncio.get_running_loop()

    async def close(self) -> None:
        self._event_loop = None
        self._subscriptions.clear()
        await self.sio.shutdown()

    def publish_updates(self, updates: Iterable[Mapping[str, Any]]) -> None:
        """Schedule completed brief updates from a PipelineService worker thread."""

        payloads = [dict(update) for update in updates]
        if not payloads:
            return
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(
            self._emit_updates(payloads),
            loop,
        )
        future.add_done_callback(_consume_future_exception)

    def _install_event_handlers(self) -> None:
        @self.sio.event
        async def connect(sid: str, environ: Mapping[str, Any], auth: Any = None) -> bool:
            del environ, auth
            self._subscriptions.setdefault(sid, set())
            return True

        @self.sio.event
        async def disconnect(sid: str) -> None:
            self._subscriptions.pop(sid, None)

        @self.sio.on(SUBSCRIBE_KEYWORDS_EVENT)
        async def subscribe_keywords(sid: str, data: Any) -> Dict[str, List[str]]:
            requested = data.get("keywords") if isinstance(data, Mapping) else data
            keywords = normalize_keywords(requested)
            previous = self._subscriptions.get(sid, set())
            current = set(keywords)
            for keyword in previous - current:
                await self.sio.leave_room(sid, keyword_room(keyword))
            for keyword in current - previous:
                await self.sio.enter_room(sid, keyword_room(keyword))
            self._subscriptions[sid] = current
            payload = {"keywords": keywords}
            await self.sio.emit(EVENT_SUBSCRIPTION_UPDATED, payload, to=sid)
            return payload

    async def _emit_updates(self, updates: Sequence[Dict[str, Any]]) -> None:
        for update in updates:
            search_text = _update_search_text(update).casefold()
            matched_rooms = [
                keyword_room(keyword)
                for keyword in sorted(self._active_keywords())
                if keyword in search_text
            ]
            if matched_rooms:
                await self.sio.emit(
                    EVENT_INDUSTRY_UPDATE,
                    update,
                    to=matched_rooms,
                )

    def _active_keywords(self) -> Set[str]:
        active: Set[str] = set()
        for subscriptions in self._subscriptions.values():
            active.update(subscriptions)
        return active


def _update_search_text(update: Mapping[str, Any]) -> str:
    values: List[str] = []
    for key in (
        "title",
        "summary",
        "reason",
        "source_name",
    ):
        value = update.get(key)
        if isinstance(value, str):
            values.append(value)
    for key in (
        "person_names",
        "industries",
        "key_points",
        "mentioned_companies",
        "mentioned_products",
    ):
        value = update.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(item for item in value if isinstance(item, str))
    return " ".join(values)


def _consume_future_exception(future: Future[None]) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("Socket.IO industry update publication failed")


__all__ = [
    "EVENT_INDUSTRY_UPDATE",
    "EVENT_SUBSCRIPTION_UPDATED",
    "KEYWORD_ROOM_PREFIX",
    "RealtimeHub",
    "SUBSCRIBE_KEYWORDS_EVENT",
    "keyword_room",
    "normalize_keywords",
]
