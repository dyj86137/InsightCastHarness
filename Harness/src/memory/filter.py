"""记忆检索硬过滤规则。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List, Optional

from memory.base import MemoryQuery, MemoryRecord, MemoryScope


def filter_records(
    records: Iterable[MemoryRecord],
    query: MemoryQuery,
) -> List[MemoryRecord]:
    """返回满足查询硬过滤条件的候选记忆。"""

    return [record for record in records if record_matches_query(record, query)]


def record_matches_query(record: MemoryRecord, query: MemoryQuery) -> bool:
    """判断记录是否满足权限、作用域和结构化查询条件。"""

    if query.kinds and record.kind not in query.kinds:
        return False
    if query.scopes and record.scope not in query.scopes:
        return False
    if not query.include_global and record.scope == MemoryScope.GLOBAL:
        return False
    if not record_matches_user(record, query):
        return False
    if not record_matches_scope(record, query):
        return False
    if query.tags and not set(query.tags).intersection(record.tags):
        return False
    if not record_matches_time_range(record.created_at, query.created_after, query.created_before):
        return False
    if not record_matches_time_range(record.updated_at, query.updated_after, query.updated_before):
        return False
    if not record_matches_metadata(record, query):
        return False
    return True


def record_matches_user(record: MemoryRecord, query: MemoryQuery) -> bool:
    """校验用户隔离，避免跨用户召回。"""

    if query.user_id:
        if record.user_id and record.user_id != query.user_id:
            return False
        if record.user_id == query.user_id:
            return True
        if record.scope == MemoryScope.GLOBAL:
            return True
        if query.session_id and record.session_id == query.session_id:
            return True
        if query.run_id and record.run_id == query.run_id:
            return True
        return False

    if record.user_id:
        return False
    return True


def record_matches_scope(record: MemoryRecord, query: MemoryQuery) -> bool:
    """校验 run/session/global 作用域。"""

    if record.scope == MemoryScope.RUN:
        return bool(query.run_id and record.run_id == query.run_id)

    if record.scope == MemoryScope.SESSION:
        return bool(query.session_id and record.session_id == query.session_id)

    if record.scope == MemoryScope.GLOBAL:
        return query.include_global

    return False


def record_matches_time_range(
    value: datetime,
    start: Optional[datetime],
    end: Optional[datetime],
) -> bool:
    """校验单个时间值是否落在查询范围内。"""

    if start and value < start:
        return False
    if end and value > end:
        return False
    return True


def record_matches_metadata(record: MemoryRecord, query: MemoryQuery) -> bool:
    """按 metadata_filters 做精确匹配。"""

    for key, expected in query.metadata_filters.items():
        actual = record.metadata.get(key)
        if not metadata_value_matches(actual, expected):
            return False
    return True


def metadata_value_matches(actual: Any, expected: Any) -> bool:
    """支持标量精确匹配和候选集合匹配。"""

    if isinstance(expected, (list, tuple, set, frozenset)):
        return actual in expected
    return actual == expected


__all__ = [
    "filter_records",
    "metadata_value_matches",
    "record_matches_metadata",
    "record_matches_query",
    "record_matches_scope",
    "record_matches_time_range",
    "record_matches_user",
]
