"""记忆系统基础协议与数据模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from pydantic import Field

from infra.exception import HarnessError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_memory_id() -> str:
    return f"mem_{uuid4().hex}"


def _new_task_state_id() -> str:
    return f"task_state_{uuid4().hex}"


class MemoryError(HarnessError):
    """记忆系统读写、检索或注入失败。"""


class MemoryKind(str, Enum):
    """记忆类型。"""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SKILL = "skill"


class MemoryScope(str, Enum):
    """记忆作用域。"""

    RUN = "run"
    SESSION = "session"
    GLOBAL = "global"


class MemoryRetrievalMode(str, Enum):
    """记忆检索模式。"""

    WEIGHTED_KEYWORD = "weighted_keyword"
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"


class MemoryRecord(SerializableModel):
    """一条可持久化、可检索、可注入的记忆。"""

    id: str = Field(default_factory=_new_memory_id)
    kind: MemoryKind
    content: str
    summary: Optional[str] = None
    scope: MemoryScope = MemoryScope.SESSION
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    source_message_ids: Tuple[str, ...] = Field(default_factory=tuple)
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    importance: float = 0.5
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    last_accessed_at: Optional[datetime] = None

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验记忆记录的领域规则。"""

        if not self.id or not self.id.strip():
            raise HarnessValidationError("MemoryRecord requires id.")
        if not self.content or not self.content.strip():
            raise HarnessValidationError(
                "MemoryRecord requires non-empty content.",
                details={"memory_id": self.id, "kind": self.kind.value},
            )
        validate_score(self.importance, field_name="importance", owner_id=self.id)
        validate_score(self.confidence, field_name="confidence", owner_id=self.id)
        validate_non_empty_values(self.source_message_ids, field_name="source_message_ids")
        validate_non_empty_values(self.tags, field_name="tags")
        if self.scope == MemoryScope.RUN and not self.run_id:
            raise HarnessValidationError(
                "Run-scoped memory requires run_id.",
                details={"memory_id": self.id},
            )
        if self.scope == MemoryScope.SESSION and not self.session_id:
            raise HarnessValidationError(
                "Session-scoped memory requires session_id.",
                details={"memory_id": self.id},
            )

    @property
    def injection_text(self) -> str:
        """返回适合注入上下文的文本。"""

        return self.summary or self.content

    def mark_accessed(self, accessed_at: Optional[datetime] = None) -> "MemoryRecord":
        """返回更新访问时间后的记忆。"""

        return self.clone(last_accessed_at=accessed_at or _utc_now())

    def with_metadata(self, key: str, value: Any) -> "MemoryRecord":
        """返回追加 metadata 后的新记忆。"""

        metadata = dict(self.metadata)
        metadata[key] = value
        return self.clone(metadata=metadata, updated_at=_utc_now())


class MemoryWriteRequest(SerializableModel):
    """创建记忆时的输入对象。"""

    kind: MemoryKind
    content: str
    summary: Optional[str] = None
    scope: MemoryScope = MemoryScope.SESSION
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    source_message_ids: Tuple[str, ...] = Field(default_factory=tuple)
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    importance: float = 0.5
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验写入请求。"""

        if not self.content or not self.content.strip():
            raise HarnessValidationError(
                "MemoryWriteRequest requires non-empty content.",
                details={"kind": self.kind.value},
            )
        validate_score(self.importance, field_name="importance", owner_id="memory_write")
        validate_score(self.confidence, field_name="confidence", owner_id="memory_write")
        validate_non_empty_values(self.source_message_ids, field_name="source_message_ids")
        validate_non_empty_values(self.tags, field_name="tags")
        if self.scope == MemoryScope.RUN and not self.run_id:
            raise HarnessValidationError("Run-scoped memory write requires run_id.")
        if self.scope == MemoryScope.SESSION and not self.session_id:
            raise HarnessValidationError("Session-scoped memory write requires session_id.")

    def to_record(self) -> MemoryRecord:
        """转换为 MemoryRecord。"""

        return MemoryRecord(
            kind=self.kind,
            content=self.content,
            summary=self.summary,
            scope=self.scope,
            session_id=self.session_id,
            run_id=self.run_id,
            user_id=self.user_id,
            source_message_ids=self.source_message_ids,
            tags=self.tags,
            importance=self.importance,
            confidence=self.confidence,
            metadata=self.metadata,
        )


class MemoryQuery(SerializableModel):
    """记忆检索请求。"""

    text: str = ""
    kinds: Tuple[MemoryKind, ...] = Field(default_factory=tuple)
    scopes: Tuple[MemoryScope, ...] = Field(default_factory=tuple)
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    include_global: bool = True
    top_k: int = 5
    min_score: float = 0.0
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验检索请求。"""

        if self.top_k <= 0:
            raise HarnessValidationError(
                "MemoryQuery top_k must be positive.",
                details={"top_k": self.top_k},
            )
        validate_score(self.min_score, field_name="min_score", owner_id="memory_query")
        validate_non_empty_values(self.tags, field_name="tags")
        validate_time_range(
            self.created_after,
            self.created_before,
            field_name="created",
        )
        validate_time_range(
            self.updated_after,
            self.updated_before,
            field_name="updated",
        )

    @property
    def has_text(self) -> bool:
        """判断查询是否包含文本语义。"""

        return bool(self.text and self.text.strip())


class MemorySearchResult(SerializableModel):
    """单条记忆检索结果。"""

    record: MemoryRecord
    score: float
    reason: str = ""
    matched_tags: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验检索结果。"""

        validate_score(self.score, field_name="score", owner_id=self.record.id)
        validate_non_empty_values(self.matched_tags, field_name="matched_tags")


class MemoryInjection(SerializableModel):
    """一次记忆注入结果。"""

    query: MemoryQuery
    results: Tuple[MemorySearchResult, ...] = Field(default_factory=tuple)
    content: str = ""
    estimated_tokens: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验注入结果。"""

        if self.estimated_tokens < 0:
            raise HarnessValidationError(
                "MemoryInjection estimated_tokens cannot be negative.",
                details={"estimated_tokens": self.estimated_tokens},
            )
        if self.results and not self.content:
            raise HarnessValidationError(
                "MemoryInjection with results requires non-empty content."
            )

    @property
    def records(self) -> Tuple[MemoryRecord, ...]:
        """返回被注入的记忆记录。"""

        return tuple(result.record for result in self.results)


class TaskStateSnapshot(SerializableModel):
    """跨 run/session 持久化的任务状态快照。"""

    id: str = Field(default_factory=_new_task_state_id)
    session_id: str
    latest_run_id: Optional[str] = None
    summary: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    open_tasks: Tuple[str, ...] = Field(default_factory=tuple)
    completed_tasks: Tuple[str, ...] = Field(default_factory=tuple)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验任务状态快照。"""

        if not self.id or not self.id.strip():
            raise HarnessValidationError("TaskStateSnapshot requires id.")
        if not self.session_id or not self.session_id.strip():
            raise HarnessValidationError(
                "TaskStateSnapshot requires session_id.",
                details={"snapshot_id": self.id},
            )
        validate_non_empty_values(self.open_tasks, field_name="open_tasks")
        validate_non_empty_values(self.completed_tasks, field_name="completed_tasks")

    def with_variable(self, key: str, value: Any) -> "TaskStateSnapshot":
        """返回更新变量后的任务状态快照。"""

        if not key or not key.strip():
            raise HarnessValidationError("TaskStateSnapshot variable key cannot be empty.")
        variables = dict(self.variables)
        variables[key] = value
        return self.clone(variables=variables, updated_at=_utc_now())

    def with_metadata(self, key: str, value: Any) -> "TaskStateSnapshot":
        """返回更新 metadata 后的任务状态快照。"""

        if not key or not key.strip():
            raise HarnessValidationError("TaskStateSnapshot metadata key cannot be empty.")
        metadata = dict(self.metadata)
        metadata[key] = value
        return self.clone(metadata=metadata, updated_at=_utc_now())


class MemoryPolicy(SerializableModel):
    """记忆写入与注入策略。"""

    enabled: bool = True
    auto_write_short_term: bool = True
    auto_write_task_state: bool = True
    auto_inject: bool = True
    default_top_k: int = 5
    max_injected_memories: int = 5
    min_retrieval_score: float = 0.1
    min_importance_to_persist: float = 0.0
    max_injected_tokens: int = 1200
    retrieval_mode: MemoryRetrievalMode = MemoryRetrievalMode.HYBRID
    enable_sparse_recall: bool = True
    enable_dense_recall: bool = True
    enable_rerank: bool = False
    sparse_recall_k: int = 50
    dense_recall_k: int = 50
    fusion_top_k: int = 100
    rerank_top_k: int = 50
    rrf_k: int = 60
    relevance_weight: float = 0.70
    business_weight: float = 0.30
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验记忆策略。"""

        if self.default_top_k <= 0:
            raise HarnessValidationError(
                "MemoryPolicy default_top_k must be positive.",
                details={"default_top_k": self.default_top_k},
            )
        if self.max_injected_memories <= 0:
            raise HarnessValidationError(
                "MemoryPolicy max_injected_memories must be positive.",
                details={"max_injected_memories": self.max_injected_memories},
            )
        if self.max_injected_tokens < 0:
            raise HarnessValidationError(
                "MemoryPolicy max_injected_tokens cannot be negative.",
                details={"max_injected_tokens": self.max_injected_tokens},
            )
        for field_name in (
            "sparse_recall_k",
            "dense_recall_k",
            "fusion_top_k",
            "rerank_top_k",
            "rrf_k",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise HarnessValidationError(
                    "MemoryPolicy retrieval limits must be positive.",
                    details={"field": field_name, "value": value},
                )
        validate_score(
            self.min_retrieval_score,
            field_name="min_retrieval_score",
            owner_id="memory_policy",
        )
        validate_score(
            self.min_importance_to_persist,
            field_name="min_importance_to_persist",
            owner_id="memory_policy",
        )
        validate_score(
            self.relevance_weight,
            field_name="relevance_weight",
            owner_id="memory_policy",
        )
        validate_score(
            self.business_weight,
            field_name="business_weight",
            owner_id="memory_policy",
        )


class MemoryStore(ABC):
    """记忆存储后端的统一抽象接口。"""

    @abstractmethod
    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入一条记忆，并返回已保存对象。"""

    async def add_many(self, records: Iterable[MemoryRecord]) -> List[MemoryRecord]:
        """批量写入记忆。"""

        saved: List[MemoryRecord] = []
        for record in records:
            saved.append(await self.add(record))
        return saved

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord:
        """按 id 读取记忆。"""

    @abstractmethod
    async def list_records(self) -> List[MemoryRecord]:
        """列出全部记忆记录。"""

    @abstractmethod
    async def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新一条记忆。"""

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """删除一条记忆。"""

    async def touch(self, memory_id: str) -> MemoryRecord:
        """更新记忆访问时间。"""

        record = await self.get(memory_id)
        updated = record.mark_accessed()
        return await self.update(updated)


class TaskStateStore(ABC):
    """任务状态快照存储后端的统一抽象接口。"""

    @abstractmethod
    async def save(self, snapshot: TaskStateSnapshot) -> TaskStateSnapshot:
        """保存任务状态快照，并返回已保存对象。"""

    @abstractmethod
    async def load(self, snapshot_id: str) -> TaskStateSnapshot:
        """按 snapshot_id 加载任务状态快照。"""

    @abstractmethod
    async def list_for_session(self, session_id: str) -> List[TaskStateSnapshot]:
        """列出某个 session 的所有任务状态快照。"""

    async def latest_for_session(self, session_id: str) -> Optional[TaskStateSnapshot]:
        """返回某个 session 最新的任务状态快照。"""

        snapshots = await self.list_for_session(session_id)
        if not snapshots:
            return None
        return sorted(snapshots, key=task_state_sort_key)[-1]

    @abstractmethod
    async def delete(self, snapshot_id: str) -> None:
        """删除一个任务状态快照。"""


def validate_score(value: float, *, field_name: str, owner_id: str) -> None:
    """校验 0 到 1 的分数字段。"""

    if value < 0 or value > 1:
        raise HarnessValidationError(
            "Memory score fields must be in [0, 1].",
            details={"owner_id": owner_id, "field": field_name, "value": value},
        )


def validate_non_empty_values(values: Iterable[str], *, field_name: str) -> None:
    """校验字符串集合中没有空值。"""

    invalid_values = [value for value in values if not value or not value.strip()]
    if invalid_values:
        raise HarnessValidationError(
            "Memory string collections cannot contain empty values.",
            details={"field": field_name, "invalid_values": invalid_values},
        )


def validate_time_range(
    start: Optional[datetime],
    end: Optional[datetime],
    *,
    field_name: str,
) -> None:
    """校验时间范围。"""

    if start and end and start > end:
        raise HarnessValidationError(
            "Memory time range start cannot be after end.",
            details={"field": field_name, "start": start, "end": end},
        )


def task_state_sort_key(snapshot: TaskStateSnapshot) -> tuple:
    """生成任务状态快照排序 key，保证恢复时选择最新快照。"""

    return snapshot.updated_at, snapshot.created_at, snapshot.id


__all__ = [
    "MemoryError",
    "MemoryInjection",
    "MemoryKind",
    "MemoryPolicy",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRetrievalMode",
    "MemoryScope",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryWriteRequest",
    "TaskStateStore",
    "TaskStateSnapshot",
    "task_state_sort_key",
    "validate_non_empty_values",
    "validate_score",
    "validate_time_range",
]
