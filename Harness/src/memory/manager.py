"""记忆系统编排入口。"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

from memory.base import (
    MemoryError,
    MemoryInjection,
    MemoryKind,
    MemoryPolicy,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
    MemoryWriteRequest,
    TaskStateSnapshot,
    TaskStateStore,
)
from memory.embedding import EmbeddingProvider
from memory.index import MemoryVectorIndex
from memory.lexical import BM25MemoryIndex
from memory.reranker import MemoryReranker
from memory.retriever import MemoryRetriever
from memory.store import InMemoryMemoryStore, InMemoryTaskStateStore


MemoryInput = Union[MemoryRecord, MemoryWriteRequest]


class MemoryManager:
    """统一编排记忆写入、检索、注入和任务状态持久化。"""

    def __init__(
        self,
        *,
        memory_store: Optional[MemoryStore] = None,
        task_state_store: Optional[TaskStateStore] = None,
        retriever: Optional[MemoryRetriever] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_index: Optional[MemoryVectorIndex] = None,
        lexical_index: Optional[BM25MemoryIndex] = None,
        reranker: Optional[MemoryReranker] = None,
        policy: Optional[MemoryPolicy] = None,
    ) -> None:
        self.policy = policy or MemoryPolicy()
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.task_state_store = task_state_store or InMemoryTaskStateStore()
        self.retriever = retriever or MemoryRetriever(
            self.memory_store,
            policy=self.policy,
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            lexical_index=lexical_index,
            reranker=reranker,
        )

    async def add_memory(self, memory: MemoryInput) -> MemoryRecord:
        """写入一条记忆。"""

        record = normalize_memory_input(memory)
        if not self.policy.enabled:
            return record
        if record.importance < self.policy.min_importance_to_persist:
            return record
        saved = await self.memory_store.add(record)
        await self.rebuild_index()
        return saved

    async def add_memories(self, memories: Iterable[MemoryInput]) -> List[MemoryRecord]:
        """批量写入记忆。"""

        saved: List[MemoryRecord] = []
        for memory in memories:
            saved.append(await self.add_memory(memory))
        return saved

    async def add_short_term(
        self,
        content: str,
        *,
        session_id: str,
        run_id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Iterable[str] = (),
        importance: float = 0.5,
    ) -> MemoryRecord:
        """写入短期会话记忆。"""

        return await self.add_memory(
            MemoryWriteRequest(
                kind=MemoryKind.SHORT_TERM,
                content=content,
                summary=summary,
                scope=MemoryScope.SESSION,
                session_id=session_id,
                run_id=run_id,
                tags=tuple(tags),
                importance=importance,
            )
        )

    async def add_long_term(
        self,
        content: str,
        *,
        summary: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Iterable[str] = (),
        importance: float = 0.7,
    ) -> MemoryRecord:
        """写入长期语义记忆。"""

        return await self.add_memory(
            MemoryWriteRequest(
                kind=MemoryKind.LONG_TERM,
                content=content,
                summary=summary,
                scope=MemoryScope.GLOBAL,
                user_id=user_id,
                tags=tuple(tags),
                importance=importance,
            )
        )

    async def add_skill_memory(
        self,
        content: str,
        *,
        summary: Optional[str] = None,
        tags: Iterable[str] = (),
        importance: float = 0.7,
    ) -> MemoryRecord:
        """写入技能记忆。"""

        return await self.add_memory(
            MemoryWriteRequest(
                kind=MemoryKind.SKILL,
                content=content,
                summary=summary,
                scope=MemoryScope.GLOBAL,
                tags=tuple(tags),
                importance=importance,
            )
        )

    async def retrieve(
        self,
        query: MemoryQuery,
        *,
        touch: bool = True,
    ) -> List[MemorySearchResult]:
        """检索记忆，并按需更新访问时间。"""

        if not self.policy.enabled:
            return []

        results = await self.retriever.retrieve(query)
        if not touch:
            return results

        touched_results: List[MemorySearchResult] = []
        for result in results:
            touched_record = await self.memory_store.touch(result.record.id)
            touched_results.append(result.clone(record=touched_record))
        return touched_results

    async def build_injection(
        self,
        query: MemoryQuery,
        *,
        touch: bool = True,
        title: str = "Relevant memory",
    ) -> MemoryInjection:
        """检索并构造可注入上下文的记忆文本。"""

        if not self.policy.enabled or not self.policy.auto_inject:
            return MemoryInjection(query=query, results=(), content="", estimated_tokens=0)

        results = await self.retrieve(query, touch=touch)
        selected = tuple(results[: self.policy.max_injected_memories])
        selected = select_results_for_injection(selected, self.policy, title=title)
        content = format_memory_injection_content(selected, title=title)
        return MemoryInjection(
            query=query,
            results=selected,
            content=content,
            estimated_tokens=estimate_memory_tokens(content),
            metadata={
                "result_count": len(results),
                "injected_count": len(selected),
                "max_injected_memories": self.policy.max_injected_memories,
            },
        )

    async def save_task_state(self, snapshot: TaskStateSnapshot) -> TaskStateSnapshot:
        """保存任务状态快照。"""

        if not self.policy.enabled or not self.policy.auto_write_task_state:
            return snapshot
        return await self.task_state_store.save(snapshot)

    async def load_task_state(self, snapshot_id: str) -> TaskStateSnapshot:
        """按 id 加载任务状态快照。"""

        return await self.task_state_store.load(snapshot_id)

    async def latest_task_state(self, session_id: str) -> Optional[TaskStateSnapshot]:
        """读取某个 session 最新任务状态快照。"""

        return await self.task_state_store.latest_for_session(session_id)

    async def delete_memory(self, memory_id: str) -> None:
        """删除一条记忆。"""

        await self.memory_store.delete(memory_id)
        await self.rebuild_index()

    async def delete_task_state(self, snapshot_id: str) -> None:
        """删除一个任务状态快照。"""

        await self.task_state_store.delete(snapshot_id)

    async def rebuild_index(self) -> None:
        """让 retriever 的派生检索索引与 store 同步。"""

        sync_indexes = getattr(self.retriever, "sync_indexes", None)
        if sync_indexes is None:
            return
        records = await self.memory_store.list_records()
        await sync_indexes(records)


def normalize_memory_input(memory: MemoryInput) -> MemoryRecord:
    """把写入输入转换为 MemoryRecord。"""

    if isinstance(memory, MemoryRecord):
        return memory
    if isinstance(memory, MemoryWriteRequest):
        return memory.to_record()
    raise MemoryError(
        "Unsupported memory input type.",
        details={"actual_type": memory.__class__.__name__},
    )


def build_memory_query(
    *,
    text: str = "",
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    kinds: Iterable[MemoryKind] = (),
    tags: Iterable[str] = (),
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    updated_after: Optional[datetime] = None,
    updated_before: Optional[datetime] = None,
    metadata_filters: Optional[dict] = None,
    policy: Optional[MemoryPolicy] = None,
) -> MemoryQuery:
    """构造 MemoryQuery 的便捷函数。"""

    selected_policy = policy or MemoryPolicy()
    return MemoryQuery(
        text=text,
        session_id=session_id,
        run_id=run_id,
        user_id=user_id,
        kinds=tuple(kinds),
        tags=tuple(tags),
        top_k=top_k or selected_policy.default_top_k,
        min_score=(
            selected_policy.min_retrieval_score
            if min_score is None
            else min_score
        ),
        created_after=created_after,
        created_before=created_before,
        updated_after=updated_after,
        updated_before=updated_before,
        metadata_filters=metadata_filters or {},
    )


def format_memory_injection_content(
    results: Iterable[MemorySearchResult],
    *,
    title: str = "Relevant memory",
) -> str:
    """把检索结果格式化为可注入 prompt 的文本。"""

    result_tuple = tuple(results)
    if not result_tuple:
        return ""

    lines = [f"{title}:"]
    for index, result in enumerate(result_tuple, start=1):
        record = result.record
        tag_text = f" tags={','.join(record.tags)}" if record.tags else ""
        lines.append(
            f"{index}. [{record.kind.value} score={result.score:.2f}{tag_text}] "
            f"{record.injection_text}"
        )
    return "\n".join(lines)


def select_results_for_injection(
    results: Iterable[MemorySearchResult],
    policy: MemoryPolicy,
    *,
    title: str = "Relevant memory",
) -> Tuple[MemorySearchResult, ...]:
    """按条数和 token budget 选择可注入记忆。"""

    if policy.max_injected_tokens <= 0:
        return ()

    selected: List[MemorySearchResult] = []
    for result in results:
        if len(selected) >= policy.max_injected_memories:
            break
        candidate = tuple(selected + [result])
        content = format_memory_injection_content(candidate, title=title)
        if estimate_memory_tokens(content) > policy.max_injected_tokens:
            continue
        selected.append(result)
    return tuple(selected)


def estimate_memory_tokens(text: str) -> int:
    """粗略估算注入文本 token 数。"""

    if not text:
        return 0
    return max(1, len(text.strip()) // 4 + 1)


def create_default_memory_manager(
    *,
    policy: Optional[MemoryPolicy] = None,
) -> MemoryManager:
    """创建默认内存版 MemoryManager。"""

    return MemoryManager(policy=policy)


__all__ = [
    "MemoryInput",
    "MemoryManager",
    "build_memory_query",
    "create_default_memory_manager",
    "estimate_memory_tokens",
    "format_memory_injection_content",
    "normalize_memory_input",
    "select_results_for_injection",
]
