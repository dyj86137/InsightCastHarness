"""记忆系统本地存储实现。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from infra.exception import SerializationError
from memory.base import (
    MemoryError,
    MemoryRecord,
    MemoryStore,
    TaskStateSnapshot,
    TaskStateStore,
    task_state_sort_key,
)


SAFE_FILE_PART_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


class InMemoryMemoryStore(MemoryStore):
    """内存版 MemoryStore，适合测试和短生命周期任务。"""

    def __init__(self, records: Optional[Iterable[MemoryRecord]] = None) -> None:
        self._records = {}
        for record in records or ():
            self._records[record.id] = record

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """写入一条记忆。"""

        if record.id in self._records:
            raise MemoryError(
                "Memory already exists.",
                details={"memory_id": record.id},
            )
        self._records[record.id] = record
        return record

    async def get(self, memory_id: str) -> MemoryRecord:
        """按 id 读取记忆。"""

        normalized = normalize_file_part(memory_id, field_name="memory_id")
        record = self._records.get(normalized)
        if record is None:
            raise MemoryError(
                "Memory not found.",
                details={"memory_id": memory_id},
            )
        return record

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新一条记忆。"""

        if record.id not in self._records:
            raise MemoryError(
                "Cannot update missing memory.",
                details={"memory_id": record.id},
            )
        self._records[record.id] = record
        return record

    async def delete(self, memory_id: str) -> None:
        """删除一条记忆。"""

        normalized = normalize_file_part(memory_id, field_name="memory_id")
        if normalized not in self._records:
            raise MemoryError(
                "Cannot delete missing memory.",
                details={"memory_id": memory_id},
            )
        del self._records[normalized]

    async def list_records(self) -> List[MemoryRecord]:
        """返回所有内存记录。"""

        return sorted(self._records.values(), key=memory_sort_key, reverse=True)


class JsonlMemoryStore(MemoryStore):
    """基于本地 JSONL 文件的 MemoryStore。

    V2 采用一个文件保存全部 MemoryRecord：
    memories/memories.jsonl
    """

    def __init__(
        self,
        directory: Union[str, Path] = "memories",
        *,
        filename: str = "memories.jsonl",
        create_dir: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.filename = normalize_file_name(filename)
        self._lock = asyncio.Lock()

        if create_dir:
            self.directory.mkdir(parents=True, exist_ok=True)

    async def add(self, record: MemoryRecord) -> MemoryRecord:
        """追加一条记忆到 JSONL 文件。"""

        async with self._lock:
            records = self._read_all_unlocked()
            if any(existing.id == record.id for existing in records):
                raise MemoryError(
                    "Memory already exists.",
                    details={"memory_id": record.id, "path": str(self.path)},
                )
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(record.to_json(exclude_none=True))
                    file.write("\n")
            except OSError as exc:
                raise MemoryError(
                    "Failed to append memory record.",
                    details={"memory_id": record.id, "path": str(self.path)},
                    cause=exc,
                ) from exc
        return record

    async def get(self, memory_id: str) -> MemoryRecord:
        """按 id 读取记忆。"""

        normalized = normalize_file_part(memory_id, field_name="memory_id")
        for record in await self.list_records():
            if record.id == normalized:
                return record
        raise MemoryError(
            "Memory not found.",
            details={"memory_id": memory_id, "path": str(self.path)},
        )

    async def update(self, record: MemoryRecord) -> MemoryRecord:
        """更新一条记忆，并重写 JSONL 文件。"""

        async with self._lock:
            records = self._read_all_unlocked()
            updated = False
            new_records = []
            for existing in records:
                if existing.id == record.id:
                    new_records.append(record)
                    updated = True
                else:
                    new_records.append(existing)
            if not updated:
                raise MemoryError(
                    "Cannot update missing memory.",
                    details={"memory_id": record.id, "path": str(self.path)},
                )
            self._write_all_unlocked(new_records)
        return record

    async def delete(self, memory_id: str) -> None:
        """删除一条记忆，并重写 JSONL 文件。"""

        normalized = normalize_file_part(memory_id, field_name="memory_id")
        async with self._lock:
            records = self._read_all_unlocked()
            new_records = [record for record in records if record.id != normalized]
            if len(new_records) == len(records):
                raise MemoryError(
                    "Cannot delete missing memory.",
                    details={"memory_id": memory_id, "path": str(self.path)},
                )
            self._write_all_unlocked(new_records)

    async def list_records(self) -> List[MemoryRecord]:
        """读取全部记忆记录。"""

        async with self._lock:
            records = self._read_all_unlocked()
        return sorted(records, key=memory_sort_key, reverse=True)

    @property
    def path(self) -> Path:
        """返回 JSONL 文件路径。"""

        return self.directory / self.filename

    def _read_all_unlocked(self) -> List[MemoryRecord]:
        """读取全部记录；调用方负责持有锁。"""

        if not self.path.exists():
            return []

        records: List[MemoryRecord] = []
        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(MemoryRecord.from_json(stripped))
                    except SerializationError as exc:
                        raise MemoryError(
                            "Failed to parse memory record line.",
                            details={
                                "path": str(self.path),
                                "line_number": line_number,
                            },
                            cause=exc,
                        ) from exc
        except OSError as exc:
            raise MemoryError(
                "Failed to read memory records.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc
        return records

    def _write_all_unlocked(self, records: Sequence[MemoryRecord]) -> None:
        """重写全部记录；调用方负责持有锁。"""

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as file:
                for record in records:
                    file.write(record.to_json(exclude_none=True))
                    file.write("\n")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise MemoryError(
                "Failed to write memory records.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc


class InMemoryTaskStateStore(TaskStateStore):
    """内存版 TaskStateStore，适合测试和短生命周期任务。"""

    def __init__(self, snapshots: Optional[Iterable[TaskStateSnapshot]] = None) -> None:
        self._snapshots = {}
        for snapshot in snapshots or ():
            self._snapshots[snapshot.id] = snapshot

    async def save(self, snapshot: TaskStateSnapshot) -> TaskStateSnapshot:
        """保存或更新任务状态快照。"""

        self._snapshots[snapshot.id] = snapshot
        return snapshot

    async def load(self, snapshot_id: str) -> TaskStateSnapshot:
        """按 id 读取任务状态快照。"""

        normalized = normalize_file_part(snapshot_id, field_name="snapshot_id")
        snapshot = self._snapshots.get(normalized)
        if snapshot is None:
            raise MemoryError(
                "Task state snapshot not found.",
                details={"snapshot_id": snapshot_id},
            )
        return snapshot

    async def list_for_session(self, session_id: str) -> List[TaskStateSnapshot]:
        """列出某个 session 的所有任务状态快照。"""

        validate_required_id(session_id, field_name="session_id")
        snapshots = [
            snapshot
            for snapshot in self._snapshots.values()
            if snapshot.session_id == session_id
        ]
        return sorted(snapshots, key=task_state_sort_key)

    async def delete(self, snapshot_id: str) -> None:
        """删除一个任务状态快照。"""

        normalized = normalize_file_part(snapshot_id, field_name="snapshot_id")
        if normalized not in self._snapshots:
            raise MemoryError(
                "Cannot delete missing task state snapshot.",
                details={"snapshot_id": snapshot_id},
            )
        del self._snapshots[normalized]

    def list_snapshots(self) -> List[TaskStateSnapshot]:
        """返回所有任务状态快照。"""

        return sorted(self._snapshots.values(), key=task_state_sort_key)


class JsonlTaskStateStore(TaskStateStore):
    """基于本地 JSONL 文件的 TaskStateStore。

    V2 采用一个文件保存全部 TaskStateSnapshot：
    memories/task_states.jsonl
    """

    def __init__(
        self,
        directory: Union[str, Path] = "memories",
        *,
        filename: str = "task_states.jsonl",
        create_dir: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.filename = normalize_file_name(filename)
        self._lock = asyncio.Lock()

        if create_dir:
            self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, snapshot: TaskStateSnapshot) -> TaskStateSnapshot:
        """保存或更新任务状态快照。"""

        async with self._lock:
            snapshots = self._read_all_unlocked()
            saved = False
            new_snapshots = []
            for existing in snapshots:
                if existing.id == snapshot.id:
                    new_snapshots.append(snapshot)
                    saved = True
                else:
                    new_snapshots.append(existing)
            if not saved:
                new_snapshots.append(snapshot)
            self._write_all_unlocked(new_snapshots)
        return snapshot

    async def load(self, snapshot_id: str) -> TaskStateSnapshot:
        """按 id 读取任务状态快照。"""

        normalized = normalize_file_part(snapshot_id, field_name="snapshot_id")
        for snapshot in await self.list_snapshots():
            if snapshot.id == normalized:
                return snapshot
        raise MemoryError(
            "Task state snapshot not found.",
            details={"snapshot_id": snapshot_id, "path": str(self.path)},
        )

    async def list_for_session(self, session_id: str) -> List[TaskStateSnapshot]:
        """列出某个 session 的所有任务状态快照。"""

        validate_required_id(session_id, field_name="session_id")
        snapshots = [
            snapshot
            for snapshot in await self.list_snapshots()
            if snapshot.session_id == session_id
        ]
        return sorted(snapshots, key=task_state_sort_key)

    async def delete(self, snapshot_id: str) -> None:
        """删除一个任务状态快照。"""

        normalized = normalize_file_part(snapshot_id, field_name="snapshot_id")
        async with self._lock:
            snapshots = self._read_all_unlocked()
            new_snapshots = [
                snapshot for snapshot in snapshots if snapshot.id != normalized
            ]
            if len(new_snapshots) == len(snapshots):
                raise MemoryError(
                    "Cannot delete missing task state snapshot.",
                    details={"snapshot_id": snapshot_id, "path": str(self.path)},
                )
            self._write_all_unlocked(new_snapshots)

    async def list_snapshots(self) -> List[TaskStateSnapshot]:
        """读取全部任务状态快照。"""

        async with self._lock:
            snapshots = self._read_all_unlocked()
        return sorted(snapshots, key=task_state_sort_key)

    @property
    def path(self) -> Path:
        """返回 JSONL 文件路径。"""

        return self.directory / self.filename

    def _read_all_unlocked(self) -> List[TaskStateSnapshot]:
        """读取全部快照；调用方负责持有锁。"""

        if not self.path.exists():
            return []

        snapshots: List[TaskStateSnapshot] = []
        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        snapshots.append(TaskStateSnapshot.from_json(stripped))
                    except SerializationError as exc:
                        raise MemoryError(
                            "Failed to parse task state snapshot line.",
                            details={
                                "path": str(self.path),
                                "line_number": line_number,
                            },
                            cause=exc,
                        ) from exc
        except OSError as exc:
            raise MemoryError(
                "Failed to read task state snapshots.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc
        return snapshots

    def _write_all_unlocked(self, snapshots: Sequence[TaskStateSnapshot]) -> None:
        """重写全部快照；调用方负责持有锁。"""

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as file:
                for snapshot in snapshots:
                    file.write(snapshot.to_json(exclude_none=True))
                    file.write("\n")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise MemoryError(
                "Failed to write task state snapshots.",
                details={"path": str(self.path)},
                cause=exc,
            ) from exc


def memory_sort_key(record: MemoryRecord) -> tuple:
    """生成记忆记录排序 key。"""

    accessed_at = record.last_accessed_at or record.updated_at
    return accessed_at, record.created_at, record.importance, record.id


def normalize_file_name(filename: str) -> str:
    """规范化本地存储文件名。"""

    if not filename or not filename.strip():
        raise MemoryError("Memory store filename cannot be empty.")
    name = filename.strip()
    if "/" in name or "\\" in name:
        raise MemoryError(
            "Memory store filename cannot contain path separators.",
            details={"filename": filename},
        )
    return name


def normalize_file_part(value: str, *, field_name: str) -> str:
    """把外部 ID 规范化为安全文件名片段。"""

    validate_required_id(value, field_name=field_name)
    safe = SAFE_FILE_PART_PATTERN.sub("_", value.strip())
    safe = safe.strip("._")
    if not safe:
        raise MemoryError(
            f"{field_name} cannot be converted to a safe filename.",
            details={field_name: value},
        )
    return safe


def validate_required_id(value: str, *, field_name: str) -> None:
    """校验必填 ID 字段。"""

    if not value or not value.strip():
        raise MemoryError(f"{field_name} cannot be empty.")


__all__ = [
    "InMemoryMemoryStore",
    "InMemoryTaskStateStore",
    "JsonlMemoryStore",
    "JsonlTaskStateStore",
    "memory_sort_key",
    "normalize_file_name",
    "normalize_file_part",
    "task_state_sort_key",
    "validate_required_id",
]
