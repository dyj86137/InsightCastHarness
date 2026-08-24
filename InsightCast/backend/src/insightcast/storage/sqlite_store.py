"""InsightCast V1 基于 SQLite 的集合存储。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from insightcast.storage.base import (
    DuplicateRecordError,
    InvalidRecordError,
    RecordNotFoundError,
    StorageError,
)


PathLike = Union[str, Path]
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def default_sqlite_path() -> Path:
    """返回 V1 默认 SQLite 数据库路径。"""

    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / "data" / "insightcast.sqlite3"


def resolve_sqlite_path(path: Optional[PathLike] = None) -> Path:
    """将数据库文件路径或存储目录解析为数据库路径。"""

    if path is None:
        return default_sqlite_path()
    resolved = Path(path)
    if resolved.suffix.lower() in SQLITE_SUFFIXES:
        return resolved
    return resolved / "insightcast.sqlite3"


class SQLiteCollectionStore:
    """管理 SQLite 数据库中的一个逻辑集合。

    每个集合用独立表存储 JSON payload。这样可以保留现有 repository/model
    边界，同时为 V1 提供可持久化的本地存储；后续也可以逐步演进为关系表。
    """

    def __init__(
        self,
        collection_name: str,
        database_path: Optional[PathLike] = None,
        id_field: str = "id",
    ) -> None:
        self.collection_name = collection_name
        self.database_path = resolve_sqlite_path(database_path)
        self.id_field = id_field
        self.table_name = _table_name(collection_name)
        self._ensure_table()

    def save_record(self, record: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
        """保存一条原始记录。"""

        record_id = self._extract_record_id(record)
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        now = _utc_timestamp()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as connection:
                row = connection.execute(
                    f"SELECT created_at FROM {self.table_name} WHERE record_id = ?",
                    (record_id,),
                ).fetchone()
                if not overwrite and row is not None:
                    raise DuplicateRecordError(
                        f"{self.collection_name} record already exists: {record_id}"
                    )
                if row is None:
                    connection.execute(
                        (
                            f"INSERT INTO {self.table_name} "
                            "(record_id, payload, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?)"
                        ),
                        (record_id, payload, now, now),
                    )
                else:
                    connection.execute(
                        (
                            f"UPDATE {self.table_name} "
                            "SET payload = ?, updated_at = ? "
                            "WHERE record_id = ?"
                        ),
                        (payload, now, record_id),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite save failed: {self.database_path}") from exc
        return dict(record)

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 读取一条原始记录。"""

        try:
            with self._connection() as connection:
                row = connection.execute(
                    f"SELECT payload FROM {self.table_name} WHERE record_id = ?",
                    (record_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite read failed: {self.database_path}") from exc
        if row is None:
            return None
        return self._decode_payload(str(row["payload"]), record_id=record_id)

    def require_record(self, record_id: str) -> Dict[str, Any]:
        """读取一条原始记录；不存在时抛出错误。"""

        record = self.get_record(record_id)
        if record is None:
            raise RecordNotFoundError(
                f"{self.collection_name} record not found: {record_id}"
            )
        return record

    def list_records(self) -> List[Dict[str, Any]]:
        """按 ID 排序列出全部原始记录。"""

        try:
            with self._connection() as connection:
                rows = connection.execute(
                    f"SELECT record_id, payload FROM {self.table_name} ORDER BY record_id"
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite list failed: {self.database_path}") from exc
        return [
            self._decode_payload(str(row["payload"]), record_id=str(row["record_id"]))
            for row in rows
        ]

    def delete_record(self, record_id: str) -> bool:
        """按 ID 删除一条原始记录。"""

        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    f"DELETE FROM {self.table_name} WHERE record_id = ?",
                    (record_id,),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite delete failed: {self.database_path}") from exc
        return cursor.rowcount > 0

    def exists(self, record_id: str) -> bool:
        """返回指定原始记录是否存在。"""

        try:
            with self._connection() as connection:
                row = connection.execute(
                    f"SELECT 1 FROM {self.table_name} WHERE record_id = ? LIMIT 1",
                    (record_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite exists failed: {self.database_path}") from exc
        return row is not None

    def count(self) -> int:
        """返回集合中的记录数。"""

        try:
            with self._connection() as connection:
                row = connection.execute(f"SELECT COUNT(*) AS count FROM {self.table_name}").fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite count failed: {self.database_path}") from exc
        return int(row["count"])

    def clear(self) -> None:
        """清空集合。"""

        self.write_records({})

    def load_records(self) -> Dict[str, Dict[str, Any]]:
        """以 ID 为键加载全部记录。"""

        records: Dict[str, Dict[str, Any]] = {}
        for record in self.list_records():
            records[self._extract_record_id(record)] = record
        return records

    def write_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        """原子替换整个集合。"""

        now = _utc_timestamp()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as connection:
                connection.execute(f"DELETE FROM {self.table_name}")
                for record in records.values():
                    record_id = self._extract_record_id(record)
                    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
                    connection.execute(
                        (
                            f"INSERT INTO {self.table_name} "
                            "(record_id, payload, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?)"
                        ),
                        (record_id, payload, now, now),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite write failed: {self.database_path}") from exc

    def _ensure_table(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as connection:
                connection.execute(
                    (
                        f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                        "record_id TEXT PRIMARY KEY, "
                        "payload TEXT NOT NULL, "
                        "created_at TEXT NOT NULL, "
                        "updated_at TEXT NOT NULL)"
                    )
                )
                connection.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_updated_at "
                    f"ON {self.table_name}(updated_at)"
                )
        except sqlite3.Error as exc:
            raise StorageError(f"sqlite init failed: {self.database_path}") from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _extract_record_id(self, record: Dict[str, Any]) -> str:
        value = record.get(self.id_field)
        record_id = str(value).strip() if value is not None else ""
        if not record_id:
            raise InvalidRecordError(
                f"{self.collection_name} record requires field: {self.id_field}"
            )
        return record_id

    def _decode_payload(self, payload: str, *, record_id: str) -> Dict[str, Any]:
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"sqlite payload is invalid JSON: {self.collection_name}#{record_id}"
            ) from exc
        if not isinstance(record, dict):
            raise StorageError(
                f"sqlite payload must be an object: {self.collection_name}#{record_id}"
            )
        return dict(record)


def _table_name(collection_name: str) -> str:
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in collection_name.strip()
    ).strip("_")
    if not safe:
        raise InvalidRecordError("collection_name cannot be empty")
    return f"collection_{safe}"


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat()


__all__ = [
    "PathLike",
    "SQLiteCollectionStore",
    "default_sqlite_path",
    "resolve_sqlite_path",
]
