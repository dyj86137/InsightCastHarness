"""基于本地 JSON 文件的低层存储实现。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from insightcast.storage.base import (
    DuplicateRecordError,
    InvalidRecordError,
    RecordNotFoundError,
    StorageError,
)


PathLike = Union[str, Path]


def default_storage_dir() -> Path:
    """返回 V1 默认业务数据存储目录。"""

    project_root = Path(__file__).resolve().parents[4]
    return project_root / "data" / "storage"


class JsonCollectionStore:
    """管理单个 JSON 集合文件。

    这个类只处理字典读写，不理解任何 InsightCast 领域模型。
    """

    def __init__(
        self,
        collection_name: str,
        root_dir: Optional[PathLike] = None,
        id_field: str = "id",
    ) -> None:
        self.collection_name = collection_name
        self.root_dir = Path(root_dir) if root_dir is not None else default_storage_dir()
        self.id_field = id_field
        self.path = self.root_dir / f"{collection_name}.json"

    def save_record(self, record: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
        """保存一条原始记录。"""

        record_id = self._extract_record_id(record)
        records = self.load_records()
        if not overwrite and record_id in records:
            raise DuplicateRecordError(
                f"{self.collection_name} record already exists: {record_id}"
            )
        records[record_id] = dict(record)
        self.write_records(records)
        return dict(records[record_id])

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 读取一条原始记录。"""

        record = self.load_records().get(record_id)
        if record is None:
            return None
        return dict(record)

    def require_record(self, record_id: str) -> Dict[str, Any]:
        """按 ID 读取原始记录，不存在时抛出异常。"""

        record = self.get_record(record_id)
        if record is None:
            raise RecordNotFoundError(
                f"{self.collection_name} record not found: {record_id}"
            )
        return record

    def list_records(self) -> List[Dict[str, Any]]:
        """按 ID 排序后列出全部原始记录。"""

        records = self.load_records()
        return [dict(records[key]) for key in sorted(records)]

    def delete_record(self, record_id: str) -> bool:
        """删除一条原始记录。"""

        records = self.load_records()
        if record_id not in records:
            return False
        del records[record_id]
        self.write_records(records)
        return True

    def exists(self, record_id: str) -> bool:
        """判断记录是否存在。"""

        return record_id in self.load_records()

    def count(self) -> int:
        """返回当前集合中的记录数量。"""

        return len(self.load_records())

    def clear(self) -> None:
        """清空当前集合。"""

        self.write_records({})

    def load_records(self) -> Dict[str, Dict[str, Any]]:
        """从磁盘加载整个集合。"""

        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise StorageError(f"invalid json store file: {self.path}") from exc

        if not isinstance(payload, dict):
            raise StorageError(f"json store root must be an object: {self.path}")

        records: Dict[str, Dict[str, Any]] = {}
        for record_id, record in payload.items():
            if not isinstance(record, dict):
                raise StorageError(
                    f"json store record must be an object: {self.path}#{record_id}"
                )
            records[str(record_id)] = dict(record)
        return records

    def write_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        """把整个集合原子写回磁盘。"""

        self.root_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(str(tmp_path), str(self.path))

    def _extract_record_id(self, record: Dict[str, Any]) -> str:
        """提取并校验记录 ID。"""

        value = record.get(self.id_field)
        record_id = str(value).strip() if value is not None else ""
        if not record_id:
            raise InvalidRecordError(
                f"{self.collection_name} record requires field: {self.id_field}"
            )
        return record_id


__all__ = [
    "JsonCollectionStore",
    "PathLike",
    "default_storage_dir",
]
