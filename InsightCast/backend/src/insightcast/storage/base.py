"""存储层的基础接口和异常。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Optional, TypeVar

from insightcast.domain.models import DomainModel


ModelT = TypeVar("ModelT", bound=DomainModel)
ModelPredicate = Callable[[ModelT], bool]


class StorageError(Exception):
    """存储层通用异常。"""


class RecordNotFoundError(StorageError):
    """指定记录不存在。"""


class DuplicateRecordError(StorageError):
    """创建记录时发现 ID 已存在。"""


class InvalidRecordError(StorageError):
    """记录缺少必要字段，或无法被领域模型解析。"""


class Repository(ABC, Generic[ModelT]):
    """领域模型仓储接口。"""

    @abstractmethod
    def create(self, model: ModelT) -> ModelT:
        """创建新记录，如果 ID 已存在则失败。"""

    @abstractmethod
    def save(self, model: ModelT) -> ModelT:
        """保存记录；如果 ID 已存在则覆盖。"""

    @abstractmethod
    def get(self, record_id: str) -> Optional[ModelT]:
        """按 ID 获取记录，不存在时返回 None。"""

    @abstractmethod
    def require(self, record_id: str) -> ModelT:
        """按 ID 获取记录，不存在时抛出异常。"""

    @abstractmethod
    def list(self, predicate: Optional[ModelPredicate[ModelT]] = None) -> List[ModelT]:
        """列出记录，可传入谓词做内存过滤。"""

    @abstractmethod
    def update(self, record_id: str, **updates: Any) -> ModelT:
        """局部更新记录，并返回更新后的领域模型。"""

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """删除记录，返回是否真的删除了数据。"""

    @abstractmethod
    def exists(self, record_id: str) -> bool:
        """判断记录是否存在。"""

    @abstractmethod
    def count(self) -> int:
        """返回当前集合中的记录数量。"""


__all__ = [
    "DuplicateRecordError",
    "InvalidRecordError",
    "ModelPredicate",
    "ModelT",
    "RecordNotFoundError",
    "Repository",
    "StorageError",
]
