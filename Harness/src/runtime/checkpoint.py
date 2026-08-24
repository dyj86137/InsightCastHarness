"""运行时 Checkpoint 管理。

Checkpoint 是 AgentState 的持久化快照，用于断点续跑、失败重放和长周期任务恢复。
V2 先提供本地文件版实现，后续可以替换为数据库、对象存储或远程状态服务。
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

from pydantic import Field

from core.event import RunEvent, RunEventType
from core.state import AgentState
from infra.exception import RunnerError, SerializationError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


SAFE_FILE_PART_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


def _utc_now() -> datetime:
    return datetime.utcnow()


def _new_checkpoint_id() -> str:
    return f"ckpt_{uuid4().hex}"


class CheckpointReason(str, Enum):
    """创建 Checkpoint 的原因。"""

    MANUAL = "manual"
    RUN_STARTED = "run_started"
    STEP_COMPLETED = "step_completed"
    RUN_FAILED = "run_failed"
    RUN_COMPLETED = "run_completed"


class CheckpointError(RunnerError):
    """Checkpoint 读写或恢复失败。"""


class Checkpoint(SerializableModel):
    """一次可恢复的 AgentState 快照。"""

    id: str = Field(default_factory=_new_checkpoint_id)
    run_id: str
    session_id: Optional[str] = None
    step: int
    state: AgentState
    reason: CheckpointReason = CheckpointReason.MANUAL
    source_event_id: Optional[str] = None
    schema_version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Checkpoint 与 AgentState 的归属一致。"""

        if not self.id or not self.id.strip():
            raise HarnessValidationError("Checkpoint requires id.")
        if not self.run_id or not self.run_id.strip():
            raise HarnessValidationError("Checkpoint requires run_id.")
        if self.step < 0:
            raise HarnessValidationError(
                "Checkpoint step cannot be negative.",
                details={"checkpoint_id": self.id, "step": self.step},
            )
        if self.schema_version <= 0:
            raise HarnessValidationError(
                "Checkpoint schema_version must be positive.",
                details={"checkpoint_id": self.id, "schema_version": self.schema_version},
            )
        if self.state.run_id != self.run_id:
            raise HarnessValidationError(
                "Checkpoint state must belong to the same run.",
                details={
                    "checkpoint_id": self.id,
                    "checkpoint_run_id": self.run_id,
                    "state_run_id": self.state.run_id,
                },
            )
        if self.state.step != self.step:
            raise HarnessValidationError(
                "Checkpoint step must match AgentState step.",
                details={
                    "checkpoint_id": self.id,
                    "checkpoint_step": self.step,
                    "state_step": self.state.step,
                },
            )
        if self.session_id and self.state.session_id and self.session_id != self.state.session_id:
            raise HarnessValidationError(
                "Checkpoint session_id must match AgentState session_id.",
                details={
                    "checkpoint_id": self.id,
                    "checkpoint_session_id": self.session_id,
                    "state_session_id": self.state.session_id,
                },
            )

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        *,
        reason: CheckpointReason = CheckpointReason.MANUAL,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Checkpoint":
        """从 AgentState 创建 Checkpoint。"""

        return cls(
            run_id=state.run_id,
            session_id=state.session_id,
            step=state.step,
            state=state,
            reason=reason,
            source_event_id=source_event_id,
            metadata=metadata or {},
        )

    def to_event(self) -> RunEvent:
        """转换为 checkpoint_created 运行事件。"""

        return RunEvent.create(
            run_id=self.run_id,
            session_id=self.session_id,
            step=self.step,
            event_type=RunEventType.CHECKPOINT_CREATED,
            message="Checkpoint created.",
            payload={
                "checkpoint_id": self.id,
                "reason": self.reason.value,
                "schema_version": self.schema_version,
            },
        )


class CheckpointStore(ABC):
    """Checkpoint 存储后端的统一抽象接口。"""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """保存 Checkpoint，并返回已保存对象。"""

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Checkpoint:
        """按 checkpoint_id 加载 Checkpoint。"""

    @abstractmethod
    async def list_for_run(self, run_id: str) -> List[Checkpoint]:
        """列出某个 run 的所有 Checkpoint。"""

    async def latest_for_run(self, run_id: str) -> Optional[Checkpoint]:
        """返回某个 run 最新的 Checkpoint。"""

        checkpoints = await self.list_for_run(run_id)
        if not checkpoints:
            return None
        return sorted(checkpoints, key=checkpoint_sort_key)[-1]


class InMemoryCheckpointStore(CheckpointStore):
    """内存版 CheckpointStore，适合测试和短生命周期任务。"""

    def __init__(self, checkpoints: Optional[Iterable[Checkpoint]] = None) -> None:
        self._checkpoints: Dict[str, Checkpoint] = {}
        for checkpoint in checkpoints or ():
            self._checkpoints[checkpoint.id] = checkpoint

    async def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """保存 Checkpoint 到内存字典。"""

        self._checkpoints[checkpoint.id] = checkpoint
        return checkpoint

    async def load(self, checkpoint_id: str) -> Checkpoint:
        """从内存字典加载 Checkpoint。"""

        normalized = normalize_file_part(checkpoint_id, field_name="checkpoint_id")
        checkpoint = self._checkpoints.get(normalized)
        if checkpoint is None:
            raise CheckpointError(
                "Checkpoint not found.",
                details={"checkpoint_id": checkpoint_id},
            )
        return checkpoint

    async def list_for_run(self, run_id: str) -> List[Checkpoint]:
        """列出指定 run 的内存 Checkpoint。"""

        if not run_id or not run_id.strip():
            raise CheckpointError("run_id cannot be empty.")
        checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if checkpoint.run_id == run_id
        ]
        return sorted(checkpoints, key=checkpoint_sort_key)


class FileCheckpointStore(CheckpointStore):
    """基于本地 JSON 文件的 CheckpointStore。

    V2 采用每个 checkpoint 一个 JSON 文件的模型：
    checkpoints/{checkpoint_id}.json
    """

    def __init__(
        self,
        directory: Union[str, Path] = "checkpoints",
        *,
        create_dir: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self._lock = asyncio.Lock()

        if create_dir:
            self.directory.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """保存 Checkpoint 到本地 JSON 文件。"""

        path = self.path_for_checkpoint(checkpoint.id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            async with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("w", encoding="utf-8") as file:
                    file.write(checkpoint.to_json(exclude_none=True, indent=2))
                    file.write("\n")
                tmp_path.replace(path)
        except OSError as exc:
            raise CheckpointError(
                "Failed to save checkpoint.",
                details={"checkpoint_id": checkpoint.id, "path": str(path)},
                cause=exc,
            ) from exc

        return checkpoint

    async def load(self, checkpoint_id: str) -> Checkpoint:
        """从本地 JSON 文件加载 Checkpoint。"""

        path = self.path_for_checkpoint(checkpoint_id)
        if not path.exists():
            raise CheckpointError(
                "Checkpoint not found.",
                details={"checkpoint_id": checkpoint_id, "path": str(path)},
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                return Checkpoint.from_json(file.read())
        except SerializationError as exc:
            raise CheckpointError(
                "Failed to parse checkpoint file.",
                details={"checkpoint_id": checkpoint_id, "path": str(path)},
                cause=exc,
            ) from exc
        except OSError as exc:
            raise CheckpointError(
                "Failed to read checkpoint file.",
                details={"checkpoint_id": checkpoint_id, "path": str(path)},
                cause=exc,
            ) from exc

    async def list_for_run(self, run_id: str) -> List[Checkpoint]:
        """扫描本地目录并返回指定 run 的所有 Checkpoint。"""

        if not run_id or not run_id.strip():
            raise CheckpointError("run_id cannot be empty.")
        if not self.directory.exists():
            return []

        checkpoints: List[Checkpoint] = []
        try:
            for path in self.directory.glob("*.json"):
                checkpoint = await self._load_path(path)
                if checkpoint.run_id == run_id:
                    checkpoints.append(checkpoint)
        except OSError as exc:
            raise CheckpointError(
                "Failed to list checkpoint files.",
                details={"run_id": run_id, "directory": str(self.directory)},
                cause=exc,
            ) from exc

        return sorted(checkpoints, key=checkpoint_sort_key)

    def path_for_checkpoint(self, checkpoint_id: str) -> Path:
        """返回 checkpoint 对应的本地文件路径。"""

        safe_id = normalize_file_part(checkpoint_id, field_name="checkpoint_id")
        return self.directory / f"{safe_id}.json"

    async def _load_path(self, path: Path) -> Checkpoint:
        """按文件路径加载 Checkpoint。"""

        try:
            with path.open("r", encoding="utf-8") as file:
                return Checkpoint.from_json(file.read())
        except SerializationError as exc:
            raise CheckpointError(
                "Failed to parse checkpoint file.",
                details={"path": str(path)},
                cause=exc,
            ) from exc
        except OSError as exc:
            raise CheckpointError(
                "Failed to read checkpoint file.",
                details={"path": str(path)},
                cause=exc,
            ) from exc


class CheckpointManager:
    """Runner 使用的 Checkpoint 编排器。"""

    def __init__(
        self,
        store: Optional[CheckpointStore] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.store = store or FileCheckpointStore()
        self.enabled = enabled

    async def create(
        self,
        state: AgentState,
        *,
        reason: CheckpointReason = CheckpointReason.MANUAL,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """强制创建并保存 Checkpoint。"""

        checkpoint = Checkpoint.from_state(
            state,
            reason=reason,
            source_event_id=source_event_id,
            metadata=metadata,
        )
        return await self.store.save(checkpoint)

    async def create_if_enabled(
        self,
        state: AgentState,
        *,
        reason: CheckpointReason = CheckpointReason.MANUAL,
        source_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Checkpoint]:
        """启用 Checkpoint 时创建快照，禁用时返回 None。"""

        if not self.enabled:
            return None
        return await self.create(
            state,
            reason=reason,
            source_event_id=source_event_id,
            metadata=metadata,
        )

    async def restore_state(self, checkpoint_id: str) -> AgentState:
        """从 Checkpoint 还原 AgentState。"""

        checkpoint = await self.store.load(checkpoint_id)
        return checkpoint.state

    async def latest_state_for_run(self, run_id: str) -> Optional[AgentState]:
        """读取指定 run 最新 Checkpoint 中的 AgentState。"""

        checkpoint = await self.store.latest_for_run(run_id)
        if checkpoint is None:
            return None
        return checkpoint.state


def normalize_file_part(value: str, *, field_name: str) -> str:
    """把外部 ID 规范化为安全文件名片段。"""

    if not value or not value.strip():
        raise CheckpointError(f"{field_name} cannot be empty.")
    safe = SAFE_FILE_PART_PATTERN.sub("_", value.strip())
    safe = safe.strip("._")
    if not safe:
        raise CheckpointError(
            f"{field_name} cannot be converted to a safe filename.",
            details={field_name: value},
        )
    return safe


def checkpoint_sort_key(checkpoint: Checkpoint) -> tuple:
    """生成 Checkpoint 排序 key，保证恢复时选择最新快照。"""

    return checkpoint.created_at, checkpoint.step, checkpoint.id


__all__ = [
    "Checkpoint",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointReason",
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "checkpoint_sort_key",
    "normalize_file_part",
]
