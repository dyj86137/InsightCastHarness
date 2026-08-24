"""JSONL TraceStore 实现。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List, Union

from core.event import RunEvent
from infra.config import TraceConfig
from infra.exception import SerializationError, TraceError
from tracing.base import TraceStore


SAFE_RUN_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


class JsonlTraceStore(TraceStore):
    """基于本地 JSONL 文件的 Trace 存储。

    V1 采用每个 run 一个文件的简单模型：
    traces/{run_id}.jsonl
    """

    def __init__(
        self,
        directory: Union[str, Path] = "traces",
        *,
        file_extension: str = ".jsonl",
        create_dir: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.file_extension = normalize_file_extension(file_extension)
        self._lock = asyncio.Lock()

        if create_dir:
            self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: TraceConfig) -> "JsonlTraceStore":
        """根据 TraceConfig 创建 JSONL TraceStore。"""

        return cls(
            directory=config.directory,
            file_extension=config.file_extension,
            create_dir=True,
        )

    async def append(self, event: RunEvent) -> None:
        """追加一条事件到对应 run 的 JSONL 文件。"""

        path = self.path_for_run(event.run_id)
        line = event.to_json(exclude_none=True)

        try:
            async with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as file:
                    file.write(line)
                    file.write("\n")
        except OSError as exc:
            raise TraceError(
                "Failed to append trace event.",
                details={"run_id": event.run_id, "path": str(path)},
                cause=exc,
            ) from exc

    async def list_events(self, run_id: str) -> List[RunEvent]:
        """读取某个 run 的所有事件。"""

        path = self.path_for_run(run_id)
        if not path.exists():
            return []

        events: List[RunEvent] = []
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        event = RunEvent.from_json(stripped)
                    except SerializationError as exc:
                        raise TraceError(
                            "Failed to parse trace event line.",
                            details={
                                "run_id": run_id,
                                "path": str(path),
                                "line_number": line_number,
                            },
                            cause=exc,
                        ) from exc
                    if event.run_id != run_id:
                        raise TraceError(
                            "Trace file contains event from another run.",
                            details={
                                "expected_run_id": run_id,
                                "actual_run_id": event.run_id,
                                "event_id": event.id,
                                "path": str(path),
                                "line_number": line_number,
                            },
                        )
                    events.append(event)
        except OSError as exc:
            raise TraceError(
                "Failed to read trace events.",
                details={"run_id": run_id, "path": str(path)},
                cause=exc,
            ) from exc

        return events

    async def clear_run(self, run_id: str) -> None:
        """删除某个 run 的本地 Trace 文件。"""

        path = self.path_for_run(run_id)
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise TraceError(
                "Failed to clear trace run.",
                details={"run_id": run_id, "path": str(path)},
                cause=exc,
            ) from exc

    def path_for_run(self, run_id: str) -> Path:
        """返回某个 run 对应的 JSONL 文件路径。"""

        safe_run_id = sanitize_run_id(run_id)
        return self.directory / f"{safe_run_id}{self.file_extension}"


def sanitize_run_id(run_id: str) -> str:
    """把 run_id 转换成安全文件名片段。"""

    if not run_id or not run_id.strip():
        raise TraceError("run_id cannot be empty.")
    safe = SAFE_RUN_ID_PATTERN.sub("_", run_id.strip())
    safe = safe.strip("._")
    if not safe:
        raise TraceError(
            "run_id cannot be converted to a safe filename.",
            details={"run_id": run_id},
        )
    return safe


def normalize_file_extension(file_extension: str) -> str:
    """规范化 Trace 文件扩展名。"""

    if not file_extension or not file_extension.strip():
        raise TraceError("Trace file extension cannot be empty.")
    normalized = file_extension.strip()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


__all__ = [
    "JsonlTraceStore",
    "normalize_file_extension",
    "sanitize_run_id",
]
