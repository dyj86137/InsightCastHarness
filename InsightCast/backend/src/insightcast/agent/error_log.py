"""Local JSONL error logs for InsightCast pipeline runs."""

from __future__ import annotations

import logging as py_logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pydantic import Field

from infra.logging import (
    JsonFormatter,
    LogContextFilter,
    get_logger,
    log_context,
    log_error,
)
from infra.serialization import serialize
from insightcast.domain.models import DomainModel, utc_now
from runtime.hooks import BaseHook, HookContext


class PipelineErrorLogEntry(DomainModel):
    """One structured entry in a run-scoped error log."""

    timestamp: datetime = Field(default_factory=utc_now)
    run_id: str
    event: str
    level: str = "error"
    message: str
    stage: Optional[str] = None
    step: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class PipelineErrorLogger:
    """Writes run-scoped pipeline errors through myHarness logging."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.path = self.log_dir / f"{run_id}.jsonl"
        self.logger = get_logger(f"insightcast.pipeline.error.{run_id}")
        self._handler: Optional[py_logging.Handler] = None

    def initialize(self) -> Path:
        """Create the per-run error log file and attach a myHarness JSON handler."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        if self._handler is not None:
            return self.path

        self._remove_stale_file_handlers()
        handler = py_logging.FileHandler(self.path, encoding="utf-8")
        handler.setLevel(py_logging.ERROR)
        handler.addFilter(LogContextFilter())
        handler.setFormatter(JsonFormatter())
        setattr(handler, "_insightcast_error_log_path", str(self.path))

        self.logger.setLevel(py_logging.ERROR)
        self.logger.propagate = False
        self.logger.addHandler(handler)
        self._handler = handler
        return self.path

    def close(self) -> None:
        """Detach and close the run-scoped myHarness file handler."""

        if self._handler is None:
            return
        self.logger.removeHandler(self._handler)
        self._handler.close()
        self._handler = None

    def append(self, entry: PipelineErrorLogEntry) -> None:
        self.initialize()
        with log_context(run_id=self.run_id):
            self.logger.error(
                entry.message,
                extra=_entry_extra(entry),
            )

    def log_stage_exception(
        self,
        *,
        stage: str,
        step: int,
        exc: Exception,
    ) -> None:
        self.initialize()
        with log_context(run_id=self.run_id):
            log_error(
                self.logger,
                exc,
                message=f"Pipeline stage failed: {stage}",
                extra={
                    "event": "stage_exception",
                    "stage": stage,
                    "step": step,
                },
            )

    def log_stage_error(
        self,
        *,
        stage: str,
        step: Optional[int],
        error: Optional[Dict[str, Any]],
    ) -> None:
        self.append(
            PipelineErrorLogEntry(
                run_id=self.run_id,
                event="stage_exception",
                message=f"Pipeline stage failed: {stage}",
                stage=stage,
                step=step,
                error=error,
            )
        )

    def log_stage_result_errors(
        self,
        *,
        stage: str,
        step: int,
        stage_status: str,
        stage_run_id: Optional[str],
        metrics: Mapping[str, Any],
    ) -> None:
        errors = _coerce_error_items(metrics.get("errors"))
        if not errors and stage_status not in ("failed", "partial"):
            return
        if not errors:
            errors = ({"message": f"stage ended with status {stage_status}"},)

        for index, item in enumerate(errors):
            self.append(
                PipelineErrorLogEntry(
                    run_id=self.run_id,
                    event="stage_result_error",
                    message=f"Pipeline stage reported {stage_status}: {stage}",
                    stage=stage,
                    step=step,
                    payload={
                        "stage_status": stage_status,
                        "stage_run_id": stage_run_id,
                        "error_index": index,
                        "error": item,
                    },
                )
            )

    def log_run_exception(self, *, exc: Exception) -> None:
        self.initialize()
        with log_context(run_id=self.run_id):
            log_error(
                self.logger,
                exc,
                message="Pipeline run failed.",
                extra={"event": "run_exception"},
            )

    def log_run_error(self, *, error: Optional[Dict[str, Any]]) -> None:
        self.append(
            PipelineErrorLogEntry(
                run_id=self.run_id,
                event="run_exception",
                message="Pipeline run failed.",
                error=error,
            )
        )

    def _remove_stale_file_handlers(self) -> None:
        for handler in list(self.logger.handlers):
            if getattr(handler, "_insightcast_error_log_path", None) == str(self.path):
                self.logger.removeHandler(handler)
                handler.close()


class PipelineErrorLogHook(BaseHook):
    """myHarness hook that writes InsightCast pipeline error logs."""

    name = "insightcast_pipeline_error_log"

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self._loggers: Dict[str, PipelineErrorLogger] = {}

    def path_for_run(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.jsonl"

    async def on_run_started(self, context: HookContext) -> None:
        self._logger_for_run(context.run_id).initialize()

    async def on_run_completed(self, context: HookContext) -> None:
        self.close_run(context.run_id)

    async def on_run_failed(self, context: HookContext) -> None:
        self._logger_for_run(context.run_id).log_run_error(error=context.error)
        self.close_run(context.run_id)

    async def on_step_completed(self, context: HookContext) -> None:
        stage = str(context.payload.get("stage") or "")
        if not stage:
            return
        metrics = context.payload.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        self._logger_for_run(context.run_id).log_stage_result_errors(
            stage=stage,
            step=context.step or 0,
            stage_status=str(context.payload.get("status") or "unknown"),
            stage_run_id=_optional_str(metrics.get("run_id")),
            metrics=metrics,
        )

    async def on_step_failed(self, context: HookContext) -> None:
        stage = str(context.payload.get("stage") or "unknown")
        self._logger_for_run(context.run_id).log_stage_error(
            stage=stage,
            step=context.step,
            error=context.error,
        )

    def close_run(self, run_id: str) -> None:
        logger = self._loggers.pop(run_id, None)
        if logger is not None:
            logger.close()

    def close_all(self) -> None:
        for run_id in list(self._loggers):
            self.close_run(run_id)

    def _logger_for_run(self, run_id: str) -> PipelineErrorLogger:
        logger = self._loggers.get(run_id)
        if logger is None:
            logger = PipelineErrorLogger(self.log_dir, run_id)
            self._loggers[run_id] = logger
        return logger


def _coerce_error_items(value: Any) -> Tuple[Dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_coerce_error_item(item) for item in value)
    return (_coerce_error_item(value),)


def _coerce_error_item(value: Any) -> Dict[str, Any]:
    serialized = serialize(value)
    if isinstance(serialized, dict):
        return serialized
    return {"value": serialized}


def _entry_extra(entry: PipelineErrorLogEntry) -> Dict[str, Any]:
    data = entry.to_dict(exclude_none=True)
    data.pop("timestamp", None)
    data.pop("run_id", None)
    data.pop("message", None)
    data.pop("level", None)
    return data


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


__all__ = [
    "PipelineErrorLogHook",
    "PipelineErrorLogEntry",
    "PipelineErrorLogger",
]
