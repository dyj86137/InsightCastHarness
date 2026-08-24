"""myHarness V1 的统一日志模块。"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging as py_logging
import sys
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from .config import LoggingConfig
from .exception import HarnessError


# LOGGER_NAME：统一框架日志的根命名空间。
# 所有子模块的 logger 都会挂在 myharness.xxx 下，和业务代码日志完全隔离，方便过滤、采集、按模块排查。
LOGGER_NAME = "myharness"

DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] "
    "[run_id=%(run_id)s session_id=%(session_id)s] %(message)s"
)

# 为什么不用普通字典？contextvars 可在多线程、异步并发执行多个 Agent 任务时，每个任务的上下文完全隔离，不会出现 A 任务的 run_id 跑到 B 任务的日志里；
# 如果用普通全局字典，并发场景下会出现数据串扰
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "myharness_log_context",
    default={},
)


def _build_standard_record_keys() -> set[str]:
    record = py_logging.LogRecord(
        name="",
        level=py_logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    return set(record.__dict__) | {"message", "asctime", "run_id", "session_id"}


STANDARD_RECORD_KEYS = _build_standard_record_keys()


def _json_default(value: Any) -> str:
    return str(value)


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    normalized = level.upper()
    resolved = py_logging.getLevelName(normalized)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unknown log level: {level}")


class LogContextFilter(py_logging.Filter):
    """从上下文变量里取出 run_id、session_id 字段，注入到日志记录中"""

    def filter(self, record: py_logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get()
        record.run_id = context.get("run_id", "-")
        record.session_id = context.get("session_id", "-")

        for key, value in context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(py_logging.Formatter):
    """把每条日志输出成单行 JSON（JSONL 格式），方便后续被 Trace 或日志平台采集。"""

    def format(self, record: py_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_RECORD_KEYS
        }
        if extra:
            payload["extra"] = extra

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    level: str | int | None = None,
    json_logs: bool | None = None,
    logger_name: str = LOGGER_NAME,
) -> py_logging.Logger:
    """框架启动时调用的初始化入口，一次性完成全局日志配置"""

    logging_config = config or LoggingConfig()
    resolved_level = _normalize_level(level or logging_config.level)
    use_json_logs = logging_config.json if json_logs is None else json_logs

    logger = py_logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(resolved_level)
    logger.propagate = False

    handler = py_logging.StreamHandler(sys.stderr)
    handler.setLevel(resolved_level)
    handler.addFilter(LogContextFilter())

    if use_json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(py_logging.Formatter(DEFAULT_LOG_FORMAT))

    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> py_logging.Logger:
    """全框架统一的 logger 获取方式，保证所有子模块的日志都在 myharness 命名空间下"""

    if not name or name == LOGGER_NAME:
        return py_logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return py_logging.getLogger(name)
    return py_logging.getLogger(f"{LOGGER_NAME}.{name}")



# 以下四个函数是配套使用的上下文管理函数组，用于给 Agent 任务绑定专属日志上下文，和 Runner 的生命周期深度配合
def current_log_context() -> dict[str, Any]:
    """返回当前日志上下文的副本。"""

    return dict(_LOG_CONTEXT.get())


def set_log_context(**values: Any) -> contextvars.Token[dict[str, Any]]:
    """合并并设置当前日志上下文，返回可用于恢复的 token。"""

    context = current_log_context()
    context.update({key: value for key, value in values.items() if value is not None})
    return _LOG_CONTEXT.set(context)


def reset_log_context(token: contextvars.Token[dict[str, Any]]) -> None:
    """根据 token 恢复之前的日志上下文。"""

    _LOG_CONTEXT.reset(token)


@contextlib.contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """在上下文管理器范围内临时绑定日志上下文。"""

    token = set_log_context(**values)
    try:
        yield
    finally:
        reset_log_context(token)


def log_error(
    logger: py_logging.Logger,
    error: BaseException,
    *,
    message: str = "Unhandled error",
    level: int = py_logging.ERROR,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """统一异常日志记录格式"""

    log_extra = dict(extra or {})
    if isinstance(error, HarnessError):
        log_extra["error"] = error.to_dict()
    else:
        log_extra["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }

    logger.log(
        level,
        message,
        extra=log_extra,
        exc_info=(type(error), error, error.__traceback__),
    )


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "LOGGER_NAME",
    "JsonFormatter",
    "LogContextFilter",
    "configure_logging",
    "current_log_context",
    "get_logger",
    "log_context",
    "log_error",
    "reset_log_context",
    "set_log_context",
]
