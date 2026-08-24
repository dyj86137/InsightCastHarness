"""myHarness V1 的配置管理模块。

V1 阶段保持配置层轻量：集中管理运行时、LLM、Trace、日志和工具执行所需的参数，
暂不引入远程配置中心或重量级依赖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# 定义环境变量中布尔值的所有合法写法，兼容业界常用的配置习惯
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


# 私有工具函数集（用于从环境变量中解析具体的值）
def _env_key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _get_str(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer.") from exc


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be a float.") from exc


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Environment variable {key} must be a boolean.")


def _get_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    value = env.get(key)
    if value is None or value == "":
        return default
    return Path(value).expanduser()


# 各模块配置类
@dataclass(frozen=True)
class LLMConfig:
    """LLM Provider 和 LLM 包装器共享的配置。"""

    provider: str = "fake"
    model: str = "fake-model"
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int | None = None

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_LLM") -> "LLMConfig":
        max_tokens_key = _env_key(prefix, "MAX_TOKENS")
        raw_max_tokens = env.get(max_tokens_key)

        return cls(
            provider=_get_str(env, _env_key(prefix, "PROVIDER"), cls.provider),
            model=_get_str(env, _env_key(prefix, "MODEL"), cls.model),
            api_key=env.get(_env_key(prefix, "API_KEY")) or None,
            base_url=env.get(_env_key(prefix, "BASE_URL")) or None,
            timeout_seconds=_get_float(env, _env_key(prefix, "TIMEOUT_SECONDS"), cls.timeout_seconds),
            max_retries=_get_int(env, _env_key(prefix, "MAX_RETRIES"), cls.max_retries),
            temperature=_get_float(env, _env_key(prefix, "TEMPERATURE"), cls.temperature),
            max_tokens=int(raw_max_tokens) if raw_max_tokens else None,
        )


@dataclass(frozen=True)
class TraceConfig:
    """本地 Trace 持久化配置。"""

    enabled: bool = True
    directory: Path = Path("traces")
    file_extension: str = ".jsonl"

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_TRACE") -> "TraceConfig":
        return cls(
            enabled=_get_bool(env, _env_key(prefix, "ENABLED"), cls.enabled),
            directory=_get_path(env, _env_key(prefix, "DIR"), cls.directory),
            file_extension=_get_str(env, _env_key(prefix, "FILE_EXTENSION"), cls.file_extension),
        )


@dataclass(frozen=True)
class LoggingConfig:
    """结构化日志配置。"""

    level: str = "INFO"
    json: bool = False

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_LOG") -> "LoggingConfig":
        return cls(
            level=_get_str(env, _env_key(prefix, "LEVEL"), cls.level).upper(),
            json=_get_bool(env, _env_key(prefix, "JSON"), cls.json),
        )


@dataclass(frozen=True)
class RunnerConfig:
    """单次 Agent 运行的限制配置。"""

    max_steps: int = 8
    run_timeout_seconds: float = 300.0
    checkpoint_enabled: bool = True

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_RUNNER") -> "RunnerConfig":
        return cls(
            max_steps=_get_int(env, _env_key(prefix, "MAX_STEPS"), cls.max_steps),
            run_timeout_seconds=_get_float(
                env,
                _env_key(prefix, "TIMEOUT_SECONDS"),
                cls.run_timeout_seconds,
            ),
            checkpoint_enabled=_get_bool(
                env,
                _env_key(prefix, "CHECKPOINT_ENABLED"),
                cls.checkpoint_enabled,
            ),
        )


@dataclass(frozen=True)
class ToolConfig:
    """V1 本地工具执行配置。"""

    cwd: Path = field(default_factory=Path.cwd)
    allow_mutating_tools: bool = False
    tool_timeout_seconds: float = 30.0

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_TOOL") -> "ToolConfig":
        return cls(
            cwd=_get_path(env, _env_key(prefix, "CWD"), Path.cwd()),
            allow_mutating_tools=_get_bool(
                env,
                _env_key(prefix, "ALLOW_MUTATING_TOOLS"),
                cls.allow_mutating_tools,
            ),
            tool_timeout_seconds=_get_float(
                env,
                _env_key(prefix, "TIMEOUT_SECONDS"),
                cls.tool_timeout_seconds,
            ),
        )


@dataclass(frozen=True)
class ContextConfig:
    """上下文窗口和 token 预算配置。"""

    max_messages: int = 20
    max_input_tokens: int = 8000
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0
    max_tool_result_tokens: int = 2000

    """从环境变量创建配置"""
    @classmethod
    def from_env(cls, env: Mapping[str, str], prefix: str = "MYHARNESS_CONTEXT") -> "ContextConfig":
        return cls(
            max_messages=_get_int(env, _env_key(prefix, "MAX_MESSAGES"), cls.max_messages),
            max_input_tokens=_get_int(env, _env_key(prefix, "MAX_INPUT_TOKENS"), cls.max_input_tokens),
            reserved_output_tokens=_get_int(
                env,
                _env_key(prefix, "RESERVED_OUTPUT_TOKENS"),
                cls.reserved_output_tokens,
            ),
            safety_margin_tokens=_get_int(
                env,
                _env_key(prefix, "SAFETY_MARGIN_TOKENS"),
                cls.safety_margin_tokens,
            ),
            max_tool_result_tokens=_get_int(
                env,
                _env_key(prefix, "MAX_TOOL_RESULT_TOKENS"),
                cls.max_tool_result_tokens,
            ),
        )


@dataclass(frozen=True)
class HarnessConfig:
    """贯穿整个 Harness 框架的顶层配置对象。"""

    env: str = "dev"
    project_root: Path = field(default_factory=Path.cwd)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    """从环境变量创建配置"""
    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        prefix: str = "MYHARNESS",
    ) -> "HarnessConfig":
        source = os.environ if env is None else env
        return cls(
            env=_get_str(source, _env_key(prefix, "ENV"), cls.env),
            project_root=_get_path(source, _env_key(prefix, "PROJECT_ROOT"), Path.cwd()),
            llm=LLMConfig.from_env(source, _env_key(prefix, "LLM")),
            trace=TraceConfig.from_env(source, _env_key(prefix, "TRACE")),
            logging=LoggingConfig.from_env(source, _env_key(prefix, "LOG")),
            runner=RunnerConfig.from_env(source, _env_key(prefix, "RUNNER")),
            tool=ToolConfig.from_env(source, _env_key(prefix, "TOOL")),
            context=ContextConfig.from_env(source, _env_key(prefix, "CONTEXT")),
        )

    def ensure_directories(self) -> None:
        """创建已启用组件所需的本地目录。"""

        if self.trace.enabled:
            self.trace.directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        """返回用于诊断和日志记录的普通字典。"""

        return {
            "env": self.env,
            "project_root": str(self.project_root),
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": "***" if self.llm.api_key else None,
                "base_url": self.llm.base_url,
                "timeout_seconds": self.llm.timeout_seconds,
                "max_retries": self.llm.max_retries,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "trace": {
                "enabled": self.trace.enabled,
                "directory": str(self.trace.directory),
                "file_extension": self.trace.file_extension,
            },
            "logging": {
                "level": self.logging.level,
                "json": self.logging.json,
            },
            "runner": {
                "max_steps": self.runner.max_steps,
                "run_timeout_seconds": self.runner.run_timeout_seconds,
                "checkpoint_enabled": self.runner.checkpoint_enabled,
            },
            "tool": {
                "cwd": str(self.tool.cwd),
                "allow_mutating_tools": self.tool.allow_mutating_tools,
                "tool_timeout_seconds": self.tool.tool_timeout_seconds,
            },
            "context": {
                "max_messages": self.context.max_messages,
                "max_input_tokens": self.context.max_input_tokens,
                "reserved_output_tokens": self.context.reserved_output_tokens,
                "safety_margin_tokens": self.context.safety_margin_tokens,
                "max_tool_result_tokens": self.context.max_tool_result_tokens,
            },
        }


def load_config(env: Mapping[str, str] | None = None) -> HarnessConfig:
    """从环境变量加载 V1 顶层 Harness 配置。"""

    return HarnessConfig.from_env(env)
