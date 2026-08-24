"""读取外部 configs 下可编辑的配置文件（支持 JSON/YAML），经过格式校验、模型校验后，批量写入 store 仓储层，完成项目基础数据的初始化与更新"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from pydantic import Field, model_validator

from infra.exception import ConfigError, SerializationError
from insightcast.domain.models import (
    DomainModel,
    IndustryProfile,
    Person,
    Source,
    UserInterest,
)
from insightcast.storage.repositories import InsightCastRepositories


PathLike = Union[str, Path]


class InsightCastConfig(DomainModel):
    """阶段 1 使用的可编辑业务种子配置。"""

    industries: Tuple[IndustryProfile, ...] = Field(default_factory=tuple)
    people: Tuple[Person, ...] = Field(default_factory=tuple)
    sources: Tuple[Source, ...] = Field(default_factory=tuple)
    user_interests: Tuple[UserInterest, ...] = Field(
        default_factory=tuple,
        alias="interests",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_interest_alias(cls, values: Any) -> Any:
        """同时支持顶层 `interests` 和 `user_interests`。"""

        if not isinstance(values, Mapping):
            return values
        payload = dict(values)
        if "interests" not in payload and "user_interests" in payload:
            payload["interests"] = payload["user_interests"]
        payload.pop("user_interests", None)
        return payload


class ConfigApplyResult(DomainModel):
    """配置写入 repository 后的数量统计。"""

    industries: int = 0
    people: int = 0
    sources: int = 0
    user_interests: int = 0


def load_config(path: PathLike) -> InsightCastConfig:
    """从单个配置文件或配置目录加载 InsightCast 配置。"""

    resolved = Path(path)
    if resolved.is_dir():
        return load_config_dir(resolved)
    return load_config_file(resolved)


def load_config_file(path: PathLike) -> InsightCastConfig:
    """从一个完整配置文件加载 InsightCast 配置。"""

    resolved = Path(path)
    payload = _read_data_file(resolved)
    if not isinstance(payload, Mapping):
        raise _config_error(
            "InsightCast config root must be an object.",
            path=resolved,
            details={"actual_type": payload.__class__.__name__},
        )
    return _parse_config(payload, path=resolved)


def load_config_dir(path: PathLike) -> InsightCastConfig:
    """从配置目录加载拆分的 seed files。

    支持的文件名：
    - industries.json/yaml/yml
    - people.json/yaml/yml
    - sources.json/yaml/yml
    - interests.json/yaml/yml 或 user_interests.json/yaml/yml

    每个文件可以直接是列表，也可以是包含同名字段的对象。
    """

    resolved = Path(path)
    if not resolved.exists():
        raise _config_error("Config directory does not exist.", path=resolved)
    if not resolved.is_dir():
        raise _config_error("Config path is not a directory.", path=resolved)

    payload: Dict[str, Any] = {}
    for field_name, stems in _collection_stems().items():
        collection_path = _find_collection_file(resolved, stems)
        if collection_path is None:
            continue
        payload[field_name] = _read_collection_file(collection_path, field_name)
    return _parse_config(payload, path=resolved)


def apply_config(
    config: InsightCastConfig,
    repositories: InsightCastRepositories,
    *,
    overwrite: bool = True,
) -> ConfigApplyResult:
    """把配置中的种子数据写入 InsightCast repositories。

    默认使用 upsert 行为，方便反复调整配置后重新导入。传入
    `overwrite=False` 时会使用 create，若 ID 已存在则由 repository 抛出重复错误。
    """

    return ConfigApplyResult(
        industries=_save_all(
            repositories.industry_profiles,
            config.industries,
            overwrite=overwrite,
        ),
        people=_save_all(repositories.people, config.people, overwrite=overwrite),
        sources=_save_all(repositories.sources, config.sources, overwrite=overwrite),
        user_interests=_save_all(
            repositories.user_interests,
            config.user_interests,
            overwrite=overwrite,
        ),
    )


def _collection_stems() -> Dict[str, Tuple[str, ...]]:
    return {
        "industries": ("industries", "industry_profiles"),
        "people": ("people",),
        "sources": ("sources",),
        "interests": ("interests", "user_interests"),
    }


def _find_collection_file(root: Path, stems: Sequence[str]) -> Optional[Path]:
    for stem in stems:
        for suffix in (".json", ".yaml", ".yml"):
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _read_collection_file(path: Path, field_name: str) -> Any:
    payload = _read_data_file(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        if field_name in payload:
            return payload[field_name]
        if field_name == "interests" and "user_interests" in payload:
            return payload["user_interests"]
        raise _config_error(
            "Collection config object is missing expected field.",
            path=path,
            details={"expected_field": field_name},
        )
    raise _config_error(
        "Collection config must be a list or an object.",
        path=path,
        details={"actual_type": payload.__class__.__name__},
    )


def _read_data_file(path: Path) -> Any:
    if not path.exists():
        raise _config_error("Config file does not exist.", path=path)
    if not path.is_file():
        raise _config_error("Config path is not a file.", path=path)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _config_error("Failed to read config file.", path=path, cause=exc) from exc

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json(raw, path)
    if suffix in (".yaml", ".yml"):
        return _read_yaml(raw, path)
    raise _config_error(
        "Unsupported config file extension.",
        path=path,
        details={"supported_extensions": [".json", ".yaml", ".yml"]},
    )


def _read_json(raw: str, path: Path) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _config_error("Invalid JSON config.", path=path, cause=exc) from exc


def _read_yaml(raw: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _config_error(
            "YAML config requires PyYAML. Use JSON or install pyyaml.",
            path=path,
            cause=exc,
        ) from exc

    try:
        return yaml.safe_load(raw) or {}
    except Exception as exc:
        raise _config_error("Invalid YAML config.", path=path, cause=exc) from exc


def _parse_config(payload: Mapping[str, Any], *, path: Path) -> InsightCastConfig:
    try:
        return InsightCastConfig.from_dict(payload)
    except SerializationError as exc:
        raise _config_error("Invalid InsightCast config.", path=path, cause=exc) from exc


def _save_all(repository: Any, models: Iterable[Any], *, overwrite: bool) -> int:
    count = 0
    for model in models:
        if overwrite:
            repository.save(model)
        else:
            repository.create(model)
        count += 1
    return count


def _config_error(
    message: str,
    *,
    path: Path,
    cause: Optional[BaseException] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> ConfigError:
    error_details = {"path": str(path)}
    if details:
        error_details.update(details)
    return ConfigError(message, details=error_details, cause=cause)


__all__ = [
    "ConfigApplyResult",
    "InsightCastConfig",
    "PathLike",
    "apply_config",
    "load_config",
    "load_config_dir",
    "load_config_file",
]
