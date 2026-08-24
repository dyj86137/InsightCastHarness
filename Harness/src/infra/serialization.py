"""myHarness V1 的统一序列化规范。"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ValidationError as PydanticValidationError

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - compatibility with Pydantic v1
    ConfigDict = None  # type: ignore[assignment]

from .exception import SerializationError


JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Dict[str, "JsonValue"], List["JsonValue"]]

T = TypeVar("T", bound="SerializableModel")
PYDANTIC_V2 = hasattr(BaseModel, "model_validate")


def _build_model_config(**overrides: Any) -> Any:
    config: Dict[str, Any] = {
        "extra": "forbid",
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
        "validate_by_name": True,
    }
    config.update(overrides)
    if PYDANTIC_V2 and ConfigDict is not None:
        return ConfigDict(**config)
    config.setdefault("json_encoders", {
        datetime: lambda value: value.isoformat(),
        date: lambda value: value.isoformat(),
        Path: str,
        UUID: str,
        Enum: lambda value: value.value,
    })
    return config


def _to_plain_value(value: Any) -> Any:
    """把 Python 对象转换成适合 JSON/Trace 记录的普通值。"""

    if isinstance(value, BaseModel):
        return _to_plain_value(_dump_base_model(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if hasattr(value, "unicode_string"):
        return value.unicode_string()
    if value.__class__.__module__.startswith("pydantic.networks"):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, set):
        return [_to_plain_value(item) for item in sorted(value, key=str)]
    return value


def _serialization_error(
    message: str,
    *,
    model_name: str,
    cause: BaseException | None = None,
    details: Mapping[str, Any] | None = None,
) -> SerializationError:
    error_details = {"model": model_name}
    if details:
        error_details.update(details)
    return SerializationError(message, details=error_details, cause=cause)


class SerializableModel(BaseModel):
    """所有核心协议对象的统一基类。

    核心协议对象统一继承该类，避免模块之间传递裸字典。
    裸字典只允许出现在序列化边界，例如从 JSON、Trace、外部 API 还原对象时。
    """

    if PYDANTIC_V2:
        model_config = _build_model_config()
    else:
        class Config:
            extra = "forbid"
            validate_assignment = True
            arbitrary_types_allowed = True
            populate_by_name = True
            validate_by_name = True
            json_encoders = {
                datetime: lambda value: value.isoformat(),
                date: lambda value: value.isoformat(),
                Path: str,
                UUID: str,
                Enum: lambda value: value.value,
            }

    @staticmethod
    def config(**overrides: Any) -> Any:
        return _build_model_config(**overrides)

    def to_dict(
        self,
        *,
        exclude_none: bool = False,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """转换为适合日志、Trace、JSON 存储的普通字典。"""

        data = _dump_base_model(
            self,
            exclude_none=exclude_none,
            by_alias=by_alias,
        )
        return _to_plain_value(data)

    def to_json(
        self,
        *,
        exclude_none: bool = False,
        by_alias: bool = False,
        indent: int | None = None,
    ) -> str:
        """转换为 JSON 字符串。"""

        return json.dumps(
            self.to_dict(exclude_none=exclude_none, by_alias=by_alias),
            ensure_ascii=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any] | T) -> T:
        """从普通字典还原为强类型协议对象。"""

        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise _serialization_error(
                "SerializableModel.from_dict expects a mapping.",
                model_name=cls.__name__,
                details={"actual_type": data.__class__.__name__},
            )

        try:
            return cls(**dict(data))
        except PydanticValidationError as exc:
            raise _serialization_error(
                "Failed to deserialize mapping into model.",
                model_name=cls.__name__,
                cause=exc,
                details={"errors": exc.errors()},
            ) from exc

    @classmethod
    def from_json(cls: Type[T], raw: str | bytes | bytearray) -> T:
        """从 JSON 字符串还原为强类型协议对象。"""

        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise _serialization_error(
                "Failed to decode JSON.",
                model_name=cls.__name__,
                cause=exc,
            ) from exc

        return cls.from_dict(data)

    def clone(self: T, **updates: Any) -> T:
        """复制当前对象，并用强校验方式应用字段更新。"""

        data = self.to_dict()
        data.update(updates)
        return self.__class__.from_dict(data)

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        by_alias: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """提供 Pydantic v2 风格的导出方法名。"""

        if mode == "json":
            return _dump_base_model(
                self,
                mode=mode,
                include=include,
                exclude=exclude,
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                by_alias=by_alias,
                **kwargs,
            )
        return _to_plain_value(
            _dump_base_model(
                self,
                mode=mode,
                include=include,
                exclude=exclude,
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                by_alias=by_alias,
                **kwargs,
            )
        )

    @classmethod
    def model_validate(cls: Type[T], data: Mapping[str, Any] | T) -> T:
        """提供 Pydantic v2 风格的校验方法名。"""

        return cls.from_dict(data)

    def dict(
        self,
        *,
        exclude_none: bool = False,
        by_alias: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """兼容旧代码中的 Pydantic v1 `dict()` 调用。"""

        return self.model_dump(
            exclude_none=exclude_none,
            by_alias=by_alias,
            **kwargs,
        )

    def json(
        self,
        *,
        exclude_none: bool = False,
        by_alias: bool = False,
        **kwargs: Any,
    ) -> str:
        """兼容旧代码中的 Pydantic v1 `json()` 调用。"""

        indent = kwargs.pop("indent", None)
        return self.to_json(
            exclude_none=exclude_none,
            by_alias=by_alias,
            indent=indent,
        )


def _dump_base_model(
    value: BaseModel,
    *,
    mode: str = "json",
    include: Any = None,
    exclude: Any = None,
    exclude_none: bool = False,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    by_alias: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """用当前 Pydantic 版本的原生方式导出 BaseModel。"""

    if PYDANTIC_V2:
        return BaseModel.model_dump(
            value,
            mode=mode,
            include=include,
            exclude=exclude,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            by_alias=by_alias,
            **kwargs,
        )
    return value.dict(
        include=include,
        exclude=exclude,
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        exclude_defaults=exclude_defaults,
        by_alias=by_alias,
    )


def serialize(value: Any) -> Any:
    """序列化任意框架对象为普通 Python 值。"""

    return _to_plain_value(value)


def deserialize(model_cls: Type[T], data: Mapping[str, Any] | T) -> T:
    """把普通字典还原为指定的 SerializableModel 子类。"""

    if not issubclass(model_cls, SerializableModel):
        raise SerializationError(
            "deserialize expects a SerializableModel subclass.",
            details={"model": getattr(model_cls, "__name__", str(model_cls))},
        )
    return model_cls.from_dict(data)


__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "SerializableModel",
    "deserialize",
    "serialize",
]
