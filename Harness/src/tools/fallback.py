"""工具级降级策略。

Fallback 负责在主工具失败后选择备用工具，并把降级过程写入最终 ToolResult。
具体工具执行仍由 ToolExecutor 负责，本模块只保存策略和结构化记录。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import Field

from core.tool import ToolResult
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from tools.registry import normalize_tool_name


NON_FALLBACK_ERROR_CODES = {
    "tool_validation_error",
    "validation_error",
}


class ToolFallbackAttemptRecord(SerializableModel):
    """一次 fallback 尝试记录。"""

    source_tool_name: str
    fallback_tool_name: str
    succeeded: bool
    result: Dict[str, Any] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)


class ToolFallbackPolicy(SerializableModel):
    """工具级 fallback 策略。

    fallback_map 使用主工具名称到备用工具名称列表的映射。
    默认不会对参数校验失败做 fallback，因为这类错误通常说明调用参数本身不合法。
    """

    enabled: bool = True
    fallback_map: Dict[str, Tuple[str, ...]] = Field(default_factory=dict)
    fallback_on_validation_error: bool = False
    fallback_read_only_only: bool = True

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**normalize_fallback_policy_data(data))
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 fallback 策略配置。"""

        for source_tool_name, fallback_names in self.fallback_map.items():
            normalize_tool_name(source_tool_name)
            if not fallback_names:
                raise HarnessValidationError(
                    "ToolFallbackPolicy fallback list cannot be empty.",
                    details={"tool_name": source_tool_name},
                )
            for fallback_name in fallback_names:
                normalize_tool_name(fallback_name)
                if fallback_name == source_tool_name:
                    raise HarnessValidationError(
                        "Tool cannot fallback to itself.",
                        details={"tool_name": source_tool_name},
                    )

    def get_fallback_names(self, tool_name: str) -> Tuple[str, ...]:
        """返回某个工具的备用工具名称列表。"""

        return self.fallback_map.get(normalize_tool_name(tool_name), ())

    def should_fallback(
        self,
        *,
        source_tool_name: str,
        result: ToolResult,
        source_is_read_only: bool,
    ) -> bool:
        """判断主工具失败结果是否应该进入 fallback。"""

        if not self.enabled:
            return False
        if not result.is_error:
            return False
        if not self.get_fallback_names(source_tool_name):
            return False
        if self.fallback_read_only_only and not source_is_read_only:
            return False
        if result.error and result.error.get("code") in NON_FALLBACK_ERROR_CODES:
            return self.fallback_on_validation_error
        return True


def normalize_fallback_policy_data(data: Mapping[str, object]) -> Dict[str, object]:
    """规范化 fallback_map 中的工具名称。"""

    normalized = dict(data)
    raw_map = normalized.get("fallback_map")
    if raw_map is None:
        return normalized
    if not isinstance(raw_map, Mapping):
        raise HarnessValidationError(
            "ToolFallbackPolicy fallback_map must be a mapping.",
            details={"actual_type": raw_map.__class__.__name__},
        )

    fallback_map: Dict[str, Tuple[str, ...]] = {}
    for source_tool_name, fallback_names in raw_map.items():
        source_name = normalize_tool_name(str(source_tool_name))
        if isinstance(fallback_names, str):
            normalized_fallbacks = (normalize_tool_name(fallback_names),)
        else:
            try:
                normalized_fallbacks = tuple(
                    normalize_tool_name(str(item)) for item in fallback_names
                )
            except TypeError as exc:
                raise HarnessValidationError(
                    "ToolFallbackPolicy fallback targets must be iterable.",
                    details={"tool_name": source_name},
                    cause=exc,
                ) from exc
        fallback_map[source_name] = normalized_fallbacks

    normalized["fallback_map"] = fallback_map
    return normalized


def build_fallback_attempt_record(
    *,
    source_tool_name: str,
    fallback_tool_name: str,
    result: ToolResult,
) -> ToolFallbackAttemptRecord:
    """从备用工具结果创建 fallback 记录。"""

    payload: Dict[str, Any] = {
        "tool_result_id": result.id,
        "tool_call_id": result.tool_call_id,
        "status": result.status.value,
        "output": result.output,
    }
    if result.error:
        payload["error"] = result.error
    if result.metadata:
        payload["metadata"] = result.metadata

    return ToolFallbackAttemptRecord(
        source_tool_name=source_tool_name,
        fallback_tool_name=fallback_tool_name,
        succeeded=not result.is_error,
        result=payload,
    )


def attach_fallback_metadata(
    result: ToolResult,
    *,
    source_result: ToolResult,
    source_tool_name: str,
    attempted_names: Tuple[str, ...],
    records: List[ToolFallbackAttemptRecord],
) -> ToolResult:
    """把 fallback 过程写入 ToolResult.metadata。"""

    if not records:
        return result

    metadata = dict(result.metadata)
    metadata["fallback"] = {
        "used": any(record.succeeded for record in records),
        "source_tool_name": source_tool_name,
        "source_tool_result_id": source_result.id,
        "source_error": source_result.error,
        "attempted_names": list(attempted_names),
        "attempts": [record.to_dict(exclude_none=True) for record in records],
    }
    return result.clone(metadata=metadata)


__all__ = [
    "NON_FALLBACK_ERROR_CODES",
    "ToolFallbackAttemptRecord",
    "ToolFallbackPolicy",
    "attach_fallback_metadata",
    "build_fallback_attempt_record",
    "normalize_fallback_policy_data",
]
