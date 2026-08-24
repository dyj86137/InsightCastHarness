"""上下文 token 估算工具。

V2 先提供启发式估算器，避免上下文系统直接依赖某个模型 tokenizer。
后续接入 tiktoken、SentencePiece 或模型 Provider tokenizer 时，只需要新增 TokenEstimator 实现。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pydantic import Field

from core.message import Message
from infra.exception import ValidationError as HarnessValidationError
from infra.serialization import SerializableModel


class TextTokenEstimate(SerializableModel):
    """一段文本的 token 估算结果。"""

    text_tokens: int
    char_count: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if self.text_tokens < 0:
            raise HarnessValidationError(
                "TextTokenEstimate text_tokens cannot be negative.",
                details={"text_tokens": self.text_tokens},
            )
        if self.char_count < 0:
            raise HarnessValidationError(
                "TextTokenEstimate char_count cannot be negative.",
                details={"char_count": self.char_count},
            )


class MessageTokenEstimate(SerializableModel):
    """单条消息的 token 估算结果。"""

    message_id: str
    role: str
    content_tokens: int
    overhead_tokens: int
    name_tokens: int = 0
    total_tokens: int
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验消息 token 估算结果。"""

        for field_name in ("content_tokens", "overhead_tokens", "name_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if value < 0:
                raise HarnessValidationError(
                    "MessageTokenEstimate token fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )
        expected = self.content_tokens + self.overhead_tokens + self.name_tokens
        if self.total_tokens != expected:
            raise HarnessValidationError(
                "MessageTokenEstimate total_tokens is inconsistent.",
                details={"expected": expected, "actual": self.total_tokens},
            )


class MessagesTokenEstimate(SerializableModel):
    """多条消息的 token 估算结果。"""

    message_estimates: Tuple[MessageTokenEstimate, ...] = Field(default_factory=tuple)
    total_tokens: int = 0
    message_count: int = 0
    metadata: Dict[str, object] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验多消息 token 估算结果。"""

        if self.total_tokens < 0:
            raise HarnessValidationError(
                "MessagesTokenEstimate total_tokens cannot be negative.",
                details={"total_tokens": self.total_tokens},
            )
        if self.message_count < 0:
            raise HarnessValidationError(
                "MessagesTokenEstimate message_count cannot be negative.",
                details={"message_count": self.message_count},
            )
        if self.message_count != len(self.message_estimates):
            raise HarnessValidationError(
                "MessagesTokenEstimate message_count is inconsistent.",
                details={
                    "message_count": self.message_count,
                    "estimate_count": len(self.message_estimates),
                },
            )


class TokenEstimator(ABC):
    """token 估算器抽象。"""

    name: str = "base"

    @abstractmethod
    def estimate_text(self, text: str) -> TextTokenEstimate:
        """估算文本 token 数。"""

    @abstractmethod
    def estimate_message(self, message: Message) -> MessageTokenEstimate:
        """估算单条消息 token 数。"""

    def estimate_messages(self, messages: Iterable[Message]) -> MessagesTokenEstimate:
        """估算多条消息 token 数。"""

        estimates = tuple(self.estimate_message(message) for message in messages)
        return MessagesTokenEstimate(
            message_estimates=estimates,
            total_tokens=sum(item.total_tokens for item in estimates),
            message_count=len(estimates),
            metadata={"estimator": self.name},
        )


class HeuristicTokenEstimator(TokenEstimator):
    """基于字符数的启发式 token 估算器。"""

    name = "heuristic"

    def __init__(
        self,
        *,
        chars_per_token: int = 4,
        message_overhead_tokens: int = 4,
        name_overhead_tokens: int = 1,
        tool_message_overhead_tokens: int = 2,
        min_text_tokens: int = 1,
    ) -> None:
        if chars_per_token <= 0:
            raise HarnessValidationError(
                "HeuristicTokenEstimator chars_per_token must be positive.",
                details={"chars_per_token": chars_per_token},
            )
        for field_name, value in (
            ("message_overhead_tokens", message_overhead_tokens),
            ("name_overhead_tokens", name_overhead_tokens),
            ("tool_message_overhead_tokens", tool_message_overhead_tokens),
            ("min_text_tokens", min_text_tokens),
        ):
            if value < 0:
                raise HarnessValidationError(
                    "HeuristicTokenEstimator token fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )

        self.chars_per_token = chars_per_token
        self.message_overhead_tokens = message_overhead_tokens
        self.name_overhead_tokens = name_overhead_tokens
        self.tool_message_overhead_tokens = tool_message_overhead_tokens
        self.min_text_tokens = min_text_tokens

    def estimate_text(self, text: str) -> TextTokenEstimate:
        """按字符长度粗略估算文本 token 数。"""

        if not text:
            return TextTokenEstimate(
                text_tokens=0,
                char_count=0,
                metadata={"estimator": self.name, "chars_per_token": self.chars_per_token},
            )

        stripped = text.strip()
        char_count = len(stripped)
        if char_count == 0:
            tokens = 0
        else:
            tokens = max(self.min_text_tokens, char_count // self.chars_per_token + 1)
        return TextTokenEstimate(
            text_tokens=tokens,
            char_count=char_count,
            metadata={"estimator": self.name, "chars_per_token": self.chars_per_token},
        )

    def estimate_message(self, message: Message) -> MessageTokenEstimate:
        """估算单条消息 token 数。"""

        text_estimate = self.estimate_text(message.content)
        name_tokens = self.name_overhead_tokens if message.name else 0
        overhead_tokens = self.message_overhead_tokens
        if message.is_tool:
            overhead_tokens += self.tool_message_overhead_tokens
        total_tokens = text_estimate.text_tokens + overhead_tokens + name_tokens
        return MessageTokenEstimate(
            message_id=message.id,
            role=message.role.value,
            content_tokens=text_estimate.text_tokens,
            overhead_tokens=overhead_tokens,
            name_tokens=name_tokens,
            total_tokens=total_tokens,
            metadata={
                "estimator": self.name,
                "char_count": text_estimate.char_count,
                "has_name": bool(message.name),
                "is_tool": message.is_tool,
            },
        )


class TiktokenTokenEstimator(TokenEstimator):
    """基于 tiktoken 的文本 token 估算器。

    文本内容使用真实 tokenizer 计数；Message 的角色/name/tool 开销仍按可配置常量估算。
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        encoding_name: Optional[str] = None,
        fallback: Optional[TokenEstimator] = None,
        strict: bool = False,
        message_overhead_tokens: int = 4,
        name_overhead_tokens: int = 1,
        tool_message_overhead_tokens: int = 2,
    ) -> None:
        for field_name, value in (
            ("message_overhead_tokens", message_overhead_tokens),
            ("name_overhead_tokens", name_overhead_tokens),
            ("tool_message_overhead_tokens", tool_message_overhead_tokens),
        ):
            if value < 0:
                raise HarnessValidationError(
                    "TiktokenTokenEstimator token fields cannot be negative.",
                    details={"field": field_name, "value": value},
                )

        self.requested_name = "tiktoken"
        self.model = model
        self.encoding_name = encoding_name
        self.fallback = fallback or HeuristicTokenEstimator()
        self.strict = strict
        self.message_overhead_tokens = message_overhead_tokens
        self.name_overhead_tokens = name_overhead_tokens
        self.tool_message_overhead_tokens = tool_message_overhead_tokens
        self.fallback_reason: Optional[str] = None
        self._encoding: Any = None

        try:
            self._encoding = self._load_encoding()
        except Exception as exc:
            if strict:
                raise HarnessValidationError(
                    "tiktoken tokenizer is unavailable or misconfigured.",
                    details={
                        "model": model,
                        "encoding_name": encoding_name,
                        "install": "python -m pip install tiktoken",
                    },
                    cause=exc,
                ) from exc
            self.fallback_reason = f"{exc.__class__.__name__}: {exc}"

        self.name = "tiktoken" if self._encoding is not None else self.fallback.name

    @property
    def available(self) -> bool:
        """返回当前是否真正启用了 tiktoken。"""

        return self._encoding is not None

    @property
    def resolved_encoding_name(self) -> Optional[str]:
        """返回最终使用的 tokenizer encoding 名称。"""

        if self._encoding is None:
            return None
        return str(getattr(self._encoding, "name", None) or self.encoding_name or "")

    def estimate_text(self, text: str) -> TextTokenEstimate:
        """使用 tokenizer 估算文本 token 数。"""

        if self._encoding is None:
            return self._fallback_text_estimate(text)

        if not text:
            return TextTokenEstimate(
                text_tokens=0,
                char_count=0,
                metadata=self._metadata(),
            )

        stripped = text.strip()
        return TextTokenEstimate(
            text_tokens=self._count_tokens(stripped) if stripped else 0,
            char_count=len(stripped),
            metadata=self._metadata(),
        )

    def estimate_message(self, message: Message) -> MessageTokenEstimate:
        """估算单条消息 token 数。"""

        if self._encoding is None:
            return self._fallback_message_estimate(message)

        text_estimate = self.estimate_text(message.content)
        name_tokens = self.name_overhead_tokens if message.name else 0
        overhead_tokens = self.message_overhead_tokens
        if message.is_tool:
            overhead_tokens += self.tool_message_overhead_tokens
        total_tokens = text_estimate.text_tokens + overhead_tokens + name_tokens
        metadata = self._metadata()
        metadata.update(
            {
                "char_count": text_estimate.char_count,
                "has_name": bool(message.name),
                "is_tool": message.is_tool,
                "message_overhead_policy": "configured_approximation",
            }
        )
        return MessageTokenEstimate(
            message_id=message.id,
            role=message.role.value,
            content_tokens=text_estimate.text_tokens,
            overhead_tokens=overhead_tokens,
            name_tokens=name_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )

    def _load_encoding(self) -> Any:
        tiktoken = import_module("tiktoken")
        if self.encoding_name:
            return tiktoken.get_encoding(self.encoding_name)
        if self.model:
            try:
                return tiktoken.encoding_for_model(self.model)
            except KeyError:
                pass
        for fallback_encoding in ("o200k_base", "cl100k_base"):
            try:
                return tiktoken.get_encoding(fallback_encoding)
            except Exception:
                continue
        return tiktoken.get_encoding("gpt2")

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self._encoding.encode(text, disallowed_special=()))
        except TypeError:
            return len(self._encoding.encode(text))

    def _metadata(self) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "estimator": self.name,
            "provider": "tiktoken",
        }
        if self.model:
            metadata["model"] = self.model
        if self.resolved_encoding_name:
            metadata["encoding"] = self.resolved_encoding_name
        return metadata

    def _fallback_text_estimate(self, text: str) -> TextTokenEstimate:
        estimate = self.fallback.estimate_text(text)
        metadata = dict(estimate.metadata)
        metadata.update(
            {
                "requested_estimator": self.requested_name,
                "fallback_reason": self.fallback_reason,
            }
        )
        return TextTokenEstimate(
            text_tokens=estimate.text_tokens,
            char_count=estimate.char_count,
            metadata=metadata,
        )

    def _fallback_message_estimate(self, message: Message) -> MessageTokenEstimate:
        estimate = self.fallback.estimate_message(message)
        metadata = dict(estimate.metadata)
        metadata.update(
            {
                "requested_estimator": self.requested_name,
                "fallback_reason": self.fallback_reason,
            }
        )
        return MessageTokenEstimate(
            message_id=estimate.message_id,
            role=estimate.role,
            content_tokens=estimate.content_tokens,
            overhead_tokens=estimate.overhead_tokens,
            name_tokens=estimate.name_tokens,
            total_tokens=estimate.total_tokens,
            metadata=metadata,
        )


DEFAULT_TOKEN_ESTIMATOR = HeuristicTokenEstimator()


def create_token_estimator_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "MYHARNESS_TOKENIZER",
) -> TokenEstimator:
    """根据环境变量创建 token 估算器。"""

    source = os.environ if env is None else env
    provider = _env_get(
        source,
        f"{prefix}_PROVIDER",
        _env_get(source, f"{prefix}_ESTIMATOR", "heuristic"),
    ).strip().lower()
    fallback = HeuristicTokenEstimator(
        chars_per_token=_env_int(source, f"{prefix}_CHARS_PER_TOKEN", 4),
        message_overhead_tokens=_env_int(source, f"{prefix}_MESSAGE_OVERHEAD_TOKENS", 4),
        name_overhead_tokens=_env_int(source, f"{prefix}_NAME_OVERHEAD_TOKENS", 1),
        tool_message_overhead_tokens=_env_int(
            source,
            f"{prefix}_TOOL_MESSAGE_OVERHEAD_TOKENS",
            2,
        ),
    )

    if provider in ("", "heuristic", "char", "chars", "character"):
        return fallback
    if provider in ("tiktoken", "tokenizer"):
        return TiktokenTokenEstimator(
            model=_env_optional_str(source, f"{prefix}_MODEL"),
            encoding_name=_env_optional_str(source, f"{prefix}_ENCODING"),
            fallback=fallback,
            strict=_env_bool(source, f"{prefix}_STRICT", False),
            message_overhead_tokens=_env_int(source, f"{prefix}_MESSAGE_OVERHEAD_TOKENS", 4),
            name_overhead_tokens=_env_int(source, f"{prefix}_NAME_OVERHEAD_TOKENS", 1),
            tool_message_overhead_tokens=_env_int(
                source,
                f"{prefix}_TOOL_MESSAGE_OVERHEAD_TOKENS",
                2,
            ),
        )

    raise HarnessValidationError(
        "Unsupported token estimator provider.",
        details={"provider": provider, "prefix": prefix},
    )


def estimate_text_tokens(
    text: str,
    *,
    estimator: Optional[TokenEstimator] = None,
) -> int:
    """估算文本 token 数。"""

    selected_estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
    return selected_estimator.estimate_text(text).text_tokens


def estimate_message_tokens(
    message: Message,
    *,
    estimator: Optional[TokenEstimator] = None,
) -> int:
    """估算单条消息 token 数。"""

    selected_estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
    return selected_estimator.estimate_message(message).total_tokens


def estimate_messages_tokens(
    messages: Iterable[Message],
    *,
    estimator: Optional[TokenEstimator] = None,
) -> int:
    """估算多条消息 token 数。"""

    selected_estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
    return selected_estimator.estimate_messages(messages).total_tokens


def estimate_messages_by_id(
    messages: Iterable[Message],
    *,
    estimator: Optional[TokenEstimator] = None,
) -> Dict[str, int]:
    """返回 message_id 到 token 估算值的映射。"""

    selected_estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
    estimates = selected_estimator.estimate_messages(messages)
    return {
        item.message_id: item.total_tokens
        for item in estimates.message_estimates
    }


def _env_get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _env_optional_str(env: Mapping[str, str], key: str) -> Optional[str]:
    value = env.get(key)
    if value is None or value == "":
        return None
    return value


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise HarnessValidationError(
            "Token estimator environment variable must be an integer.",
            details={"key": key, "value": value},
        ) from exc


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise HarnessValidationError(
        "Token estimator environment variable must be a boolean.",
        details={"key": key, "value": value},
    )


__all__ = [
    "DEFAULT_TOKEN_ESTIMATOR",
    "HeuristicTokenEstimator",
    "MessageTokenEstimate",
    "MessagesTokenEstimate",
    "TextTokenEstimate",
    "TiktokenTokenEstimator",
    "TokenEstimator",
    "create_token_estimator_from_env",
    "estimate_message_tokens",
    "estimate_messages_by_id",
    "estimate_messages_tokens",
    "estimate_text_tokens",
]
