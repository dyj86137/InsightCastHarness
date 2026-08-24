"""LLM Provider 注册与创建。"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from pydantic import Field

from infra.config import LLMConfig
from infra.exception import ConfigError, ValidationError as HarnessValidationError
from infra.serialization import SerializableModel
from llm.base import BaseLLM
from llm.fake import FakeLLM
from llm.openai_compatible import DEFAULT_OPENAI_MODEL, OpenAICompatibleLLM


LLMProviderFactory = Callable[[LLMConfig], BaseLLM]


class ProviderSpec(SerializableModel):
    """LLM Provider 的能力描述。"""

    name: str
    default_model: str
    supports_streaming: bool = True
    supports_structured_output: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)

    model_config = SerializableModel.config(frozen=True)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """校验 Provider 描述的基础规则。"""

        if not self.name or not self.name.strip():
            raise HarnessValidationError("ProviderSpec requires non-empty name.")
        if not self.default_model or not self.default_model.strip():
            raise HarnessValidationError(
                "ProviderSpec requires non-empty default_model.",
                details={"provider": self.name},
            )


class ProviderRegistration:
    """Provider 注册项。"""

    def __init__(self, spec: ProviderSpec, factory: LLMProviderFactory) -> None:
        self.spec = spec
        self.factory = factory


class LLMProviderRegistry:
    """管理 LLM Provider 工厂的注册表。"""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderRegistration] = {}

    def register(
        self,
        *,
        spec: ProviderSpec,
        factory: LLMProviderFactory,
        override: bool = False,
    ) -> None:
        """注册一个 Provider 工厂。"""

        provider_name = normalize_provider_name(spec.name)
        if provider_name in self._providers and not override:
            raise ConfigError(
                "LLM provider already registered.",
                details={"provider": provider_name},
            )
        self._providers[provider_name] = ProviderRegistration(spec=spec, factory=factory)

    def has(self, provider_name: str) -> bool:
        """判断 Provider 是否已经注册。"""

        return normalize_provider_name(provider_name) in self._providers

    def get_spec(self, provider_name: str) -> ProviderSpec:
        """返回 Provider 能力描述。"""

        normalized = normalize_provider_name(provider_name)
        registration = self._providers.get(normalized)
        if registration is None:
            raise unknown_provider_error(normalized, self.list_provider_names())
        return registration.spec

    def list_specs(self) -> List[ProviderSpec]:
        """返回所有 Provider 能力描述。"""

        return [registration.spec for registration in self._providers.values()]

    def list_provider_names(self) -> List[str]:
        """返回所有已注册 Provider 名称。"""

        return sorted(self._providers)

    def create(self, config: LLMConfig) -> BaseLLM:
        """根据配置创建 LLM Provider 实例。"""

        provider_name = normalize_provider_name(config.provider)
        registration = self._providers.get(provider_name)
        if registration is None:
            raise unknown_provider_error(provider_name, self.list_provider_names())
        return registration.factory(config)


def normalize_provider_name(provider_name: str) -> str:
    """统一 Provider 名称格式。"""

    if not provider_name or not provider_name.strip():
        raise ConfigError("LLM provider name cannot be empty.")
    return provider_name.strip().lower()


def unknown_provider_error(provider_name: str, available: List[str]) -> ConfigError:
    """构造未知 Provider 配置错误。"""

    return ConfigError(
        "Unknown LLM provider.",
        details={
            "provider": provider_name,
            "available_providers": available,
        },
    )


def create_fake_llm(config: LLMConfig) -> FakeLLM:
    """根据配置创建 FakeLLM。"""

    return FakeLLM(model_name=config.model)


def create_openai_llm(config: LLMConfig) -> OpenAICompatibleLLM:
    """根据配置创建 OpenAI 官方 Provider。"""

    return OpenAICompatibleLLM(
        provider_name="openai",
        model_name=resolve_model_name(config.model, DEFAULT_OPENAI_MODEL),
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        default_temperature=config.temperature,
        default_max_tokens=config.max_tokens,
        include_stream_usage=True,
    )


def create_openai_compatible_llm(config: LLMConfig) -> OpenAICompatibleLLM:
    """根据配置创建 OpenAI-compatible Provider。"""

    return OpenAICompatibleLLM(
        provider_name=normalize_provider_name(config.provider),
        model_name=resolve_model_name(config.model, DEFAULT_OPENAI_MODEL),
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        default_temperature=config.temperature,
        default_max_tokens=config.max_tokens,
    )


def resolve_model_name(config_model: str, default_model: str) -> str:
    """当只切换 provider 但未设置模型时，使用 provider 默认模型。"""

    if not config_model or not config_model.strip():
        return default_model
    if config_model == LLMConfig.model:
        return default_model
    return config_model


def create_default_registry() -> LLMProviderRegistry:
    """创建 V2 默认 Provider 注册表。"""

    registry = LLMProviderRegistry()
    registry.register(
        spec=ProviderSpec(
            name="fake",
            default_model="fake-model",
            supports_streaming=True,
            supports_structured_output=True,
        ),
        factory=create_fake_llm,
    )
    registry.register(
        spec=ProviderSpec(
            name="openai",
            default_model=DEFAULT_OPENAI_MODEL,
            supports_streaming=True,
            supports_structured_output=True,
            metadata={"api": "chat_completions"},
        ),
        factory=create_openai_llm,
    )
    registry.register(
        spec=ProviderSpec(
            name="openai-compatible",
            default_model=DEFAULT_OPENAI_MODEL,
            supports_streaming=True,
            supports_structured_output=True,
            metadata={"api": "chat_completions", "requires_base_url": "true"},
        ),
        factory=create_openai_compatible_llm,
    )
    return registry


DEFAULT_PROVIDER_REGISTRY = create_default_registry()


def create_llm(
    config: Optional[LLMConfig] = None,
    *,
    registry: Optional[LLMProviderRegistry] = None,
) -> BaseLLM:
    """使用配置和注册表创建 LLM 实例。"""

    llm_config = config or LLMConfig()
    provider_registry = registry or DEFAULT_PROVIDER_REGISTRY
    return provider_registry.create(llm_config)


__all__ = [
    "DEFAULT_PROVIDER_REGISTRY",
    "LLMProviderFactory",
    "LLMProviderRegistry",
    "ProviderRegistration",
    "ProviderSpec",
    "create_default_registry",
    "create_fake_llm",
    "create_llm",
    "create_openai_compatible_llm",
    "create_openai_llm",
    "normalize_provider_name",
    "resolve_model_name",
]
