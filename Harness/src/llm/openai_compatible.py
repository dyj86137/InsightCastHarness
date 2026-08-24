"""OpenAI Chat Completions 兼容 Provider。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from core.message import Message, MessageRole
from infra.exception import LLMProviderError
from llm.base import BaseLLM, ChatRequest, ChatResponse, FinishReason, LLMUsage, StreamChunk


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class OpenAICompatibleLLM(BaseLLM):
    """基于 OpenAI Python SDK 的 Chat Completions 兼容适配器。

    该类把 OpenAI-compatible 响应翻译成 myHarness 内部统一的
    ChatResponse / StreamChunk。`openai` SDK 是可选依赖，只有实例化
    该 Provider 时才需要安装。
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_OPENAI_MODEL,
        provider_name: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        max_tokens_param: str = "max_tokens",
        include_stream_usage: bool = False,
    ) -> None:
        if not model_name or not model_name.strip():
            raise LLMProviderError(
                "OpenAI-compatible model_name cannot be empty.",
                retryable=False,
            )
        if not provider_name or not provider_name.strip():
            raise LLMProviderError(
                "OpenAI-compatible provider_name cannot be empty.",
                retryable=False,
            )
        if max_tokens_param not in ("max_tokens", "max_completion_tokens"):
            raise LLMProviderError(
                "Unsupported OpenAI max token parameter.",
                details={"max_tokens_param": max_tokens_param},
                retryable=False,
            )

        self._model_name = model_name
        self._provider_name = provider_name.strip().lower()
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._max_tokens_param = max_tokens_param
        self._include_stream_usage = include_stream_usage
        self._client = create_async_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        """返回 Provider 名称。"""

        return self._provider_name

    @property
    def model_name(self) -> str:
        """返回默认模型名称。"""

        return self._model_name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """执行非流式 Chat Completions 调用。"""

        params = self._build_chat_params(request, stream=False)
        try:
            response = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise wrap_openai_error(
                exc,
                provider=self.provider_name,
                model=params["model"],
                request_id=request.id,
            ) from exc
        return self._to_chat_response(response, request)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """执行流式 Chat Completions 调用。"""

        params = self._build_chat_params(request, stream=True)
        try:
            stream = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise wrap_openai_error(
                exc,
                provider=self.provider_name,
                model=params["model"],
                request_id=request.id,
            ) from exc

        index = 0
        response_id: Optional[str] = None
        async for event in stream:
            response_id = response_id or getattr(event, "id", None)
            usage = usage_from_openai(getattr(event, "usage", None))
            choices = list(getattr(event, "choices", []) or [])
            for choice in choices:
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                finish_reason = map_finish_reason(getattr(choice, "finish_reason", None))
                if not text and finish_reason is None:
                    continue
                yield StreamChunk(
                    request_id=request.id,
                    response_id=response_id,
                    index=index,
                    delta=text or "",
                    finish_reason=finish_reason,
                    usage=usage if finish_reason is not None else None,
                    metadata={
                        "provider": self.provider_name,
                        "model": params["model"],
                    },
                )
                index += 1

    def _build_chat_params(self, request: ChatRequest, *, stream: bool) -> Dict[str, Any]:
        model = request.model or self.model_name
        params: Dict[str, Any] = {
            "model": model,
            "messages": [message_to_openai(item) for item in request.messages],
            "stream": stream,
        }

        temperature = (
            request.temperature
            if request.temperature is not None
            else self._default_temperature
        )
        if temperature is not None:
            params["temperature"] = temperature

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self._default_max_tokens
        )
        if max_tokens is not None:
            params[self._max_tokens_param] = max_tokens

        if request.timeout_seconds is not None:
            params["timeout"] = request.timeout_seconds
        if request.stop:
            params["stop"] = list(request.stop)
        if stream and self._include_stream_usage:
            params["stream_options"] = {"include_usage": True}

        extra_body = request.metadata.get("openai_extra_body")
        if isinstance(extra_body, dict):
            params.update(extra_body)
        return params

    def _to_chat_response(self, response: Any, request: ChatRequest) -> ChatResponse:
        choices = list(getattr(response, "choices", []) or [])
        if not choices:
            raise LLMProviderError(
                "OpenAI-compatible response contains no choices.",
                details={
                    "provider": self.provider_name,
                    "model": request.model or self.model_name,
                    "request_id": request.id,
                },
            )

        choice = choices[0]
        response_message = getattr(choice, "message", None)
        content = getattr(response_message, "content", None) if response_message else None
        if content is None:
            content = ""

        return ChatResponse(
            request_id=request.id,
            message=Message.assistant(content),
            provider=self.provider_name,
            model=getattr(response, "model", None) or request.model or self.model_name,
            finish_reason=map_finish_reason(getattr(choice, "finish_reason", None))
            or FinishReason.UNKNOWN,
            usage=usage_from_openai(getattr(response, "usage", None)),
            raw=to_plain_openai_object(response),
            metadata={
                "openai_response_id": getattr(response, "id", None),
                "object": getattr(response, "object", None),
            },
        )


def create_async_openai_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout_seconds: Optional[float],
) -> Any:
    """延迟导入并创建 AsyncOpenAI 客户端。"""

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise LLMProviderError(
            "OpenAI Python SDK is not installed.",
            details={"install": "python -m pip install openai"},
            retryable=False,
            cause=exc,
        ) from exc

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    return AsyncOpenAI(**kwargs)


def message_to_openai(message: Message) -> Dict[str, Any]:
    """把内部 Message 转成 OpenAI Chat Completions message。"""

    payload: Dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.role == MessageRole.TOOL and message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def usage_from_openai(usage: Any) -> LLMUsage:
    """把 OpenAI usage 对象转成统一 LLMUsage。"""

    if usage is None:
        return LLMUsage.empty()

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_tokens = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        metadata={"total_tokens": int(getattr(usage, "total_tokens", 0) or 0)},
    )


def map_finish_reason(reason: Any) -> Optional[FinishReason]:
    """映射 OpenAI finish_reason。"""

    if reason is None:
        return None
    normalized = str(reason).strip().lower()
    if normalized == "stop":
        return FinishReason.STOP
    if normalized == "length":
        return FinishReason.LENGTH
    if normalized == "tool_calls":
        return FinishReason.TOOL_CALLS
    if normalized in ("content_filter", "error"):
        return FinishReason.ERROR
    return FinishReason.UNKNOWN


def wrap_openai_error(
    error: Exception,
    *,
    provider: str,
    model: str,
    request_id: str,
) -> LLMProviderError:
    """把 OpenAI SDK 异常包装为统一异常。"""

    name = error.__class__.__name__
    retryable = name in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
    return LLMProviderError(
        "OpenAI-compatible provider call failed.",
        details={
            "provider": provider,
            "model": model,
            "request_id": request_id,
            "error_type": name,
        },
        retryable=retryable,
        cause=error,
    )


def to_plain_openai_object(value: Any) -> Any:
    """把 SDK 对象尽量转换为可序列化字典。"""

    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain_openai_object(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_openai_object(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_openai_object(item) for key, item in value.items()}
    return str(value)


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "OpenAICompatibleLLM",
    "create_async_openai_client",
    "map_finish_reason",
    "message_to_openai",
    "to_plain_openai_object",
    "usage_from_openai",
    "wrap_openai_error",
]
