"""用于测试和离线开发的 Fake LLM。"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Iterable, List, Optional, Union

from core.message import Message
from infra.exception import LLMProviderError, ValidationError as HarnessValidationError
from llm.base import BaseLLM, ChatRequest, ChatResponse, FinishReason, LLMUsage, StreamChunk


FakeResponse = Union[str, Message, ChatResponse, BaseException]


class FakeLLM(BaseLLM):
    """可脚本化的 LLM 测试替身。

    FakeLLM 不调用任何外部模型，适合单元测试、CI、离线开发和失败场景模拟。
    """

    def __init__(
        self,
        *,
        model_name: str = "fake-model",
        responses: Optional[Iterable[FakeResponse]] = None,
        default_response: str = "ok",
        stream_chunk_size: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        if stream_chunk_size < 0:
            raise HarnessValidationError(
                "FakeLLM stream_chunk_size cannot be negative.",
                details={"stream_chunk_size": stream_chunk_size},
            )
        if latency_seconds < 0:
            raise HarnessValidationError(
                "FakeLLM latency_seconds cannot be negative.",
                details={"latency_seconds": latency_seconds},
            )

        self._model_name = model_name
        self._responses: Deque[FakeResponse] = deque(responses or ())
        self._default_response = default_response
        self._stream_chunk_size = stream_chunk_size
        self._latency_seconds = latency_seconds
        self.requests: List[ChatRequest] = []

    @property
    def provider_name(self) -> str:
        """返回 Provider 名称。"""

        return "fake"

    @property
    def model_name(self) -> str:
        """返回默认模型名称。"""

        return self._model_name

    @property
    def remaining_responses(self) -> int:
        """返回剩余脚本化响应数量。"""

        return len(self._responses)

    @property
    def last_request(self) -> Optional[ChatRequest]:
        """返回最近一次请求。"""

        if not self.requests:
            return None
        return self.requests[-1]

    def add_response(self, response: FakeResponse) -> None:
        """追加一个脚本化响应。"""

        self._responses.append(response)

    def add_error(self, error: BaseException) -> None:
        """追加一个脚本化错误。"""

        self._responses.append(error)

    def reset(self) -> None:
        """清空请求记录和剩余响应。"""

        self.requests.clear()
        self._responses.clear()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """返回预置响应，或在没有预置响应时返回默认响应。"""

        self.requests.append(request)
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        scripted_response = self._responses.popleft() if self._responses else self._default_response
        if isinstance(scripted_response, BaseException):
            raise scripted_response

        return self._to_chat_response(scripted_response, request)

    async def stream(self, request: ChatRequest):
        """按固定大小切分文本，模拟流式输出。"""

        response = await self.chat(request)
        text = response.text

        if self._stream_chunk_size <= 0 or len(text) <= self._stream_chunk_size:
            yield StreamChunk(
                request_id=request.id,
                response_id=response.id,
                index=0,
                delta=text,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
            return

        index = 0
        for start in range(0, len(text), self._stream_chunk_size):
            end = start + self._stream_chunk_size
            is_final = end >= len(text)
            yield StreamChunk(
                request_id=request.id,
                response_id=response.id,
                index=index,
                delta=text[start:end],
                finish_reason=response.finish_reason if is_final else None,
                usage=response.usage if is_final else None,
            )
            index += 1

    def _to_chat_response(self, response: FakeResponse, request: ChatRequest) -> ChatResponse:
        if isinstance(response, ChatResponse):
            return response.clone(
                request_id=request.id,
                provider=self.provider_name,
                model=request.model or response.model or self.model_name,
            )

        if isinstance(response, Message):
            message = response
        elif isinstance(response, str):
            message = Message.assistant(response)
        else:
            raise LLMProviderError(
                "Unsupported FakeLLM response type.",
                details={"response_type": response.__class__.__name__},
            )

        return ChatResponse(
            request_id=request.id,
            message=message,
            provider=self.provider_name,
            model=request.model or self.model_name,
            finish_reason=FinishReason.STOP,
            usage=self._estimate_usage(request, message.content),
        )

    def _estimate_usage(self, request: ChatRequest, completion: str) -> LLMUsage:
        """使用简单字符规则估算 token，用于测试统计链路。"""

        prompt_text = "\n".join(message.content for message in request.messages)
        return LLMUsage(
            prompt_tokens=_estimate_tokens(prompt_text),
            completion_tokens=_estimate_tokens(completion),
        )


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数，保证测试中用量字段非空。"""

    if not text:
        return 0
    return max(1, len(text.strip()) // 4 + 1)


__all__ = [
    "FakeLLM",
    "FakeResponse",
]
