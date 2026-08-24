"""基于 myHarness RunEvent JSONL trace 的 LLM trace 辅助工具。"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from core.event import RunEvent, RunEventLevel, RunEventType, error_to_payload
from core.state import AgentState
from llm import BaseLLM, ChatRequest, ChatResponse
from loops.base import LoopRunContext
from loops.react import ReactLoop
from insightcast.harness_events import HarnessEventRecorder


DEFAULT_MAX_TRACE_CONTENT_CHARS = 8000
TRACE_MAX_CONTENT_CHARS_ENV = "INSIGHTCAST_LLM_TRACE_MAX_CONTENT_CHARS"
TRACE_DEBUG_ENV = "INSIGHTCAST_LLM_TRACE_DEBUG"

class LLMTraceContext:
    """当前 pipeline 中用于 LLM 调用的 trace 上下文。"""

    def __init__(
        self,
        *,
        run_id: str,
        trace_dir: Optional[Path],
        stage: str,
        step: int,
        event_recorder: Optional[HarnessEventRecorder] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.run_id = run_id
        self.trace_dir = trace_dir
        self.stage = stage
        self.step = step
        self.event_recorder = event_recorder
        self.env = env


_CURRENT_LLM_TRACE_CONTEXT: ContextVar[Optional[LLMTraceContext]] = ContextVar(
    "insightcast_llm_trace_context",
    default=None,
)


class TracedLLM(BaseLLM):
    """在存在上下文时发出 myHarness LLM trace 事件的 BaseLLM 包装器。"""

    def __init__(self, inner: BaseLLM) -> None:
        self.inner = inner

    @property
    def provider_name(self) -> str:
        return self.inner.provider_name

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    async def chat(self, request: ChatRequest) -> ChatResponse:
        context = _CURRENT_LLM_TRACE_CONTEXT.get()
        if context is None:
            return await self.inner.chat(request)

        await _append_llm_event(
            context,
            RunEvent.llm_requested(
                run_id=context.run_id,
                provider=self.provider_name,
                model=request.model or self.model_name,
                step=context.step,
                payload=safe_request_payload(request, context),
            ),
        )
        started_at = time.perf_counter()
        try:
            response = await self.inner.chat(request)
        except Exception as exc:
            duration_ms = _duration_ms(started_at)
            await _append_llm_event(
                context,
                RunEvent.create(
                    run_id=context.run_id,
                    event_type=RunEventType.LLM_FAILED,
                    step=context.step,
                    level=RunEventLevel.ERROR,
                    message="LLM 调用失败。",
                    payload=safe_failure_payload(
                        request,
                        context,
                        duration_ms=duration_ms,
                    ),
                    error=error_to_payload(exc),
                ),
            )
            raise

        duration_ms = _duration_ms(started_at)
        await _append_llm_event(
            context,
            RunEvent.llm_responded(
                run_id=context.run_id,
                provider=response.provider,
                model=response.model,
                step=context.step,
                payload=safe_response_payload(
                    request,
                    response,
                    context,
                    duration_ms=duration_ms,
                ),
            ),
        )
        return response


def ensure_traced_llm(llm: BaseLLM) -> BaseLLM:
    """确保一个 LLM 只被 TracedLLM 包装一次。"""

    if isinstance(llm, TracedLLM):
        return llm
    return TracedLLM(llm)


@contextmanager
def llm_trace_scope(
    *,
    run_id: str,
    trace_dir: Optional[Path],
    stage: str,
    step: int,
    event_recorder: Optional[HarnessEventRecorder] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Iterator[None]:
    """为当前代码块中的所有 TracedLLM 调用设置 trace 上下文。"""

    if trace_dir is None and event_recorder is None:
        yield
        return

    recorder = event_recorder or HarnessEventRecorder(trace_dir=trace_dir)
    token = _CURRENT_LLM_TRACE_CONTEXT.set(
        LLMTraceContext(
            run_id=run_id,
            trace_dir=trace_dir,
            stage=stage,
            step=step,
            event_recorder=recorder,
            env=env,
        )
    )
    try:
        yield
    finally:
        _CURRENT_LLM_TRACE_CONTEXT.reset(token)


async def _append_llm_event(context: LLMTraceContext, event: RunEvent) -> None:
    if context.event_recorder is None:
        return
    await context.event_recorder.record_event(event)


class InsightCastReactLoop(ReactLoop):
    """使用 InsightCast LLM trace payload 的 ReAct loop。"""

    def __init__(
        self,
        *,
        trace_metadata: Optional[Mapping[str, Any]] = None,
        trace_env: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.trace_metadata = dict(trace_metadata or {})
        self.trace_env = trace_env

    async def _call_llm(
        self,
        state: AgentState,
        context: LoopRunContext,
    ) -> ChatResponse:
        request = await self._build_request(state)
        trace_context = LLMTraceContext(
            run_id=state.run_id,
            trace_dir=None,
            stage=str(self.trace_metadata.get("stage") or state.metadata.get("stage") or ""),
            step=state.step,
            env=self.trace_env,
        )
        request = request.clone(
            metadata={
                **request.metadata,
                **state.metadata,
                **self.trace_metadata,
            }
        )

        await self.record_event(
            context,
            RunEvent.llm_requested(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                provider=self.llm.provider_name,
                model=request.model or self.llm.model_name,
                payload=safe_request_payload(request, trace_context),
            ),
            state=state,
        )

        started_at = time.perf_counter()
        try:
            response = await self.llm.chat(request)
        except Exception as exc:
            duration_ms = _duration_ms(started_at)
            await self.record_event(
                context,
                RunEvent.create(
                    run_id=state.run_id,
                    session_id=state.session_id,
                    event_type=RunEventType.LLM_FAILED,
                    step=state.step,
                    level=RunEventLevel.ERROR,
                    message="LLM call failed.",
                    payload=safe_failure_payload(
                        request,
                        trace_context,
                        duration_ms=duration_ms,
                    ),
                    error=error_to_payload(exc),
                ),
                state=state,
            )
            raise

        duration_ms = _duration_ms(started_at)
        await self.record_event(
            context,
            RunEvent.llm_responded(
                run_id=state.run_id,
                session_id=state.session_id,
                step=state.step,
                provider=response.provider,
                model=response.model,
                payload=safe_response_payload(
                    request,
                    response,
                    trace_context,
                    duration_ms=duration_ms,
                ),
            ),
            state=state,
        )
        return response


def safe_request_payload(
    request: ChatRequest,
    context: LLMTraceContext,
) -> Dict[str, Any]:
    max_chars = _max_content_chars(context.env)
    payload: Dict[str, Any] = {
        "request_id": request.id,
        "messages": _serialize_messages(request.messages, max_chars=max_chars),
    }
    if _debug_enabled(context.env):
        payload["debug_metadata"] = dict(request.metadata)
    return payload


def safe_response_payload(
    request: ChatRequest,
    response: ChatResponse,
    context: LLMTraceContext,
    *,
    duration_ms: float,
) -> Dict[str, Any]:
    max_chars = _max_content_chars(context.env)
    output = _text_payload(response.text, max_chars=max_chars)
    payload: Dict[str, Any] = {
        "request_id": request.id,
        "response_id": response.id,
        "finish_reason": response.finish_reason.value,
        "duration_ms": duration_ms,
        "output_text": output["content"],
    }
    if _debug_enabled(context.env):
        payload["debug_metadata"] = dict(request.metadata)
        if response.metadata:
            payload["debug_response_metadata"] = dict(response.metadata)
    return payload


def safe_failure_payload(
    request: ChatRequest,
    context: LLMTraceContext,
    *,
    duration_ms: float,
) -> Dict[str, Any]:
    payload = safe_request_payload(request, context)
    payload.update(
        {
            "duration_ms": duration_ms,
        }
    )
    return payload


def _serialize_messages(
    messages: Sequence[Any],
    *,
    max_chars: int,
) -> list[Dict[str, Any]]:
    result = []
    for message in messages:
        item = {
            "id": message.id,
            "role": message.role.value,
            **_text_payload(message.content, max_chars=max_chars),
        }
        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        result.append(item)
    return result


def _text_payload(text: str, *, max_chars: int) -> Dict[str, Any]:
    value = str(text)
    content_hash = _content_hash(value)
    if max_chars >= 0 and len(value) > max_chars:
        return {
            "content": value[:max_chars],
            "content_chars": len(value),
            "truncated": True,
            "content_hash": content_hash,
        }
    return {
        "content": value,
        "content_chars": len(value),
        "truncated": False,
    }


def _max_content_chars(env: Optional[Mapping[str, str]]) -> int:
    return _env_int(
        env,
        TRACE_MAX_CONTENT_CHARS_ENV,
        DEFAULT_MAX_TRACE_CONTENT_CHARS,
    )


def _debug_enabled(env: Optional[Mapping[str, str]]) -> bool:
    return _env_flag(env, TRACE_DEBUG_ENV, False)


def _env_int(
    env: Optional[Mapping[str, str]],
    key: str,
    default: int,
) -> int:
    source = os.environ if env is None else env
    value = source.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_flag(
    env: Optional[Mapping[str, str]],
    key: str,
    default: bool,
) -> bool:
    source = os.environ if env is None else env
    value = source.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


__all__ = [
    "DEFAULT_MAX_TRACE_CONTENT_CHARS",
    "InsightCastReactLoop",
    "LLMTraceContext",
    "TRACE_DEBUG_ENV",
    "TRACE_MAX_CONTENT_CHARS_ENV",
    "TracedLLM",
    "ensure_traced_llm",
    "llm_trace_scope",
    "safe_failure_payload",
    "safe_request_payload",
    "safe_response_payload",
]
