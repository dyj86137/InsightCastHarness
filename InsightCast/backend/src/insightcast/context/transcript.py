"""基于 myHarness 上下文预算构建转录稿上下文。"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from context.budget import TokenBudgetManager
from context.token import DEFAULT_TOKEN_ESTIMATOR, TokenEstimator, create_token_estimator_from_env
from infra.config import ContextConfig
from pydantic import Field

from insightcast.context.trace import append_context_built_event
from insightcast.domain.models import DomainModel, Interview, Transcript


DEFAULT_MAX_TRANSCRIPT_TOKENS = 60000
DEFAULT_TRANSCRIPT_BUDGET_RATIO = 0.70
DEFAULT_CHUNK_TOKEN_LIMIT = 1500
DEFAULT_RESERVED_OUTPUT_TOKENS = 4000
DEFAULT_SAFETY_MARGIN_TOKENS = 4000


class TranscriptContextChunk(DomainModel):
    """被选入或被摘要进 LLM 上下文的一段转录稿。"""

    index: int
    text: str
    estimated_tokens: int
    char_count: int
    start_char: int = 0
    end_char: int = 0


class TranscriptContext(DomainModel):
    """传给总结 LLM 的预算化转录稿上下文。"""

    text: str
    original_char_count: int = 0
    selected_char_count: int = 0
    original_estimated_tokens: int = 0
    estimated_tokens: int = 0
    token_budget: int = 0
    chunk_count: int = 0
    selected_chunks: Tuple[TranscriptContextChunk, ...] = Field(default_factory=tuple)
    dropped_chunk_indexes: Tuple[int, ...] = Field(default_factory=tuple)
    dropped_chunk_summaries: Tuple[str, ...] = Field(default_factory=tuple)
    was_compressed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryContextProvider(ABC):
    """为总结生成构建上下文。"""

    @abstractmethod
    async def build_transcript_context(
        self,
        interview: Interview,
        transcript: Transcript,
    ) -> TranscriptContext:
        """返回单个访谈的预算化转录稿上下文。"""


class TranscriptContextProvider(SummaryContextProvider):
    """为总结 Prompt 构建分块、可压缩的转录稿上下文。"""

    def __init__(
        self,
        *,
        context_config: Optional[ContextConfig] = None,
        estimator: Optional[TokenEstimator] = None,
        max_transcript_tokens: int = DEFAULT_MAX_TRANSCRIPT_TOKENS,
        transcript_budget_ratio: float = DEFAULT_TRANSCRIPT_BUDGET_RATIO,
        chunk_token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT,
    ) -> None:
        self.estimator = estimator or DEFAULT_TOKEN_ESTIMATOR
        self.context_config = context_config or ContextConfig(
            max_input_tokens=100000,
            reserved_output_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
            safety_margin_tokens=DEFAULT_SAFETY_MARGIN_TOKENS,
        )
        self.max_transcript_tokens = max_transcript_tokens
        self.transcript_budget_ratio = transcript_budget_ratio
        self.chunk_token_limit = chunk_token_limit

    async def build_transcript_context(
        self,
        interview: Interview,
        transcript: Transcript,
    ) -> TranscriptContext:
        raw_text = transcript_text_for_context(transcript)
        token_budget = self._transcript_token_budget()
        original_tokens = estimate_text_tokens(raw_text, estimator=self.estimator)
        chunks = chunk_transcript_text(
            raw_text,
            chunk_token_limit=self.chunk_token_limit,
            estimator=self.estimator,
        )

        if not raw_text or original_tokens <= token_budget:
            context = self._full_context(
                raw_text,
                chunks=chunks,
                token_budget=token_budget,
                original_tokens=original_tokens,
            )
        else:
            context = self._compressed_context(
                raw_text,
                chunks=chunks,
                token_budget=token_budget,
                original_tokens=original_tokens,
            )

        await append_context_built_event(
            name="summary_transcript",
            payload={
                "interview_id": interview.id,
                "transcript_id": transcript.id,
                "original_char_count": context.original_char_count,
                "selected_char_count": context.selected_char_count,
                "original_estimated_tokens": context.original_estimated_tokens,
                "estimated_tokens": context.estimated_tokens,
                "token_budget": context.token_budget,
                "chunk_count": context.chunk_count,
                "selected_chunk_count": len(context.selected_chunks),
                "dropped_chunks": len(context.dropped_chunk_indexes),
                "was_compressed": context.was_compressed,
                "estimator": context.metadata.get("estimator"),
                "tokenizer_encoding": context.metadata.get("tokenizer_encoding"),
            },
        )
        return context

    def _transcript_token_budget(self) -> int:
        budget_manager = TokenBudgetManager.from_config(
            self.context_config,
            estimator=self.estimator,
        )
        budget = budget_manager.build_budget()
        ratio_budget = int(budget.available_input_tokens * self.transcript_budget_ratio)
        return max(1, min(self.max_transcript_tokens, ratio_budget))

    def _full_context(
        self,
        raw_text: str,
        *,
        chunks: Sequence[TranscriptContextChunk],
        token_budget: int,
        original_tokens: int,
    ) -> TranscriptContext:
        text = raw_text.strip()
        estimated_tokens = estimate_text_tokens(text, estimator=self.estimator)
        return TranscriptContext(
            text=text,
            original_char_count=len(raw_text),
            selected_char_count=len(text),
            original_estimated_tokens=original_tokens,
            estimated_tokens=estimated_tokens,
            token_budget=token_budget,
            chunk_count=len(chunks),
            selected_chunks=tuple(chunks),
            was_compressed=False,
            metadata={
                "provider": "insightcast_transcript_context",
                "mode": "full",
                **estimator_metadata(self.estimator),
            },
        )

    def _compressed_context(
        self,
        raw_text: str,
        *,
        chunks: Sequence[TranscriptContextChunk],
        token_budget: int,
        original_tokens: int,
    ) -> TranscriptContext:
        selected_chunks = select_chunks_for_budget(
            chunks,
            token_budget=token_budget,
            summary_reserve_tokens=_summary_reserve_tokens(token_budget),
        )
        selected_indexes = {chunk.index for chunk in selected_chunks}
        dropped_chunks = [chunk for chunk in chunks if chunk.index not in selected_indexes]
        dropped_summaries = tuple(
            summarize_dropped_chunk(chunk)
            for chunk in dropped_chunks
        )
        text, kept_summaries = render_compressed_context(
            selected_chunks,
            dropped_summaries,
            original_tokens=original_tokens,
            token_budget=token_budget,
            estimator=self.estimator,
        )
        estimated_tokens = estimate_text_tokens(text, estimator=self.estimator)
        if estimated_tokens > token_budget:
            text = hard_trim_to_token_budget(
                text,
                token_budget=token_budget,
                estimator=self.estimator,
            )
            estimated_tokens = estimate_text_tokens(text, estimator=self.estimator)

        return TranscriptContext(
            text=text,
            original_char_count=len(raw_text),
            selected_char_count=sum(chunk.char_count for chunk in selected_chunks),
            original_estimated_tokens=original_tokens,
            estimated_tokens=estimated_tokens,
            token_budget=token_budget,
            chunk_count=len(chunks),
            selected_chunks=tuple(selected_chunks),
            dropped_chunk_indexes=tuple(chunk.index for chunk in dropped_chunks),
            dropped_chunk_summaries=kept_summaries,
            was_compressed=True,
            metadata={
                "provider": "insightcast_transcript_context",
                "mode": "compressed",
                "chunk_token_limit": self.chunk_token_limit,
                **estimator_metadata(self.estimator),
            },
        )


def transcript_text_for_context(transcript: Transcript) -> str:
    """返回适合上下文构建的转录稿文本。"""

    if transcript.text:
        return transcript.text.strip()
    return "\n\n".join(segment.strip() for segment in transcript.segments if segment.strip())


def chunk_transcript_text(
    text: str,
    *,
    chunk_token_limit: int,
    estimator: TokenEstimator,
) -> Tuple[TranscriptContextChunk, ...]:
    """按 token 预算将转录稿拆分为多个块。"""

    stripped = text.strip()
    if not stripped:
        return ()
    blocks = split_transcript_blocks(stripped, chunk_token_limit=chunk_token_limit, estimator=estimator)
    chunks: List[TranscriptContextChunk] = []
    current_blocks: List[str] = []
    current_start = 0
    current_end = 0

    for block_text, start_char, end_char in blocks:
        candidate_blocks = current_blocks + [block_text]
        candidate_text = "\n\n".join(candidate_blocks)
        candidate_tokens = estimate_text_tokens(candidate_text, estimator=estimator)
        if current_blocks and candidate_tokens > chunk_token_limit:
            chunks.append(
                make_chunk(
                    index=len(chunks),
                    text="\n\n".join(current_blocks),
                    start_char=current_start,
                    end_char=current_end,
                    estimator=estimator,
                )
            )
            current_blocks = [block_text]
            current_start = start_char
        else:
            if not current_blocks:
                current_start = start_char
            current_blocks = candidate_blocks
        current_end = end_char

    if current_blocks:
        chunks.append(
            make_chunk(
                index=len(chunks),
                text="\n\n".join(current_blocks),
                start_char=current_start,
                end_char=current_end,
                estimator=estimator,
            )
        )
    return tuple(chunks)


def split_transcript_blocks(
    text: str,
    *,
    chunk_token_limit: int,
    estimator: TokenEstimator,
) -> Tuple[Tuple[str, int, int], ...]:
    """将转录稿拆分为近似段落块，并保留粗略字符位置。"""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    result: List[Tuple[str, int, int]] = []
    cursor = 0
    max_chars = max(1, chunk_token_limit * chars_per_token(estimator))
    for paragraph in paragraphs:
        start = text.find(paragraph, cursor)
        if start < 0:
            start = cursor
        for piece in split_large_block(paragraph, max_chars=max_chars):
            piece_start = text.find(piece, start)
            if piece_start < 0:
                piece_start = start
            piece_end = piece_start + len(piece)
            result.append((piece, piece_start, piece_end))
            start = piece_end
        cursor = start
    return tuple(result)


def split_large_block(block: str, *, max_chars: int) -> Tuple[str, ...]:
    """按近似句子边界拆分过长段落。"""

    if len(block) <= max_chars:
        return (block,)
    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", block) if part.strip()]
    if len(sentences) <= 1:
        return tuple(block[index : index + max_chars] for index in range(0, len(block), max_chars))
    pieces: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return tuple(pieces)


def make_chunk(
    *,
    index: int,
    text: str,
    start_char: int,
    end_char: int,
    estimator: TokenEstimator,
) -> TranscriptContextChunk:
    stripped = text.strip()
    return TranscriptContextChunk(
        index=index,
        text=stripped,
        estimated_tokens=estimate_text_tokens(stripped, estimator=estimator),
        char_count=len(stripped),
        start_char=start_char,
        end_char=end_char,
    )


def select_chunks_for_budget(
    chunks: Sequence[TranscriptContextChunk],
    *,
    token_budget: int,
    summary_reserve_tokens: int,
) -> Tuple[TranscriptContextChunk, ...]:
    """选择转录稿块，并为被丢弃块的摘要预留空间。"""

    if not chunks:
        return ()
    body_budget = max(1, token_budget - summary_reserve_tokens)
    if sum(chunk.estimated_tokens for chunk in chunks) <= body_budget:
        return tuple(chunks)

    selected: List[TranscriptContextChunk] = []
    used_tokens = 0
    tail = chunks[-1] if len(chunks) > 1 else None
    tail_reserve = (
        tail.estimated_tokens
        if tail is not None and tail.estimated_tokens <= int(body_budget * 0.35)
        else 0
    )
    head_budget = max(1, body_budget - tail_reserve)
    for chunk in chunks:
        if tail is not None and chunk.index == tail.index:
            continue
        if used_tokens + chunk.estimated_tokens > head_budget and selected:
            break
        if used_tokens + chunk.estimated_tokens > body_budget:
            break
        selected.append(chunk)
        used_tokens += chunk.estimated_tokens

    if not selected:
        selected.append(chunks[0])
        used_tokens = chunks[0].estimated_tokens

    if (
        tail is not None
        and tail.index not in {chunk.index for chunk in selected}
        and used_tokens + tail.estimated_tokens <= body_budget
    ):
        selected.append(tail)

    return tuple(sorted(selected, key=lambda chunk: chunk.index))


def render_compressed_context(
    selected_chunks: Sequence[TranscriptContextChunk],
    dropped_summaries: Sequence[str],
    *,
    original_tokens: int,
    token_budget: int,
    estimator: TokenEstimator,
) -> Tuple[str, Tuple[str, ...]]:
    """渲染已选择的块，以及被丢弃块的抽取式摘要。"""

    kept_summaries = tuple(dropped_summaries)
    while True:
        text = _render_compressed_context_text(
            selected_chunks,
            kept_summaries,
            original_tokens=original_tokens,
            token_budget=token_budget,
        )
        if estimate_text_tokens(text, estimator=estimator) <= token_budget or not kept_summaries:
            return text, kept_summaries
        kept_summaries = kept_summaries[:-1]


def _render_compressed_context_text(
    selected_chunks: Sequence[TranscriptContextChunk],
    dropped_summaries: Sequence[str],
    *,
    original_tokens: int,
    token_budget: int,
) -> str:
    lines = [
        (
            "[上下文已压缩] "
            f"原始转录稿估算 tokens={original_tokens}; "
            f"转录稿 token 预算={token_budget}; "
            f"已选择块数={len(selected_chunks)}."
        )
    ]
    if dropped_summaries:
        lines.append("被丢弃块的抽取式摘要：")
        for summary in dropped_summaries:
            lines.append(f"- {summary}")
    for chunk in selected_chunks:
        lines.append(f"[转录稿块 {chunk.index + 1}]")
        lines.append(chunk.text)
    return "\n\n".join(lines).strip()


def summarize_dropped_chunk(chunk: TranscriptContextChunk, *, max_chars: int = 220) -> str:
    first_sentence = re.split(r"(?<=[。！？.!?])\s+", chunk.text.strip())[0]
    summary = first_sentence[:max_chars].strip()
    if len(first_sentence) > max_chars:
        summary += "..."
    return f"块 {chunk.index + 1}: {summary}"


def hard_trim_to_token_budget(
    text: str,
    *,
    token_budget: int,
    estimator: TokenEstimator,
) -> str:
    chars = max(1, token_budget * chars_per_token(estimator))
    trimmed = text[:chars].rstrip()
    suffix = "\n\n[上下文文本已裁剪到 token 预算]"
    while estimate_text_tokens(trimmed + suffix, estimator=estimator) > token_budget and len(trimmed) > 1:
        trimmed = trimmed[: int(len(trimmed) * 0.9)].rstrip()
    return trimmed + suffix


def transcript_context_payload(context: TranscriptContext) -> Dict[str, Any]:
    """返回用于构建 Prompt 的紧凑 JSON 可序列化 payload。"""

    return {
        "text": context.text,
        "token_budget": context.token_budget,
        "estimated_tokens": context.estimated_tokens,
        "original_estimated_tokens": context.original_estimated_tokens,
        "original_char_count": context.original_char_count,
        "selected_char_count": context.selected_char_count,
        "chunk_count": context.chunk_count,
        "selected_chunk_indexes": [chunk.index for chunk in context.selected_chunks],
        "dropped_chunk_indexes": list(context.dropped_chunk_indexes),
        "was_compressed": context.was_compressed,
        "metadata": context.metadata,
    }


def context_metadata_for_request(
    context: Optional[TranscriptContext],
) -> Optional[Dict[str, Any]]:
    if context is None:
        return None
    return {
        "provider": context.metadata.get("provider"),
        "mode": context.metadata.get("mode"),
        "estimated_tokens": context.estimated_tokens,
        "token_budget": context.token_budget,
        "original_estimated_tokens": context.original_estimated_tokens,
        "chunk_count": context.chunk_count,
        "selected_chunk_count": len(context.selected_chunks),
        "dropped_chunks": len(context.dropped_chunk_indexes),
        "was_compressed": context.was_compressed,
        "estimator": context.metadata.get("estimator"),
        "tokenizer_encoding": context.metadata.get("tokenizer_encoding"),
    }


def create_summary_context_provider_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "INSIGHTCAST_SUMMARY_CONTEXT",
    tokenizer_prefix: str = "INSIGHTCAST_TOKENIZER",
) -> TranscriptContextProvider:
    """根据 InsightCast 环境变量创建总结阶段 transcript 上下文构建器。"""

    source = os.environ if env is None else env
    context_config = ContextConfig(
        max_messages=_env_int(source, f"{prefix}_MAX_MESSAGES", 20),
        max_input_tokens=_env_int(source, f"{prefix}_MAX_INPUT_TOKENS", 100000),
        reserved_output_tokens=_env_int(
            source,
            f"{prefix}_RESERVED_OUTPUT_TOKENS",
            DEFAULT_RESERVED_OUTPUT_TOKENS,
        ),
        safety_margin_tokens=_env_int(
            source,
            f"{prefix}_SAFETY_MARGIN_TOKENS",
            DEFAULT_SAFETY_MARGIN_TOKENS,
        ),
        max_tool_result_tokens=_env_int(source, f"{prefix}_MAX_TOOL_RESULT_TOKENS", 2000),
    )
    return TranscriptContextProvider(
        context_config=context_config,
        estimator=create_token_estimator_from_env(source, prefix=tokenizer_prefix),
        max_transcript_tokens=_env_int(
            source,
            f"{prefix}_TRANSCRIPT_MAX_TOKENS",
            DEFAULT_MAX_TRANSCRIPT_TOKENS,
        ),
        transcript_budget_ratio=_env_float(
            source,
            f"{prefix}_TRANSCRIPT_BUDGET_RATIO",
            DEFAULT_TRANSCRIPT_BUDGET_RATIO,
        ),
        chunk_token_limit=_env_int(
            source,
            f"{prefix}_CHUNK_TOKEN_LIMIT",
            DEFAULT_CHUNK_TOKEN_LIMIT,
        ),
    )


def estimator_metadata(estimator: TokenEstimator) -> Dict[str, Any]:
    """提取适合写入 trace 和 LLM metadata 的 tokenizer 信息。"""

    metadata: Dict[str, Any] = {"estimator": estimator.name}
    resolved_encoding = getattr(estimator, "resolved_encoding_name", None)
    if resolved_encoding:
        metadata["tokenizer_encoding"] = resolved_encoding
    model = getattr(estimator, "model", None)
    if model:
        metadata["tokenizer_model"] = model
    fallback_reason = getattr(estimator, "fallback_reason", None)
    if fallback_reason:
        metadata["tokenizer_fallback_reason"] = fallback_reason
    return metadata


def estimate_text_tokens(text: str, *, estimator: TokenEstimator) -> int:
    return estimator.estimate_text(text).text_tokens


def chars_per_token(estimator: TokenEstimator) -> int:
    return int(getattr(estimator, "chars_per_token", 4) or 4)


def _summary_reserve_tokens(token_budget: int) -> int:
    return max(80, min(300, int(token_budget * 0.15)))


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    return float(value)


__all__ = [
    "DEFAULT_CHUNK_TOKEN_LIMIT",
    "DEFAULT_MAX_TRANSCRIPT_TOKENS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_TRANSCRIPT_BUDGET_RATIO",
    "SummaryContextProvider",
    "TranscriptContext",
    "TranscriptContextChunk",
    "TranscriptContextProvider",
    "chunk_transcript_text",
    "context_metadata_for_request",
    "create_summary_context_provider_from_env",
    "hard_trim_to_token_budget",
    "select_chunks_for_budget",
    "summarize_dropped_chunk",
    "transcript_context_payload",
    "transcript_text_for_context",
]
