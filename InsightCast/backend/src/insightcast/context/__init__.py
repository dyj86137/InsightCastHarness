"""Context adapters for myHarness-backed InsightCast prompts."""

from insightcast.context.trace import (
    ContextTraceContext,
    append_context_built_event,
    context_trace_scope,
)
from insightcast.context.transcript import (
    DEFAULT_CHUNK_TOKEN_LIMIT,
    DEFAULT_MAX_TRANSCRIPT_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    DEFAULT_TRANSCRIPT_BUDGET_RATIO,
    SummaryContextProvider,
    TranscriptContext,
    TranscriptContextChunk,
    TranscriptContextProvider,
    chunk_transcript_text,
    context_metadata_for_request,
    create_summary_context_provider_from_env,
    hard_trim_to_token_budget,
    select_chunks_for_budget,
    summarize_dropped_chunk,
    transcript_context_payload,
    transcript_text_for_context,
)


__all__ = [
    "ContextTraceContext",
    "DEFAULT_CHUNK_TOKEN_LIMIT",
    "DEFAULT_MAX_TRANSCRIPT_TOKENS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_TRANSCRIPT_BUDGET_RATIO",
    "SummaryContextProvider",
    "TranscriptContext",
    "TranscriptContextChunk",
    "TranscriptContextProvider",
    "append_context_built_event",
    "chunk_transcript_text",
    "context_metadata_for_request",
    "context_trace_scope",
    "create_summary_context_provider_from_env",
    "hard_trim_to_token_budget",
    "select_chunks_for_budget",
    "summarize_dropped_chunk",
    "transcript_context_payload",
    "transcript_text_for_context",
]
