"""基于 LLM 的访谈结构化总结。"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

from pydantic import Field, field_validator

from core.message import Message
from infra.config import LLMConfig
from llm import BaseLLM, ChatRequest, create_llm
from memory import MemoryInjection

from insightcast.agent.llm_trace import ensure_traced_llm
from insightcast.evidence import EvidenceClaim
from insightcast.domain.enums import TranscriptStatus
from insightcast.domain.models import (
    ExtractedInsight,
    DomainModel,
    Interview,
    InterviewSummary,
    Transcript,
)
from insightcast.context import (
    DEFAULT_MAX_TRANSCRIPT_TOKENS,
    SummaryContextProvider,
    TranscriptContext,
    TranscriptContextProvider,
    context_metadata_for_request,
    transcript_context_payload,
)
from insightcast.memory import SummaryMemoryProvider


SUMMARY_SYSTEM_PROMPT = """你是一名商业情报分析师，负责把行业领袖访谈整理成结构化情报摘要。

仅返回符合以下格式的 JSON:
{
  "summary": "string",
  "key_points": ["string"],
  "potential_opportunities": ["string"],
  "industry_judgements": ["string"],
  "mentioned_companies": ["string"],
  "mentioned_products": ["string"],
  "audience": ["string"],
  "novelty_assessment": "string|null",
  "evidence_claims": [
    {
      "field": "summary|potential_opportunities",
      "claim": "string",
      "model_confidence": number,
      "evidence": [
        {
          "text": "exact excerpt from the supplied transcript",
          "segment_id": "string|null"
        }
      ]
    }
  ],
  "insights": [
    {
      "statement": "string",
      "category": "string",
      "confidence": number,
      "evidence": "string|null",
      "mentioned_companies": ["string"],
      "mentioned_products": ["string"]
    }
  ]
}

总结要求：
- 用中文输出，保留公司、产品、人物等专有名词的常用英文名。
- 证据优先于信息量：只能使用输入 transcript 明确表达或由多处原文可以稳妥合并出的内容，不要为了显得完整而补充常识、行业背景或模型自己的判断。
- summary 应完整概括访谈主题、核心观点、关键事实和主要结论。完整 transcript 时控制在 300～500 个中文字符，硬上限为 600 个中文字符；如果只有标题、简介或 partial transcript，控制在 150～300 个中文字符，并明确说明证据不足。
- summary 的字符数是硬性约束：完整 transcript 请写成 5～8 句、目标 400～500 个非空白字符，必须输出 300～600 个非空白字符，输出前请自行检查；少于 300 个字符时必须继续补充原文支持的主题、观点、事实或结论，不能返回过短摘要。partial transcript 只能输出 150～300 个非空白字符。
- summary 每句话只保留一个主要事实或观点。不要把多个独立事实、因果关系、结果判断和业务建议塞进同一句。
- 只有 transcript 明确支持时，才可以写人物归因、数字、时间、公司/产品名称、因果关系、市场规模、竞争结果或“导致/因此/否则”等结果性表述。原文没有明确数字或名称时不得自行补全。
- summary 的字符数是硬性约束：如果需要增加长度，只能增加新的、能够单独找到原文证据的事实或观点，不能扩写已有句子或加入未经证实的推论。
- 允许对多段原文做忠实的简洁概括，但概括不能改变原文的范围、确定性或因果关系。无法被原文支持的内容应删掉或改写得更保守，而不是保留后再降低 confidence。
- potential_opportunities 输出 0～3 条基于访谈文本的机会方向。机会不是事实预测，只能在 transcript 明确出现相关需求、趋势、技术能力或行动建议时，给出有限度的“可关注”推导；不要把推导写成已经发生或必然成功的事实。
- 关注机会中不要擅自扩展受益人群、行业、产品类型、技术案例、收益、替代关系或时间周期。除非 transcript 明确提到，否则不要列出具体行业、技术或应用清单。
- 如果 transcript 没有足够依据支持一个完整机会方向，返回空数组。不要为了凑够 1～3 条而推测。
- evidence_claims 为摘要和关注机会提供审计证据。摘要按主要事实或观点拆分，关注机会每条对应一条声明；每条声明都应引用能支持该声明核心内容的最短原文片段，并尽量提供对应 segment_id。不要把证据写进 summary 或 potential_opportunities 文本。
- evidence_claims.claim 必须逐字对应最终 summary 中的一句话，或 potential_opportunities 中的一条最终内容；不要重新改写 claim。
- evidence_claims 只能引用 prompt 中提供的 transcript 原文，不要引用历史 memory、模型常识或自己改写的内容；如果一条声明的核心内容没有可引用原文，删掉或重写该声明，不要只提交一条无法覆盖整句的证据。
- 如果 transcript 只是标题/简介/show notes，要明确降低确定性，不要编造访谈细节。
- 如果 prompt 中提供 memory，只把它作为历史观点参照，用于判断观点变化；不要把历史观点当成本次访谈内容。
- novelty_assessment 可以说明“暂无历史观点可比，仅基于本次材料判断”。
"""


class InterviewSummaryOutput(DomainModel):
    """LLM 生成访谈摘要时的结构化输出。"""

    summary: str
    key_points: Tuple[str, ...] = Field(default_factory=tuple)
    potential_opportunities: Tuple[str, ...] = Field(default_factory=tuple)
    industry_judgements: Tuple[str, ...] = Field(default_factory=tuple)
    mentioned_companies: Tuple[str, ...] = Field(default_factory=tuple)
    mentioned_products: Tuple[str, ...] = Field(default_factory=tuple)
    audience: Tuple[str, ...] = Field(default_factory=tuple)
    novelty_assessment: Optional[str] = None
    evidence_claims: Tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    insights: Tuple[ExtractedInsight, ...] = Field(default_factory=tuple)

    @field_validator("summary")
    def validate_summary(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("summary.summary cannot be empty")
        return text


class InterviewSummarizer(ABC):
    """访谈结构化总结接口。"""

    @abstractmethod
    async def summarize_async(
        self,
        interview: Interview,
        transcript: Transcript,
    ) -> "InterviewSummaryResult":
        """为单条访谈生成结构化摘要。"""


class InterviewSummaryResult(DomainModel):
    """一次摘要生成的用户内容和独立证据声明。"""

    summary: InterviewSummary
    evidence_claims: Tuple[EvidenceClaim, ...] = Field(default_factory=tuple)


class LLMInterviewSummarizer(InterviewSummarizer):
    """使用 myHarness BaseLLM 的结构化输出能力总结访谈。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout_seconds: Optional[float] = None,
        max_transcript_chars: int = 0,
        memory_provider: Optional[SummaryMemoryProvider] = None,
        context_provider: Optional[SummaryContextProvider] = None,
        context_enabled: bool = True,
    ) -> None:
        self.llm = ensure_traced_llm(llm)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_transcript_chars = max_transcript_chars
        self.memory_provider = memory_provider
        self.context_provider = (
            context_provider
            if context_provider is not None
            else (
                TranscriptContextProvider(
                    max_transcript_tokens=(
                        max(1, max_transcript_chars // 4 + 1)
                        if max_transcript_chars > 0
                        else DEFAULT_MAX_TRANSCRIPT_TOKENS
                    )
                )
                if context_enabled
                else None
            )
        )

    async def summarize_async(
        self,
        interview: Interview,
        transcript: Transcript,
    ) -> InterviewSummaryResult:
        memory_injection = await self._build_memory_injection(interview)
        transcript_context = await self._build_transcript_context(interview, transcript)
        metadata = {
            "task": "insightcast_interview_summary",
            "interview_id": interview.id,
            "transcript_id": transcript.id,
            "person_ids": list(interview.person_ids),
            "source_id": interview.source_id,
        }
        memory_metadata = memory_metadata_for_request(memory_injection)
        if memory_metadata:
            metadata["memory"] = memory_metadata
        context_metadata = context_metadata_for_request(transcript_context)
        if context_metadata:
            metadata["context"] = context_metadata

        request = ChatRequest.create(
            messages=(
                Message.system(SUMMARY_SYSTEM_PROMPT),
                Message.user(
                    build_summary_prompt(
                        interview,
                        transcript,
                        max_transcript_chars=self.max_transcript_chars,
                        memory_injection=memory_injection,
                        transcript_context=transcript_context,
                    )
                ),
            ),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            metadata=metadata,
        )
        output = await self.llm.structured(request, InterviewSummaryOutput)
        validate_evidence_claim_alignment(output)
        return InterviewSummaryResult(
            summary=output_to_summary(
                interview,
                output,
                transcript_status=transcript.status,
                llm=self.llm,
                memory_injection=memory_injection,
                transcript_context=transcript_context,
            ),
            evidence_claims=output.evidence_claims,
        )

    async def _build_memory_injection(
        self,
        interview: Interview,
    ) -> Optional[MemoryInjection]:
        if self.memory_provider is None:
            return None
        return await self.memory_provider.build_summary_memory(interview)

    async def _build_transcript_context(
        self,
        interview: Interview,
        transcript: Transcript,
    ) -> Optional[TranscriptContext]:
        if self.context_provider is None:
            return None
        return await self.context_provider.build_transcript_context(interview, transcript)


def build_summary_prompt(
    interview: Interview,
    transcript: Transcript,
    *,
    max_transcript_chars: int = 0,
    memory_injection: Optional[MemoryInjection] = None,
    transcript_context: Optional[TranscriptContext] = None,
) -> str:
    """为单条访谈构造结构化总结 prompt。"""

    if transcript_context is None:
        transcript_text = transcript_text_for_prompt(transcript, max_chars=max_transcript_chars)
        context_payload = None
    else:
        transcript_text = transcript_context.text
        context_payload = transcript_context_payload(transcript_context)
    payload = {
        "interview": {
            "id": interview.id,
            "title": interview.title,
            "url": str(interview.url),
            "type": interview.type.value,
            "format": interview.format.value,
            "source_name": interview.source_name,
            "person_names": list(interview.person_names),
            "industries": [industry.value for industry in interview.industries],
            "published_at": interview.published_at.isoformat()
            if interview.published_at
            else None,
            "duration_seconds": interview.duration_seconds,
        },
        "transcript": {
            "id": transcript.id,
            "status": transcript.status.value,
            "source": transcript.source.value,
            "is_partial_or_metadata_only": transcript.status.value != "ready",
            "text": transcript_text,
            "segments": [
                {
                    "segment_id": f"transcript_segment_{index + 1:04d}",
                    "text": segment,
                }
                for index, segment in enumerate(transcript.segments)
                if segment.strip()
            ],
        },
    }
    if context_payload:
        payload["transcript"]["context"] = context_payload
    memory_payload = memory_payload_for_prompt(memory_injection)
    if memory_payload:
        payload["memory"] = memory_payload
    return (
        "请基于下面的访谈信息和可用文本生成结构化商业情报摘要。"
        "如果文本只是简介或元数据，请在判断中保持克制，不要编造细节。"
        "如果提供了 memory.person_history，请用它比较历史观点和本次材料，"
        "但不要把历史观点当成本次访谈事实。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def transcript_text_for_prompt(transcript: Transcript, *, max_chars: int) -> str:
    """提取并截断传给 LLM 的 transcript 文本。"""

    if transcript.text:
        text = transcript.text
    else:
        text = "\n\n".join(transcript.segments)
    text = text.strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n\n[内容已截断]"
    return text


def output_to_summary(
    interview: Interview,
    output: InterviewSummaryOutput,
    *,
    transcript_status: TranscriptStatus,
    llm: BaseLLM,
    memory_injection: Optional[MemoryInjection] = None,
    transcript_context: Optional[TranscriptContext] = None,
) -> InterviewSummary:
    """把 LLM 结构化输出转换为领域层的访谈摘要模型。"""

    summary_text = validate_summary_length(output.summary, transcript_status)

    metadata = {
        "summarizer": "llm",
        "provider": llm.provider_name,
        "model": llm.model_name,
    }
    memory_metadata = memory_metadata_for_request(memory_injection)
    if memory_metadata:
        metadata["memory"] = memory_metadata
    context_metadata = context_metadata_for_request(transcript_context)
    if context_metadata:
        metadata["context"] = context_metadata

    return InterviewSummary(
        interview_id=interview.id,
        summary=summary_text,
        key_points=output.key_points,
        potential_opportunities=output.potential_opportunities,
        industry_judgements=output.industry_judgements,
        mentioned_companies=output.mentioned_companies,
        mentioned_products=output.mentioned_products,
        audience=output.audience,
        novelty_assessment=output.novelty_assessment,
        insights=output.insights,
        model=llm.model_name,
        metadata=metadata,
    )


SUMMARY_READY_MIN_CHARS = 300
SUMMARY_READY_MAX_CHARS = 600
SUMMARY_PARTIAL_MIN_CHARS = 150
SUMMARY_PARTIAL_MAX_CHARS = 300


def validate_summary_length(value: str, status: TranscriptStatus) -> str:
    """按 Transcript 完整度校验摘要字符数。"""

    text = str(value).strip()
    char_count = len(re.sub(r"\s+", "", text))
    if status == TranscriptStatus.READY:
        minimum, maximum = SUMMARY_READY_MIN_CHARS, SUMMARY_READY_MAX_CHARS
    else:
        minimum, maximum = SUMMARY_PARTIAL_MIN_CHARS, SUMMARY_PARTIAL_MAX_CHARS
    if not minimum <= char_count <= maximum:
        raise ValueError(
            f"summary.summary must contain {minimum}-{maximum} non-whitespace characters "
            f"for transcript status {status.value}; got {char_count}"
        )
    return text


def validate_evidence_claim_alignment(output: InterviewSummaryOutput) -> None:
    """确保证据声明对应结构化摘要中的最终内容。"""

    summary_sentences = split_summary_sentences(output.summary)
    opportunity_items = tuple(output.potential_opportunities)
    for claim in output.evidence_claims:
        if claim.field == "summary":
            candidates = summary_sentences
        else:
            candidates = opportunity_items
        if not any(
            normalize_claim_text(claim.claim) == normalize_claim_text(candidate)
            for candidate in candidates
        ):
            raise ValueError(
                "evidence claim does not match a final summary or opportunity sentence: "
                f"{claim.claim}"
            )


def split_summary_sentences(value: str) -> Tuple[str, ...]:
    """按常见中英文句末标点拆分最终摘要句子。"""

    return tuple(
        part.strip()
        for part in re.split(r"(?<=[。！？!?])\s*", str(value).strip())
        if part.strip()
    )


def normalize_claim_text(value: str) -> str:
    """比较 claim 与最终内容时忽略空白和句末标点差异。"""

    return re.sub(r"\s+", "", str(value).strip()).rstrip("。！？!?")


def memory_payload_for_prompt(
    memory_injection: Optional[MemoryInjection],
) -> Optional[Dict[str, Any]]:
    if memory_injection is None or not memory_injection.content:
        return None
    return {
        "person_history": {
            "instruction": (
                "Use this only as historical viewpoint context for novelty_assessment."
            ),
            "content": memory_injection.content,
            "memory_ids": [result.record.id for result in memory_injection.results],
        }
    }


def memory_metadata_for_request(
    memory_injection: Optional[MemoryInjection],
) -> Optional[Dict[str, Any]]:
    if memory_injection is None:
        return None
    return {
        "provider": memory_injection.metadata.get("provider"),
        "result_count": memory_injection.metadata.get("result_count", 0),
        "injected_count": len(memory_injection.results),
        "memory_ids": [result.record.id for result in memory_injection.results],
    }


def create_summary_llm_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "INSIGHTCAST_SUMMARY_LLM",
) -> BaseLLM:
    """根据环境变量创建用于结构化总结的 myHarness LLM。"""

    source = os.environ if env is None else env
    config = LLMConfig(
        provider=_env_get(source, f"{prefix}_PROVIDER", "openai"),
        model=_env_get(source, f"{prefix}_MODEL", "gpt-4.1-mini"),
        api_key=(
            source.get(f"{prefix}_API_KEY")
            or source.get("OPENAI_API_KEY")
            or source.get("MYHARNESS_LLM_API_KEY")
        ),
        base_url=source.get(f"{prefix}_BASE_URL") or source.get("MYHARNESS_LLM_BASE_URL"),
        timeout_seconds=float(_env_get(source, f"{prefix}_TIMEOUT_SECONDS", "60")),
        temperature=float(_env_get(source, f"{prefix}_TEMPERATURE", "0")),
        max_tokens=_optional_int(source.get(f"{prefix}_MAX_TOKENS")),
    )
    return create_llm(config)


def _env_get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


__all__ = [
    "InterviewSummarizer",
    "InterviewSummaryResult",
    "InterviewSummaryOutput",
    "LLMInterviewSummarizer",
    "SUMMARY_SYSTEM_PROMPT",
    "build_summary_prompt",
    "create_summary_llm_from_env",
    "memory_metadata_for_request",
    "memory_payload_for_prompt",
    "normalize_claim_text",
    "output_to_summary",
    "split_summary_sentences",
    "transcript_text_for_prompt",
    "validate_evidence_claim_alignment",
    "validate_summary_length",
]
