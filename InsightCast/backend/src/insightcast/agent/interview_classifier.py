"""基于 LLM 的访谈识别。"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Tuple

from pydantic import Field, field_validator

from core.message import Message
from infra.config import LLMConfig
from llm import BaseLLM, ChatRequest, create_llm

from insightcast.agent.llm_trace import ensure_traced_llm
from insightcast.domain.enums import InterviewType
from insightcast.domain.models import (
    CandidateItem,
    DomainModel,
    InterviewDecision,
    score_0_1,
)


CLASSIFIER_SYSTEM_PROMPT = """你是一名分析师，负责判断一条媒体内容是否属于目标人物参与的、值得收录的内容。
可收录内容不限于传统访谈，也包括播客对话、炉边谈话、主题演讲、公开演说、圆桌讨论和财报电话会议。

仅返回符合以下格式的 JSON:
{
  "is_qualifying_content": boolean,
  "target_person_present": boolean,
  "content_type": "interview|podcast_interview|fireside_chat|keynote|panel|earnings_call|speech|news_clip|commentary|unknown",
  "is_short_clip_or_commentary": boolean,
  "confidence": number,
  "reason": string
}

分类准则：
- 目标人物大概率参与，且内容属于访谈、播客对话、炉边谈话、主题演讲、公开演说、圆桌讨论或财报电话会议时，将 is_qualifying_content 设为 true。
- 不予收录：新闻综述、人物评论、反应类内容、短视频、高光片段、以解说为主的二次加工内容。
- 视频是否由官方账号发布、是否为转载或翻译版本，不影响 is_qualifying_content 的判断。
- 若佐证信息不足，请调低置信度，并说明不确定性的原因。
- 除非标题、简介、来源或元数据能够提供支撑，否则不得判定目标人物参与了该内容。
"""


class InterviewClassificationOutput(DomainModel):
    """LLM 对候选内容做可收录性识别时的结构化输出。"""

    is_qualifying_content: bool
    target_person_present: bool
    content_type: InterviewType = InterviewType.UNKNOWN
    is_short_clip_or_commentary: bool = False
    confidence: float = 0.0
    reason: str = ""

    @field_validator("confidence")
    def validate_confidence(cls, value: float) -> float:
        return score_0_1(value, field_name="classification.confidence")


class InterviewClassifier(ABC):
    """候选内容可收录性识别接口。"""

    @abstractmethod
    async def classify_async(self, candidate: CandidateItem) -> InterviewDecision:
        """识别一条候选内容是否属于可收录内容。"""


class LLMInterviewClassifier(InterviewClassifier):
    """使用 myHarness BaseLLM 的结构化输出能力判断候选内容是否可收录。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 50000,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.llm = ensure_traced_llm(llm)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    async def classify_async(self, candidate: CandidateItem) -> InterviewDecision:
        request = ChatRequest.create(
            messages=(
                Message.system(CLASSIFIER_SYSTEM_PROMPT),
                Message.user(build_candidate_prompt(candidate)),
            ),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            metadata={
                "task": "insightcast_interview_classification",
                "candidate_id": candidate.id,
                "person_ids": list(candidate.detected_person_ids),
                "source_id": candidate.source_id,
            },
        )
        output = await self.llm.structured(request, InterviewClassificationOutput)
        return output_to_decision(candidate, output, llm=self.llm)


def build_candidate_prompt(candidate: CandidateItem) -> str:
    """为单条候选内容构造紧凑的 prompt 输入。"""

    payload = {
        "candidate_id": candidate.id,
        "title": candidate.title,
        "description": candidate.description,
        "url": str(candidate.url),
        "source_name": candidate.source_name,
        "source_type": candidate.source_type.value,
        "format": candidate.format.value,
        "published_at": candidate.published_at.isoformat()
        if candidate.published_at
        else None,
        "duration_seconds": candidate.duration_seconds,
        "detected_person_ids": list(candidate.detected_person_ids),
        "detected_person_names": list(candidate.detected_person_names),
        "query_text": candidate.query_text,
        "raw_metadata": compact_metadata(candidate.raw_metadata),
    }
    return (
        "请判断这条采集到的候选内容是否属于目标人物参与的可收录内容，"
        "不限于传统访谈，也包括演讲、公开演说、圆桌讨论等形式。"
        "请结合 detected_person_names 和 query_text 中的目标人物信息判断。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def compact_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """保留对 prompt 有用的少量元数据，避免输入过长。"""

    allowed_keys = {
        "channelTitle",
        "owner",
        "author",
        "upic",
        "tag",
        "typename",
        "play",
        "favorites",
        "review",
    }
    compact = {}
    for key, value in metadata.items():
        if key in allowed_keys:
            compact[key] = value
    return compact


def output_to_decision(
    candidate: CandidateItem,
    output: InterviewClassificationOutput,
    *,
    llm: BaseLLM,
) -> InterviewDecision:
    """把 LLM 结构化输出转换为领域层的候选判断模型。"""

    return InterviewDecision(
        candidate_id=candidate.id,
        is_qualifying_content=output.is_qualifying_content,
        target_person_present=output.target_person_present,
        content_type=output.content_type,
        is_short_clip_or_commentary=output.is_short_clip_or_commentary,
        confidence=output.confidence,
        reason=output.reason,
        metadata={
            "classifier": "llm",
            "provider": llm.provider_name,
            "model": llm.model_name,
        },
    )


def create_classifier_llm_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "INSIGHTCAST_CLASSIFIER_LLM",
) -> BaseLLM:
    """根据环境变量创建用于访谈识别的 myHarness LLM。"""

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
        timeout_seconds=float(_env_get(source, f"{prefix}_TIMEOUT_SECONDS", "30")),
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
    "CLASSIFIER_SYSTEM_PROMPT",
    "InterviewClassificationOutput",
    "InterviewClassifier",
    "LLMInterviewClassifier",
    "build_candidate_prompt",
    "compact_metadata",
    "create_classifier_llm_from_env",
    "output_to_decision",
]
