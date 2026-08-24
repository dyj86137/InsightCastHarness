"""访谈摘要证据链的结构和本地 JSONL 审计文件写入。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pydantic import Field, field_validator

from insightcast.domain.models import DomainModel, Interview, Transcript, non_empty_string


class EvidenceSpan(DomainModel):
    """模型为一条声明提供的原文片段。"""

    text: str
    segment_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @field_validator("text")
    def validate_text(cls, value: str) -> str:
        return non_empty_string(value, field_name="evidence.text")


class EvidenceClaim(DomainModel):
    """摘要或关注机会中的一条可核查原子声明。"""

    field: str
    claim: str
    evidence: Tuple[EvidenceSpan, ...] = Field(default_factory=tuple)
    model_confidence: Optional[float] = None

    @field_validator("field")
    def validate_field(cls, value: str) -> str:
        field = non_empty_string(value, field_name="evidence.field")
        if field not in {"summary", "potential_opportunities"}:
            raise ValueError(
                "evidence.field must be summary or potential_opportunities"
            )
        return field

    @field_validator("claim")
    def validate_claim(cls, value: str) -> str:
        return non_empty_string(value, field_name="evidence.claim")


def default_evidence_dir(root_dir: Path) -> Path:
    """返回与当前 InsightCast 仓储对应的证据目录。"""

    return root_dir / "evidence"


def evidence_path(evidence_dir: Path, run_id: str) -> Path:
    """返回一次摘要运行的 JSONL 证据文件路径。"""

    return evidence_dir / f"summary_{run_id}.jsonl"


def initialize_evidence_file(path: Path) -> Path:
    """创建或清空一次运行的证据文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def append_interview_evidence(
    path: Path,
    *,
    run_id: str,
    interview: Interview,
    transcript: Transcript,
    claims: Sequence[EvidenceClaim],
) -> None:
    """将单条访谈的证据记录追加为一个 JSONL 对象。"""

    source_text = transcript_source_text(transcript)
    source_segments = transcript_source_segments(transcript, source_text)
    record = {
        "record_type": "interview_evidence",
        "run_id": run_id,
        "interview_id": interview.id,
        "title": interview.title,
        "url": str(interview.url),
        "transcript_id": transcript.id,
        "transcript_status": transcript.status.value,
        "transcript_source": transcript.source.value,
        "transcript_content_hash": transcript.content_hash,
        "source_char_count": len(source_text),
        "claims": [
            materialize_claim(claim, source_text=source_text, source_segments=source_segments)
            for claim in claims
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def transcript_source_text(transcript: Transcript) -> str:
    """返回用于证据定位的原始文本。"""

    if transcript.text:
        return transcript.text.strip()
    return "\n\n".join(segment.strip() for segment in transcript.segments if segment.strip())


def transcript_source_segments(
    transcript: Transcript,
    source_text: str,
) -> Tuple[Dict[str, Any], ...]:
    """为原文片段生成稳定的 segment_id 和字符范围。"""

    if transcript.segments:
        segments = []
        cursor = 0
        for index, raw_segment in enumerate(transcript.segments):
            segment = raw_segment.strip()
            if not segment:
                continue
            relative_start, relative_end = locate_text(source_text[cursor:], segment)
            if relative_start is None or relative_end is None:
                continue
            start = cursor + relative_start
            end = cursor + relative_end
            segments.append(
                {
                    "segment_id": f"transcript_segment_{index + 1:04d}",
                    "start_char": start,
                    "end_char": end,
                }
            )
            cursor = end
        if segments:
            return tuple(segments)

    return tuple(
        {
            "segment_id": f"transcript_block_{index + 1:04d}",
            "start_char": start,
            "end_char": end,
        }
        for index, (start, end) in enumerate(text_blocks(source_text))
    )


def materialize_claim(
    claim: EvidenceClaim,
    *,
    source_text: str,
    source_segments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """将模型返回的证据片段补充为可定位的原文范围。"""

    materialized_evidence = []
    for span in claim.evidence:
        materialized = materialize_span(
            span,
            source_text=source_text,
            source_segments=source_segments,
        )
        if materialized is not None:
            materialized_evidence.append(materialized)
    return {
        "field": claim.field,
        "claim": claim.claim,
        "model_confidence": claim.model_confidence,
        "evidence": materialized_evidence,
    }


def materialize_span(
    span: EvidenceSpan,
    *,
    source_text: str,
    source_segments: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """定位证据文本，并只写入 transcript 中实际存在的原文。"""

    start, end = locate_text(source_text, span.text)
    if start is None or end is None:
        return None
    segment_id = segment_for_range(start, end, source_segments)
    return {
        "text": source_text[start:end],
        "segment_id": segment_id,
        "start_char": start,
        "end_char": end,
    }


def locate_text(source_text: str, evidence_text: str) -> Tuple[Optional[int], Optional[int]]:
    """先精确匹配，再按空白归一化匹配证据片段。"""

    exact_start = source_text.find(evidence_text)
    if exact_start >= 0:
        return exact_start, exact_start + len(evidence_text)

    normalized_source, source_offsets = normalize_with_offsets(source_text)
    normalized_evidence = re.sub(r"\s+", " ", evidence_text.strip())
    if not normalized_evidence:
        return None, None
    normalized_start = normalized_source.find(normalized_evidence)
    if normalized_start < 0:
        return None, None
    start = source_offsets[normalized_start]
    end_index = normalized_start + len(normalized_evidence) - 1
    end = source_offsets[end_index] + 1
    return start, end


def normalize_with_offsets(value: str) -> Tuple[str, Tuple[int, ...]]:
    """归一化空白，同时保留归一化字符到原文的偏移。"""

    chars = []
    offsets = []
    previous_space = False
    for index, char in enumerate(value):
        if char.isspace():
            if previous_space:
                continue
            chars.append(" ")
            offsets.append(index)
            previous_space = True
            continue
        chars.append(char)
        offsets.append(index)
        previous_space = False
    return "".join(chars), tuple(offsets)


def segment_for_range(
    start: int,
    end: int,
    source_segments: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    for segment in source_segments:
        if start >= int(segment["start_char"]) and end <= int(segment["end_char"]):
            return str(segment["segment_id"])
    return None


def text_blocks(value: str) -> Tuple[Tuple[int, int], ...]:
    blocks = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", value, flags=re.DOTALL):
        blocks.append((match.start(), match.end()))
    return tuple(blocks)


__all__ = [
    "EvidenceClaim",
    "EvidenceSpan",
    "append_interview_evidence",
    "default_evidence_dir",
    "evidence_path",
    "initialize_evidence_file",
    "locate_text",
    "transcript_source_segments",
    "transcript_source_text",
]
