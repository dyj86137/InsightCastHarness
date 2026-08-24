from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from llm import FakeLLM
from memory import (
    MemoryInjection,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
)

from insightcast.agent import LLMInterviewSummarizer, SummaryRunner
from insightcast.agent.interview_summarizer import validate_summary_length
from insightcast.domain.enums import (
    ContentFormat,
    InterviewStatus,
    InterviewType,
    TranscriptSource,
    TranscriptStatus,
)
from insightcast.domain.models import Interview, Transcript
from insightcast.storage.repositories import InsightCastRepositories


class InterviewSummarizerTests(unittest.TestCase):
    def test_summary_length_validation_uses_transcript_completeness(self) -> None:
        ready_summary = "中" * 300
        partial_summary = "中" * 150

        self.assertEqual(
            validate_summary_length(ready_summary, TranscriptStatus.READY),
            ready_summary,
        )
        self.assertEqual(
            validate_summary_length(partial_summary, TranscriptStatus.PARTIAL),
            partial_summary,
        )
        self.assertEqual(
            len(validate_summary_length("中" * 600, TranscriptStatus.READY)),
            600,
        )
        self.assertEqual(
            len(validate_summary_length("中" * 300, TranscriptStatus.PARTIAL)),
            300,
        )
        with self.assertRaises(ValueError):
            validate_summary_length("中" * 299, TranscriptStatus.READY)
        with self.assertRaises(ValueError):
            validate_summary_length("中" * 149, TranscriptStatus.PARTIAL)
        with self.assertRaises(ValueError):
            validate_summary_length("中" * 601, TranscriptStatus.READY)

    def test_llm_summarizer_returns_interview_summary(self) -> None:
        llm = FakeLLM(responses=[summary_response("AI 基础设施需求继续增长。")])
        summarizer = LLMInterviewSummarizer(llm)
        interview = make_interview()
        transcript = make_transcript(interview_id=interview.id)

        generated = asyncio.run(summarizer.summarize_async(interview, transcript))
        summary = generated.summary

        self.assertEqual(summary.interview_id, interview.id)
        self.assertIn("AI 基础设施", summary.summary)
        self.assertEqual(summary.key_points, ("GPU 需求仍然强劲",))
        self.assertEqual(summary.potential_opportunities, ("NVIDIA 继续强调数据中心扩张",))
        self.assertEqual(summary.mentioned_companies, ("NVIDIA",))
        self.assertEqual(summary.insights[0].statement, "企业 AI 推理需求正在扩大")
        self.assertEqual(len(generated.evidence_claims), 2)
        self.assertEqual(generated.evidence_claims[0].field, "summary")
        self.assertEqual(generated.evidence_claims[0].evidence[0].text, "Description: NVIDIA CEO discusses AI infrastructure demand.")
        self.assertEqual(summary.metadata["provider"], "fake")
        self.assertEqual(llm.last_request.max_tokens, 2000)
        self.assertIn('"status": "partial"', llm.last_request.messages[1].content)

    def test_llm_summarizer_injects_person_history_memory(self) -> None:
        llm = FakeLLM(responses=[summary_response("本次更强调企业推理。")])
        summarizer = LLMInterviewSummarizer(
            llm,
            memory_provider=StaticMemoryProvider(),
        )
        interview = make_interview()
        transcript = make_transcript(interview_id=interview.id)

        summary = asyncio.run(summarizer.summarize_async(interview, transcript)).summary

        prompt = llm.last_request.messages[1].content
        self.assertIn('"memory"', prompt)
        self.assertIn("Historical person memory", prompt)
        self.assertIn("此前主要强调训练和数据中心供给", prompt)
        self.assertEqual(summary.metadata["memory"]["injected_count"], 1)
        self.assertEqual(summary.metadata["memory"]["memory_ids"], ["mem_history_1"])

    def test_summary_runner_creates_summary_and_updates_interview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            interview = make_interview(id="interview_create")
            transcript = make_transcript(interview_id=interview.id)
            repositories.interviews.create(interview)
            repositories.transcripts.create(transcript)
            llm = FakeLLM(responses=[summary_response("这是一场关于 AI 基础设施的访谈。")])

            result = SummaryRunner(
                repositories,
                LLMInterviewSummarizer(llm),
            ).run_once()

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.created_count, 1)
            self.assertEqual(result.updated_count, 0)
            self.assertEqual(repositories.interview_summaries.count(), 1)
            evidence_path = Path(result.evidence_path)
            self.assertTrue(evidence_path.exists())
            evidence_record = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(evidence_record["interview_id"], interview.id)
            self.assertEqual(evidence_record["claims"][0]["field"], "summary")
            self.assertIsNotNone(evidence_record["claims"][0]["evidence"][0]["start_char"])
            updated_interview = repositories.interviews.require(interview.id)
            self.assertEqual(updated_interview.status, InterviewStatus.SUMMARY_READY)
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.metadata["stage"], "summary")

    def test_summary_runner_updates_existing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            interview = make_interview(id="interview_update")
            transcript = make_transcript(interview_id=interview.id)
            repositories.interviews.create(interview)
            repositories.transcripts.create(transcript)
            llm = FakeLLM(
                responses=[
                    summary_response("第一次摘要。"),
                    summary_response("第二次摘要。"),
                ]
            )
            runner = SummaryRunner(repositories, LLMInterviewSummarizer(llm))

            first_result = runner.run_once()
            first_summary = repositories.interview_summaries.find_by_interview_id(interview.id)
            repositories.interviews.update(interview.id, status=InterviewStatus.TRANSCRIPT_READY)
            second_result = runner.run_once()
            updated_summary = repositories.interview_summaries.find_by_interview_id(interview.id)

            self.assertEqual(first_result.created_count, 1)
            self.assertEqual(second_result.updated_count, 1)
            self.assertEqual(updated_summary.id, first_summary.id)
        self.assertIn("第二次摘要", updated_summary.summary)

    def test_summary_runner_marks_missing_transcript_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repositories = InsightCastRepositories(root_dir=Path(tmp) / "storage")
            interview = make_interview(id="interview_missing_transcript")
            repositories.interviews.create(interview)
            llm = FakeLLM(responses=[summary_response("不会被使用。")])

            result = SummaryRunner(
                repositories,
                LLMInterviewSummarizer(llm),
            ).run_once()

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(
                repositories.interviews.require(interview.id).status,
                InterviewStatus.FAILED,
            )
            run = repositories.run_records.require(result.run_id)
            self.assertEqual(run.error, "1 summary errors")


def make_interview(id: str = "interview_test") -> Interview:
    return Interview(
        id=id,
        title="Jensen Huang interview on AI infrastructure",
        url=f"https://www.youtube.com/watch?v={id}",
        type=InterviewType.INTERVIEW,
        format=ContentFormat.VIDEO,
        status=InterviewStatus.TRANSCRIPT_READY,
        person_names=("Jensen Huang",),
        transcript_status=TranscriptStatus.PARTIAL,
    )


def make_transcript(interview_id: str) -> Transcript:
    return Transcript(
        id=f"transcript_{interview_id}",
        interview_id=interview_id,
        status=TranscriptStatus.PARTIAL,
        source=TranscriptSource.SOURCE_METADATA,
        text="Interview title: Jensen Huang interview\n\nDescription: NVIDIA CEO discusses AI infrastructure demand.",
    )


def summary_response(summary: str) -> str:
    return f"""
    {{
      "summary": "{summary} 数据中心需求仍然是核心主题。Jensen Huang 强调企业 AI 推理正在扩大。NVIDIA 的战略重点仍是计算基础设施。该访谈也显示企业部署需要同时关注算力成本、供应链弹性和应用落地节奏。对于关注 AI 基础设施的人而言，这些信息可以帮助判断需求变化和服务机会。当前 transcript 只包含标题和简介，证据不足，不能确认访谈中是否有更多具体承诺、产品信息或时间表。因此上述内容仅用于概括可见材料，不能替代完整逐字稿，也不应被解读为已经发生的商业合作。",
      "key_points": ["GPU 需求仍然强劲"],
      "potential_opportunities": ["NVIDIA 继续强调数据中心扩张"],
      "industry_judgements": ["AI 基础设施仍处于高景气阶段"],
      "mentioned_companies": ["NVIDIA"],
      "mentioned_products": ["GPU"],
      "evidence_claims": [
        {{
          "field": "summary",
          "claim": "{summary}",
          "model_confidence": 0.82,
          "evidence": [{{"text": "Description: NVIDIA CEO discusses AI infrastructure demand.", "segment_id": null}}]
        }},
        {{
          "field": "potential_opportunities",
          "claim": "NVIDIA 继续强调数据中心扩张",
          "model_confidence": 0.76,
          "evidence": [{{"text": "Description: NVIDIA CEO discusses AI infrastructure demand.", "segment_id": null}}]
        }}
      ],
      "audience": ["AI 投资研究员", "企业软件从业者"],
      "novelty_assessment": "暂无历史观点可比，仅基于本次材料判断。",
      "insights": [
        {{
          "statement": "企业 AI 推理需求正在扩大",
          "category": "demand",
          "confidence": 0.82,
          "evidence": "访谈材料强调数据中心需求。",
          "mentioned_companies": ["NVIDIA"],
          "mentioned_products": ["GPU"]
        }}
      ]
    }}
    """


class StaticMemoryProvider:
    async def build_summary_memory(self, interview: Interview) -> MemoryInjection:
        record = MemoryRecord(
            id="mem_history_1",
            kind=MemoryKind.LONG_TERM,
            content="Jensen Huang 此前主要强调训练和数据中心供给。",
            summary="此前主要强调训练和数据中心供给。",
            scope=MemoryScope.GLOBAL,
            tags=("person:jensen_huang",),
            metadata={"interview_id": "interview_old"},
        )
        result = MemorySearchResult(
            record=record,
            score=0.9,
            reason="tag_match",
            matched_tags=("person:jensen_huang",),
        )
        return MemoryInjection(
            query=MemoryQuery(text=interview.title, tags=("person:jensen_huang",)),
            results=(result,),
            content=(
                "Historical person memory:\n"
                "1. [long_term score=0.90 tags=person:jensen_huang] "
                "此前主要强调训练和数据中心供给。"
            ),
            estimated_tokens=20,
            metadata={
                "provider": "test_person_history",
                "result_count": 1,
                "injected_count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
