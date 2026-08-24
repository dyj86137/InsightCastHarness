"""InsightCast transcript IO 的 myHarness 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from core.tool import ToolCall, ToolResult
from pydantic import BaseModel, Field
from tools.base import BaseTool, ToolExecutionContext

from insightcast.domain.models import CandidateItem, Interview
from insightcast.tools.harness import (
    TracingToolExecutor,
    current_tool_execution_context,
    run_tool_sync,
)
from insightcast.transcript import (
    AudioDownloadResult,
    AudioTranscriptionResult,
    DEFAULT_PREFERRED_CAPTION_LANGUAGES,
    PlatformCaptionFetchResult,
    download_candidate_audio,
    fetch_platform_caption,
)


FETCH_PLATFORM_CAPTION_TOOL_NAME = "insightcast_fetch_platform_caption"
DOWNLOAD_AUDIO_TOOL_NAME = "insightcast_download_audio"
LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME = "insightcast_lemonade_transcribe_audio"
DEFAULT_LEMONADE_MCP_TOOL_NAME = "lemonade.lemonade_transcribe_audio"
DEFAULT_LEMONADE_TRANSCRIBE_MODEL = "Whisper-Large-v3-Turbo"


class FetchPlatformCaptionInput(BaseModel):
    """抓取平台字幕的参数。"""

    candidate: Dict[str, Any]
    preferred_languages: Tuple[str, ...] = Field(default_factory=tuple)
    timeout_seconds: Optional[float] = None


class DownloadAudioInput(BaseModel):
    """下载候选内容音频的参数。"""

    candidate: Dict[str, Any]
    output_dir: str
    timeout_seconds: Optional[float] = None


class LemonadeTranscribeAudioInput(BaseModel):
    """通过 Lemonade 转写音频文件的参数。"""

    audio_path: str
    interview_id: Optional[str] = None
    candidate_id: Optional[str] = None
    source_type: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None


class FetchPlatformCaptionTool(BaseTool):
    """抓取 YouTube/Bilibili 已有字幕。"""

    name = FETCH_PLATFORM_CAPTION_TOOL_NAME
    description = "为 InsightCast 抓取 YouTube/Bilibili 已有字幕。"
    input_model = FetchPlatformCaptionInput
    is_read_only = True
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "transcript", "caption")
    metadata = {"component": "transcript"}

    def __init__(self, *, env: Optional[Mapping[str, str]] = None) -> None:
        self.env = env

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = coerce_platform_caption_input(arguments)
        result = fetch_platform_caption(
            CandidateItem(**args.candidate),
            preferred_languages=args.preferred_languages
            or DEFAULT_PREFERRED_CAPTION_LANGUAGES,
            timeout_seconds=args.timeout_seconds,
            env=self.env,
        )
        if result is None:
            return ToolResult.success(
                tool_call=tool_call,
                output="未找到平台字幕。",
                data={"found": False},
                metadata={"found": False},
            )
        data = result.to_dict()
        data["found"] = True
        return ToolResult.success(
            tool_call=tool_call,
            output="已抓取平台字幕。",
            data=data,
            metadata={
                "found": True,
                "candidate_id": result.candidate_id,
                "source": result.source.value,
                "language": result.language,
                "text_chars": len(result.text),
            },
        )


class DownloadAudioTool(BaseTool):
    """下载 YouTube/Bilibili 候选内容音频。"""

    name = DOWNLOAD_AUDIO_TOOL_NAME
    description = "下载 YouTube/Bilibili 音频，用于 transcript 兜底转写。"
    input_model = DownloadAudioInput
    is_read_only = False
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "transcript", "audio")
    metadata = {"component": "transcript"}

    def __init__(self, *, env: Optional[Mapping[str, str]] = None) -> None:
        self.env = env

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = coerce_download_audio_input(arguments)
        result = download_candidate_audio(
            CandidateItem(**args.candidate),
            output_dir=Path(args.output_dir),
            timeout_seconds=args.timeout_seconds,
            env=self.env,
        )
        if result is None:
            return ToolResult.success(
                tool_call=tool_call,
                output="未下载到可用音频。",
                data={"found": False},
                metadata={"found": False},
            )
        data = result.to_dict()
        data["found"] = True
        return ToolResult.success(
            tool_call=tool_call,
            output="已下载音频。",
            data=data,
            metadata={
                "found": True,
                "candidate_id": result.candidate_id,
                "audio_path": result.audio_path,
            },
        )


class LemonadeTranscribeAudioTool(BaseTool):
    """把 Lemonade MCP 转写能力包装成 InsightCast transcript 工具。"""

    name = LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME
    description = "通过 Lemonade MCP 转写音频，并返回规范化文本。"
    input_model = LemonadeTranscribeAudioInput
    is_read_only = True
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "transcript", "audio", "lemonade")
    metadata = {"component": "transcript", "provider": "lemonade"}

    def __init__(
        self,
        executor: TracingToolExecutor,
        *,
        mcp_tool_name: str = DEFAULT_LEMONADE_MCP_TOOL_NAME,
        model: str = DEFAULT_LEMONADE_TRANSCRIBE_MODEL,
        response_format: str = "verbose_json",
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executor = executor
        self.mcp_tool_name = mcp_tool_name
        self.model = model
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = coerce_lemonade_transcribe_input(arguments)
        mcp_call = ToolCall.create(
            name=self.mcp_tool_name,
            arguments={
                "model": self.model,
                "audio_path": args.audio_path,
                "response_format": self.response_format,
            },
            metadata={
                "adapter": "lemonade_audio_transcriber",
                "interview_id": args.interview_id,
                "candidate_id": args.candidate_id,
            },
        )
        result = await self.executor.execute(mcp_call, context)
        if result.is_error:
            return result.clone(tool_call_id=tool_call.id, tool_name=tool_call.name)

        transcription = audio_transcription_result_from_tool_result(result)
        if transcription is None:
            return ToolResult.failure(
                tool_call=tool_call,
                error=RuntimeError("Lemonade 转写结果中没有可用文本。"),
                metadata={
                    "provider": "lemonade",
                    "mcp_tool_name": self.mcp_tool_name,
                },
            )
        return ToolResult.success(
            tool_call=tool_call,
            output=transcription.text,
            data={
                "text": transcription.text,
                "language": transcription.language,
                "segments": list(transcription.segments),
                "provider": "lemonade",
                "model": self.model,
                "response_format": self.response_format,
                "mcp_tool_name": self.mcp_tool_name,
            },
            metadata={
                "provider": "lemonade",
                "model": self.model,
                "mcp_tool_name": self.mcp_tool_name,
                "text_chars": len(transcription.text),
            },
        )


class ToolPlatformCaptionFetcher:
    """由 myHarness ToolExecutor 支撑的同步平台字幕抓取适配器。"""

    def __init__(
        self,
        executor: TracingToolExecutor,
        *,
        preferred_languages: Tuple[str, ...] = (),
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executor = executor
        self.preferred_languages = preferred_languages
        self.timeout_seconds = timeout_seconds

    def __call__(self, candidate: CandidateItem) -> Optional[PlatformCaptionFetchResult]:
        tool_call = ToolCall.create(
            name=FETCH_PLATFORM_CAPTION_TOOL_NAME,
            arguments={
                "candidate": candidate.to_dict(),
                "preferred_languages": self.preferred_languages,
                "timeout_seconds": self.timeout_seconds,
            },
            metadata={"adapter": "platform_caption_fetcher"},
        )
        context = current_tool_execution_context(
            allow_mutation=False,
            timeout_seconds=self.timeout_seconds,
            metadata={"tool_kind": "platform_caption_fetcher"},
        )
        result = run_tool_sync(self.executor.execute(tool_call, context))
        if result.is_error:
            raise RuntimeError(tool_error_message(result))
        data = result.data
        if not isinstance(data, Mapping) or not data.get("found"):
            return None
        payload = dict(data)
        payload.pop("found", None)
        return PlatformCaptionFetchResult(**payload)


class ToolAudioDownloader:
    """由 myHarness ToolExecutor 支撑的同步音频下载适配器。"""

    def __init__(
        self,
        executor: TracingToolExecutor,
        *,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executor = executor
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds

    def __call__(self, candidate: CandidateItem) -> Optional[AudioDownloadResult]:
        tool_call = ToolCall.create(
            name=DOWNLOAD_AUDIO_TOOL_NAME,
            arguments={
                "candidate": candidate.to_dict(),
                "output_dir": str(self.output_dir),
                "timeout_seconds": self.timeout_seconds,
            },
            metadata={"adapter": "audio_downloader"},
        )
        context = current_tool_execution_context(
            allow_mutation=True,
            timeout_seconds=self.timeout_seconds,
            metadata={"tool_kind": "audio_downloader"},
        )
        result = run_tool_sync(self.executor.execute(tool_call, context))
        if result.is_error:
            raise RuntimeError(tool_error_message(result))
        data = result.data
        if not isinstance(data, Mapping) or not data.get("found"):
            return None
        payload = dict(data)
        payload.pop("found", None)
        return AudioDownloadResult(**payload)


class ToolAudioTranscriber:
    """由 myHarness ToolExecutor 支撑的同步外部转写适配器。"""

    def __init__(
        self,
        executor: TracingToolExecutor,
        *,
        tool_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executor = executor
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        download: AudioDownloadResult,
        interview: Interview,
        candidate: CandidateItem,
    ) -> Optional[AudioTranscriptionResult]:
        tool_call = ToolCall.create(
            name=self.tool_name,
            arguments={
                "audio_path": download.audio_path,
                "interview_id": interview.id,
                "candidate_id": candidate.id,
                "source_type": candidate.source_type.value,
                "url": str(candidate.url),
                "title": candidate.title,
            },
            metadata={"adapter": "audio_transcriber"},
        )
        context = current_tool_execution_context(
            allow_mutation=False,
            timeout_seconds=self.timeout_seconds,
            metadata={"tool_kind": "audio_transcriber"},
        )
        result = run_tool_sync(self.executor.execute(tool_call, context))
        if result.is_error:
            raise RuntimeError(tool_error_message(result))
        return audio_transcription_result_from_tool_result(result)


def audio_transcription_result_from_tool_result(
    result: ToolResult,
) -> Optional[AudioTranscriptionResult]:
    data = result.data
    if isinstance(data, Mapping):
        text = (
            data.get("text")
            or data.get("transcript")
            or text_from_verbose_json(data)
            or result.output
        )
        if not isinstance(text, str) or not text.strip():
            return None
        segments = data.get("segments") or ()
        if isinstance(segments, str):
            segments = (segments,)
        return AudioTranscriptionResult(
            text=text,
            language=data.get("language") if isinstance(data.get("language"), str) else None,
            segments=tuple(str(segment) for segment in segments),
            metadata={
                "transcriber_tool": result.tool_name,
                "tool_metadata": dict(result.metadata),
            },
        )
    if result.output and result.output.strip():
        return AudioTranscriptionResult(
            text=result.output,
            metadata={"transcriber_tool": result.tool_name},
        )
    return None


def text_from_verbose_json(data: Mapping[str, Any]) -> Optional[str]:
    content = data.get("content")
    if isinstance(content, Mapping):
        value = content.get("text") or content.get("transcript")
        if isinstance(value, str):
            return value
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                value = item.get("text") or item.get("transcript")
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        if parts:
            return "\n".join(parts)
    return None


def coerce_platform_caption_input(arguments: BaseModel) -> FetchPlatformCaptionInput:
    if isinstance(arguments, FetchPlatformCaptionInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return FetchPlatformCaptionInput(**arguments.model_dump())
    return FetchPlatformCaptionInput(**arguments.dict())


def coerce_download_audio_input(arguments: BaseModel) -> DownloadAudioInput:
    if isinstance(arguments, DownloadAudioInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return DownloadAudioInput(**arguments.model_dump())
    return DownloadAudioInput(**arguments.dict())


def coerce_lemonade_transcribe_input(arguments: BaseModel) -> LemonadeTranscribeAudioInput:
    if isinstance(arguments, LemonadeTranscribeAudioInput):
        return arguments
    if hasattr(arguments, "model_dump"):
        return LemonadeTranscribeAudioInput(**arguments.model_dump())
    return LemonadeTranscribeAudioInput(**arguments.dict())


def tool_error_message(result: ToolResult) -> str:
    error = result.error or {}
    return str(error.get("message") or result.output or "工具调用失败")


__all__ = [
    "DEFAULT_LEMONADE_MCP_TOOL_NAME",
    "DEFAULT_LEMONADE_TRANSCRIBE_MODEL",
    "DOWNLOAD_AUDIO_TOOL_NAME",
    "FETCH_PLATFORM_CAPTION_TOOL_NAME",
    "LEMONADE_TRANSCRIBE_AUDIO_TOOL_NAME",
    "DownloadAudioInput",
    "DownloadAudioTool",
    "FetchPlatformCaptionInput",
    "FetchPlatformCaptionTool",
    "LemonadeTranscribeAudioInput",
    "LemonadeTranscribeAudioTool",
    "ToolAudioDownloader",
    "ToolAudioTranscriber",
    "ToolPlatformCaptionFetcher",
]
