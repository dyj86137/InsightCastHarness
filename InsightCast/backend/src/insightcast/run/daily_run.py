"""从终端或调度器运行完整的 InsightCast V1 日常 pipeline。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence


def _bootstrap_paths() -> None:
    """在项目打包前支持直接执行脚本。"""

    current = Path(__file__).resolve()
    backend_src = current.parents[2]
    workspace_root = current.parents[6]
    harness_src = workspace_root / "myHarness" / "myHarness-V2" / "src"
    for path in (backend_src, harness_src):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_paths()

from runtime.hooks import HookManager  # noqa: E402

from insightcast.agent import (  # noqa: E402
    BriefWritingAgent,
    DailyPipelineRunResult,
    DailyPipelineRunner,
    DiscoveryResearchAgent,
    InterviewClassifier,
    InterviewSummarizer,
    LLMInterviewClassifier,
    LLMInterviewSummarizer,
    create_classifier_llm_from_env,
    create_discovery_llm_from_env,
    create_summary_llm_from_env,
)
from insightcast.briefs import (  # noqa: E402
    DirectMarkdownBriefGenerator,
    MarkdownBriefGenerator,
    create_brief_llm_from_env,
)
from insightcast.config import apply_config, load_config, load_env_file  # noqa: E402
from insightcast.context import create_summary_context_provider_from_env  # noqa: E402
from insightcast.memory import PersonHistoryMemoryProvider  # noqa: E402
from insightcast.sources.discovery import JsonFetcher  # noqa: E402
from insightcast.storage.repositories import InsightCastRepositories  # noqa: E402


def run_daily_pipeline(
    *,
    config_path: Path,
    storage_dir: Optional[Path] = None,
    storage_backend: str = "json",
    sqlite_path: Optional[Path] = None,
    brief_output_dir: Optional[Path] = None,
    trace_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    error_log_dir: Optional[Path] = None,
    trace_enabled: bool = True,
    checkpoint_enabled: bool = True,
    resume_run_id: Optional[str] = None,
    run_id: Optional[str] = None,
    tools_enabled: bool = True,
    memory_enabled: bool = True,
    context_enabled: bool = True,
    discovery_agent_enabled: bool = True,
    brief_agent_enabled: bool = False,
    discovery_fetch_timeout_seconds: Optional[float] = None,
    transcript_fetch_limit: Optional[int] = None,
    fail_fast: bool = False,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[JsonFetcher] = None,
    classifier: Optional[InterviewClassifier] = None,
    summarizer: Optional[InterviewSummarizer] = None,
    brief_generator: Optional[MarkdownBriefGenerator] = None,
    discovery_agent: Optional[DiscoveryResearchAgent] = None,
    brief_agent: Optional[BriefWritingAgent] = None,
    hook_manager: Optional[HookManager] = None,
) -> DailyPipelineRunResult:
    """负责初始化全局外部依赖（包括外部的配置文件、仓储数据库、环境变量），并执行一次完整日常 pipeline。"""

    if env is None:
        load_env_file()
        env = os.environ

    config = load_config(config_path)
    repositories = InsightCastRepositories(
        root_dir=storage_dir,
        backend=storage_backend,
        sqlite_path=sqlite_path,
    )
    apply_config(config, repositories)

    classifier = classifier or LLMInterviewClassifier(
        create_classifier_llm_from_env(env)
    )
    memory_provider = (
        PersonHistoryMemoryProvider(repositories)
        if memory_enabled
        else None
    )
    summarizer = summarizer or LLMInterviewSummarizer(
        create_summary_llm_from_env(env),
        memory_provider=memory_provider,
        context_provider=(
            create_summary_context_provider_from_env(env)
            if context_enabled
            else None
        ),
        context_enabled=context_enabled,
    )
    brief_generator = brief_generator or DirectMarkdownBriefGenerator()
    if brief_agent is None and brief_agent_enabled:
        brief_agent = BriefWritingAgent(create_brief_llm_from_env(env), env=env)
    if discovery_agent is None and discovery_agent_enabled:
        discovery_agent = DiscoveryResearchAgent(
            create_discovery_llm_from_env(env),
            env=env,
            fetch_json=fetch_json,
        )

    return DailyPipelineRunner(
        repositories,
        classifier=classifier,
        summarizer=summarizer,
        brief_generator=brief_generator,
        env=env,
        fetch_json=fetch_json,
        fail_fast=fail_fast,
        discovery_agent=discovery_agent,
        brief_output_dir=brief_output_dir,
        brief_agent=brief_agent,
        trace_dir=trace_dir,
        checkpoint_dir=checkpoint_dir,
        error_log_dir=error_log_dir,
        hook_manager=hook_manager,
        trace_enabled=trace_enabled,
        checkpoint_enabled=checkpoint_enabled,
        resume_run_id=resume_run_id,
        run_id=run_id,
        tools_enabled=tools_enabled,
        discovery_fetch_timeout_seconds=discovery_fetch_timeout_seconds,
        transcript_fetch_limit=transcript_fetch_limit,
    ).run_once()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="执行一次完整的 InsightCast V1 日常 pipeline。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="完整配置文件或拆分配置目录的路径。",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="本地 JSON 仓储文件目录，或 SQLite 数据库目录。",
    )
    parser.add_argument(
        "--storage-backend",
        choices=("json", "sqlite"),
        default="sqlite",
        help="要使用的本地存储后端。",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="SQLite 数据库路径。默认使用 backend/data/insightcast.sqlite3。",
    )
    parser.add_argument(
        "--brief-output-dir",
        type=Path,
        default=None,
        help="生成的 Markdown 日报输出目录。",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="myHarness JSONL trace 文件目录。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="myHarness checkpoint 文件目录。",
    )
    parser.add_argument(
        "--error-log-dir",
        type=Path,
        default=None,
        help="每次运行的 JSONL 错误日志目录。",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="从某次历史 pipeline 运行的最新 checkpoint 恢复。",
    )
    parser.add_argument(
        "--discovery-fetch-timeout-seconds",
        type=float,
        default=None,
        help="Discovery 阶段单次 JSON 网络请求的超时秒数。",
    )
    parser.add_argument(
        "--transcript-fetch-limit",
        type=int,
        default=None,
        help="Transcript 获取阶段最多处理的 interview 数量。默认不限制。",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="关闭 myHarness JSONL trace 输出。",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="关闭 myHarness checkpoint 输出。",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="关闭 myHarness ToolExecutor 适配器。",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="关闭 myHarness Memory 适配器。",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="关闭 myHarness Context 适配器。",
    )
    parser.add_argument(
        "--no-discovery-agent",
        action="store_true",
        help="关闭默认 myHarness ReactLoop DiscoveryResearchAgent。",
    )
    parser.add_argument(
        "--no-brief-agent",
        action="store_true",
        help="关闭默认 myHarness ReactLoop BriefWritingAgent。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="在支持的阶段中遇到首个条目错误时立即停止。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="打印机器可读的 JSON 输出。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_daily_pipeline(
            config_path=args.config,
            storage_dir=args.storage_dir,
            storage_backend=args.storage_backend,
            sqlite_path=args.sqlite_path,
            brief_output_dir=args.brief_output_dir,
            trace_dir=args.trace_dir,
            checkpoint_dir=args.checkpoint_dir,
            error_log_dir=args.error_log_dir,
            trace_enabled=not args.no_trace,
            checkpoint_enabled=not args.no_checkpoint,
            resume_run_id=args.resume_run_id,
            tools_enabled=not args.no_tools,
            memory_enabled=not args.no_memory,
            context_enabled=not args.no_context,
            discovery_agent_enabled=not args.no_discovery_agent,
            brief_agent_enabled=not args.no_brief_agent,
            discovery_fetch_timeout_seconds=args.discovery_fetch_timeout_seconds,
            transcript_fetch_limit=args.transcript_fetch_limit,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Daily pipeline failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(result))

    if result.status == "failed":
        return 1
    if result.status == "partial":
        return 2
    return 0


def format_result(result: DailyPipelineRunResult) -> str:
    lines = [
        f"Daily pipeline {result.status}: {result.run_id}",
        f"discovered: {result.discovered_count}",
        f"accepted_interviews: {result.accepted_count}",
        f"pushed: {result.pushed_count}",
    ]
    for stage in result.stages:
        lines.append(f"{stage.name}: {stage.status} ({stage.run_id or 'no-run'})")
    if result.markdown_path:
        lines.append(f"markdown: {result.markdown_path}")
    if result.error_log_path:
        lines.append(f"error_log: {result.error_log_path}")
    return "\n".join(lines)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "local.example.json"


if __name__ == "__main__":
    raise SystemExit(main())
