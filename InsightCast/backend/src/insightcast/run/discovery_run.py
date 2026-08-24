"""从终端或调度任务运行一次 InsightCast 发现流程。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


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

from insightcast.agent import DiscoveryRunResult, DiscoveryRunner
from insightcast.config import apply_config, load_config, load_env_file
from insightcast.sources.discovery import JsonFetcher
from insightcast.storage.repositories import InsightCastRepositories


def run_discovery(
    *,
    config_path: Path,
    storage_dir: Optional[Path] = None,
    storage_backend: str = "json",
    sqlite_path: Optional[Path] = None,
    fail_fast: bool = False,
    env: Optional[Mapping[str, str]] = None,
    fetch_json: Optional[JsonFetcher] = None,
) -> DiscoveryRunResult:
    """加载配置、初始化仓储，并执行一次发现流程。"""

    if env is None:
        load_env_file()

    config = load_config(config_path)
    repositories = InsightCastRepositories(
        root_dir=storage_dir,
        backend=storage_backend,
        sqlite_path=sqlite_path,
    )
    apply_config(config, repositories)
    runner = DiscoveryRunner(
        repositories,
        env=env,
        fetch_json=fetch_json,
        fail_fast=fail_fast,
    )
    return runner.run_once()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="执行一次 InsightCast 发现流程。",
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
        help="本地 JSON 仓储文件目录。",
    )
    parser.add_argument(
        "--storage-backend",
        choices=("json", "sqlite"),
        default="json",
        help="要使用的本地存储后端。",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="SQLite 数据库路径。默认使用 backend/data/insightcast.sqlite3。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到首个发现错误时立即停止，而不是继续执行。",
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
        result = run_discovery(
            config_path=args.config,
            storage_dir=args.storage_dir,
            storage_backend=args.storage_backend,
            sqlite_path=args.sqlite_path,
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
            print(f"Discovery run failed: {exc}", file=sys.stderr)
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


def format_result(result: DiscoveryRunResult) -> str:
    lines = [
        f"Discovery run {result.status}: {result.run_id}",
        f"queries: {result.query_count}",
        f"discovered: {result.discovered_count}",
        f"created: {result.created_count}",
        f"updated: {result.updated_count}",
        f"duplicates: {result.duplicate_count}",
    ]
    if result.errors:
        lines.append(f"errors: {len(result.errors)}")
        first_error = result.errors[0]
        lines.append(
            "first_error: "
            f"{first_error.source_id or 'unknown'} "
            f"{first_error.query_text or ''} "
            f"({first_error.error_type}: {first_error.message})"
        )
    return "\n".join(lines)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "local.example.json"


if __name__ == "__main__":
    raise SystemExit(main())
