"""Settings for the local InsightCast API process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class ApiSettings:
    backend_root: Path
    config_path: Path
    storage_backend: str
    sqlite_path: Path
    brief_output_dir: Path
    trace_dir: Path
    checkpoint_dir: Path
    error_log_dir: Path
    cors_origins: Tuple[str, ...]
    max_workers: int = 1

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
    ) -> "ApiSettings":
        source = os.environ if env is None else env
        backend_root = Path(__file__).resolve().parents[3]
        data_root = backend_root / "data"

        def path_value(name: str, default: Path) -> Path:
            value = source.get(name)
            return Path(value).expanduser() if value else default

        origins = tuple(
            origin.strip()
            for origin in source.get(
                "INSIGHTCAST_API_CORS_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        return cls(
            backend_root=backend_root,
            config_path=path_value(
                "INSIGHTCAST_API_CONFIG_PATH",
                backend_root / "configs" / "local.example.json",
            ),
            storage_backend=source.get("INSIGHTCAST_API_STORAGE_BACKEND", "sqlite"),
            sqlite_path=path_value(
                "INSIGHTCAST_API_SQLITE_PATH",
                data_root / "insightcast.sqlite3",
            ),
            brief_output_dir=path_value(
                "INSIGHTCAST_API_BRIEF_OUTPUT_DIR",
                data_root / "briefs",
            ),
            trace_dir=path_value(
                "INSIGHTCAST_API_TRACE_DIR",
                data_root / "traces",
            ),
            checkpoint_dir=path_value(
                "INSIGHTCAST_API_CHECKPOINT_DIR",
                data_root / "checkpoints",
            ),
            error_log_dir=path_value(
                "INSIGHTCAST_API_ERROR_LOG_DIR",
                data_root / "error_logs",
            ),
            cors_origins=origins,
            max_workers=max(1, int(source.get("INSIGHTCAST_API_MAX_WORKERS", "1"))),
        )
