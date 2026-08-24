"""InsightCast 发现阶段的业务 Runner。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import Field

from insightcast.agent.discovery_research_agent import DiscoveryResearchAgent
from insightcast.domain.models import CandidateItem, DomainModel, RunRecord, SearchQuery
from insightcast.sources import (
    create_discovery_source,
    generate_and_save_search_queries,
    ingest_candidates,
)
from insightcast.sources.discovery import DiscoveryError, JsonFetcher
from insightcast.storage.repositories import InsightCastRepositories


class DiscoveryRunError(DomainModel):
    """单条查询或来源在发现阶段的错误。"""

    query_id: Optional[str] = None
    query_text: Optional[str] = None
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    error_type: str
    message: str


class DiscoveryRunResult(DomainModel):
    """一次发现阶段运行的结果。"""

    run_id: str
    status: str
    query_count: int = 0
    discovered_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    duplicate_count: int = 0
    candidate_ids: Tuple[str, ...] = Field(default_factory=tuple)
    errors: Tuple[DiscoveryRunError, ...] = Field(default_factory=tuple)


class DiscoveryRunner:
    """编排 query 生成、真实来源发现和候选入库。"""

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
        discovery_agent: Optional[DiscoveryResearchAgent] = None,
        fail_fast: bool = False,
    ) -> None:
        self.repositories = repositories
        self.env = env
        self.fetch_json = fetch_json
        self.discovery_agent = discovery_agent
        self.fail_fast = fail_fast

    def run_once(self) -> DiscoveryRunResult:
        """执行一次发现任务。"""

        run = self.repositories.run_records.create(
            RunRecord(
                status="running",
                metadata={"stage": "discovery"},
            )
        )

        try:
            result = self._run(run)
        except Exception as exc:
            self.repositories.run_records.finish(
                run.id,
                status="failed",
                error=str(exc),
            )
            raise

        self.repositories.run_records.finish(
            run.id,
            status=result.status,
            error=_error_summary(result.errors),
            discovered_count=result.discovered_count,
            accepted_count=result.created_count + result.updated_count,
            pushed_count=0,
            metadata={
                "stage": "discovery",
                "query_count": result.query_count,
                "created_count": result.created_count,
                "updated_count": result.updated_count,
                "duplicate_count": result.duplicate_count,
                "candidate_ids": list(result.candidate_ids),
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        return result

    def _run(self, run: RunRecord) -> DiscoveryRunResult:
        agent_error: Optional[DiscoveryRunError] = None
        if self.discovery_agent is not None:
            try:
                return self._run_agent(run)
            except Exception as exc:
                agent_error = DiscoveryRunError(
                    error_type="DiscoveryResearchAgentFailed",
                    message=str(exc),
                )

        queries = generate_and_save_search_queries(self.repositories)
        candidates, errors = self._discover_all(queries)
        merged_errors = tuple(([agent_error] if agent_error is not None else [])) + errors
        ingest_result = ingest_candidates(self.repositories.candidates, candidates)
        status = _final_status(
            query_count=len(queries),
            discovered_count=len(candidates),
            errors=merged_errors,
        )
        return DiscoveryRunResult(
            run_id=run.id,
            status=status,
            query_count=len(queries),
            discovered_count=ingest_result.discovered_count,
            created_count=ingest_result.created_count,
            updated_count=ingest_result.updated_count,
            duplicate_count=ingest_result.duplicate_count,
            candidate_ids=ingest_result.candidate_ids,
            errors=merged_errors,
        )

    def _run_agent(self, run: RunRecord) -> DiscoveryRunResult:
        result = asyncio.run(
            self.discovery_agent.run_async(
                repositories=self.repositories,
                run_id=run.id,
            )
        )
        return DiscoveryRunResult(
            run_id=run.id,
            status=result.status,
            query_count=result.query_count,
            discovered_count=result.discovered_count,
            created_count=result.created_count,
            updated_count=result.updated_count,
            duplicate_count=result.duplicate_count,
            candidate_ids=result.candidate_ids,
            errors=tuple(DiscoveryRunError(**error) for error in result.errors),
        )

    def _discover_all(
        self,
        queries: Tuple[SearchQuery, ...],
    ) -> Tuple[Tuple[CandidateItem, ...], Tuple[DiscoveryRunError, ...]]:
        sources = {source.id: source for source in self.repositories.sources.list_enabled()}
        adapters: Dict[str, Any] = {}
        candidates: List[CandidateItem] = []
        errors: List[DiscoveryRunError] = []

        for query in queries:
            try:
                source_id = _require_source_id(query)
                source = sources[source_id]
                adapter = adapters.get(source_id)
                if adapter is None:
                    adapter = create_discovery_source(
                        source,
                        env=self.env,
                        fetch_json=self.fetch_json,
                    )
                    adapters[source_id] = adapter
                candidates.extend(adapter.discover(query))
            except Exception as exc:
                error = _run_error(query, exc)
                errors.append(error)
                if self.fail_fast:
                    raise DiscoveryError(error.message) from exc

        return tuple(candidates), tuple(errors)


def _require_source_id(query: SearchQuery) -> str:
    if not query.source_id:
        raise DiscoveryError(f"SearchQuery missing source_id: {query.id}")
    return query.source_id


def _run_error(query: SearchQuery, exc: Exception) -> DiscoveryRunError:
    return DiscoveryRunError(
        query_id=query.id,
        query_text=query.text,
        source_id=query.source_id,
        source_type=query.source_type.value if query.source_type else None,
        error_type=exc.__class__.__name__,
        message=str(exc),
    )


def _final_status(
    *,
    query_count: int,
    discovered_count: int,
    errors: Tuple[DiscoveryRunError, ...],
) -> str:
    if errors and (query_count == len(errors) or discovered_count == 0):
        return "failed"
    if errors:
        return "partial"
    return "finished"


def _error_summary(errors: Tuple[DiscoveryRunError, ...]) -> Optional[str]:
    if not errors:
        return None
    return f"{len(errors)} discovery errors"


__all__ = [
    "DiscoveryRunError",
    "DiscoveryRunResult",
    "DiscoveryRunner",
]
