"""基于 ReAct 的受控内容发现 Agent。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from context.builder import ContextBuilder
from core.message import Message
from core.state import AgentState
from core.tool import ToolCall, ToolResult
from infra.config import ContextConfig, LLMConfig, ToolConfig
from llm import BaseLLM, create_llm
from tools.base import BaseTool, EmptyToolInput, ToolExecutionContext
from tools.executor import ToolExecutor

from insightcast.agent.llm_trace import InsightCastReactLoop
from insightcast.domain.models import CandidateItem, DomainModel, SearchQuery
from insightcast.harness_events import (
    HarnessLoopContext,
    current_harness_event_recorder,
)
from insightcast.sources import (
    create_discovery_source,
    generate_and_save_search_queries,
    ingest_candidates,
)
from insightcast.sources.discovery import DiscoveryError, JsonFetcher
from insightcast.storage.repositories import InsightCastRepositories


PREPARE_DISCOVERY_QUERIES_TOOL_NAME = "insightcast_prepare_discovery_queries"
RUN_DISCOVERY_QUERIES_TOOL_NAME = "insightcast_run_discovery_queries"


DISCOVERY_RESEARCH_REACT_SYSTEM_PROMPT = """你是一个搜索 Agent。

你必须严格使用 ReAct 格式。

可用工具：
- insightcast_prepare_discovery_queries：生成并持久化配置中的搜索查询。
- insightcast_run_discovery_queries：执行已持久化的查询、规范化候选内容，并使用确定性的去重逻辑入库。

必须遵循的流程：
1. 调用 insightcast_prepare_discovery_queries。
2. 查看 Observation，再调用 insightcast_run_discovery_queries。
3. 如果发现结果 status 为 failed 且 discovered_count 为 0，在 Final Answer 中说明失败原因。
4. 否则在 Final Answer 中返回 status 和 discovered_count。

不要编造 URL 或候选内容。候选入库只能通过 discovery 工具完成。
"""


class DiscoveryResearchAgentResult(DomainModel):
    """DiscoveryResearchAgent 返回的结果。"""

    status: str
    query_count: int = 0
    discovered_count: int = 0
    created_count: int = 0
    updated_count: int = 0
    duplicate_count: int = 0
    candidate_ids: Tuple[str, ...] = ()
    errors: Tuple[Dict[str, Any], ...] = ()
    final_answer: str = ""


class PrepareDiscoveryQueriesTool(BaseTool):
    """生成并持久化配置中的发现查询。"""

    name = PREPARE_DISCOVERY_QUERIES_TOOL_NAME
    description = "生成并持久化 InsightCast 发现查询。"
    input_model = EmptyToolInput
    is_read_only = False
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "discovery", "query")
    metadata = {"component": "discovery"}

    def __init__(self, repositories: InsightCastRepositories) -> None:
        self.repositories = repositories

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del arguments, context
        queries = generate_and_save_search_queries(self.repositories)
        data = {
            "query_count": len(queries),
            "queries": [_query_payload(query) for query in queries],
        }
        return ToolResult.success(
            tool_call=tool_call,
            output=f"已准备 {len(queries)} 条发现查询。",
            data=data,
            metadata={"query_count": len(queries)},
        )


class RunDiscoveryQueriesTool(BaseTool):
    """执行已持久化的发现查询并将候选内容入库。"""

    name = RUN_DISCOVERY_QUERIES_TOOL_NAME
    description = "执行发现查询，规范化结果，并将候选内容入库。"
    input_model = EmptyToolInput
    is_read_only = False
    source = "insightcast"
    version = "1"
    tags = ("insightcast", "discovery", "ingest")
    metadata = {"component": "discovery"}

    def __init__(
        self,
        repositories: InsightCastRepositories,
        *,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
    ) -> None:
        self.repositories = repositories
        self.env = env
        self.fetch_json = fetch_json
        self.last_result: Optional[DiscoveryResearchAgentResult] = None

    async def execute(
        self,
        tool_call: ToolCall,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del arguments, context
        queries = tuple(self.repositories.search_queries.list())
        if not queries:
            queries = generate_and_save_search_queries(self.repositories)

        candidates, errors = self._discover_all(queries)
        ingest_result = ingest_candidates(self.repositories.candidates, candidates)
        result = DiscoveryResearchAgentResult(
            status=_final_status(
                query_count=len(queries),
                discovered_count=len(candidates),
                errors=tuple(errors),
            ),
            query_count=len(queries),
            discovered_count=ingest_result.discovered_count,
            created_count=ingest_result.created_count,
            updated_count=ingest_result.updated_count,
            duplicate_count=ingest_result.duplicate_count,
            candidate_ids=ingest_result.candidate_ids,
            errors=tuple(errors),
        )
        self.last_result = result
        data = result.to_dict()
        return ToolResult.success(
            tool_call=tool_call,
            output="发现完成：" + json.dumps(data, ensure_ascii=False),
            data=data,
            metadata={
                "status": result.status,
                "query_count": result.query_count,
                "discovered_count": result.discovered_count,
                "error_count": len(result.errors),
            },
        )

    def _discover_all(
        self,
        queries: Sequence[SearchQuery],
    ) -> Tuple[Tuple[CandidateItem, ...], Tuple[Dict[str, Any], ...]]:
        sources = {source.id: source for source in self.repositories.sources.list_enabled()}
        adapters: Dict[str, Any] = {}
        candidates: List[CandidateItem] = []
        errors: List[Dict[str, Any]] = []

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
                errors.append(_run_error_payload(query, exc))

        return tuple(candidates), tuple(errors)


class DiscoveryResearchAgent:
    """使用 myHarness ReactLoop 编排发现阶段。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        env: Optional[Mapping[str, str]] = None,
        fetch_json: Optional[JsonFetcher] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        timeout_seconds: Optional[float] = None,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.env = env
        self.fetch_json = fetch_json
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations

    async def run_async(
        self,
        *,
        repositories: InsightCastRepositories,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> DiscoveryResearchAgentResult:
        recorder = current_harness_event_recorder()
        active_run_id = (
            recorder.run_id
            if recorder is not None and recorder.run_id is not None
            else run_id
        )
        active_session_id = (
            recorder.session_id
            if recorder is not None and recorder.session_id is not None
            else session_id
        )
        start_step = recorder.step if recorder is not None and recorder.step is not None else 0
        run_tool = RunDiscoveryQueriesTool(
            repositories,
            env=self.env,
            fetch_json=self.fetch_json,
        )
        loop = InsightCastReactLoop(
            llm=self.llm,
            tool_executor=ToolExecutor.from_tools(
                (
                    PrepareDiscoveryQueriesTool(repositories),
                    run_tool,
                ),
                config=ToolConfig(allow_mutating_tools=True),
            ),
            context_builder=ContextBuilder(
                ContextConfig(max_messages=20, max_input_tokens=6000),
                system_message=Message.system(DISCOVERY_RESEARCH_REACT_SYSTEM_PROMPT),
            ),
            llm_config=LLMConfig(
                provider=self.llm.provider_name,
                model=self.model or self.llm.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds or 60.0,
            ),
            trace_metadata={
                "stage": "discovery",
                "task": "insightcast_discovery_research",
                "agent": "discovery_research_agent",
            },
            trace_env=self.env,
        )
        state = AgentState(
            run_id=active_run_id or "discovery_research_agent",
            session_id=active_session_id,
            step=start_step,
            max_steps=start_step + self.max_iterations,
            messages=(Message.user(build_discovery_research_agent_prompt(repositories)),),
            metadata={
                "stage": "discovery",
                "agent": "discovery_research_agent",
            },
        ).mark_running()
        final_state = await loop.run(state, HarnessLoopContext(recorder))
        if run_tool.last_result is None:
            raise RuntimeError("DiscoveryResearchAgent finished without running discovery.")
        return run_tool.last_result.clone(final_answer=final_state.final_output or "")


def build_discovery_research_agent_prompt(
    repositories: InsightCastRepositories,
) -> str:
    return (
        "为已配置的人物、来源和兴趣运行 InsightCast 发现阶段。\n"
        f"enabled_people: {repositories.people.count()}\n"
        f"enabled_sources: {repositories.sources.count()}\n"
        "必须按要求顺序使用工具，不要编造候选内容。"
    )


def create_discovery_llm_from_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    prefix: str = "INSIGHTCAST_DISCOVERY_LLM",
) -> BaseLLM:
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


def _query_payload(query: SearchQuery) -> Dict[str, Any]:
    return {
        "id": query.id,
        "text": query.text,
        "source_id": query.source_id,
        "source_type": query.source_type.value if query.source_type else None,
        "person_id": query.person_id,
        "language": query.language,
    }


def _require_source_id(query: SearchQuery) -> str:
    if not query.source_id:
        raise DiscoveryError(f"SearchQuery missing source_id: {query.id}")
    return query.source_id


def _run_error_payload(query: SearchQuery, exc: Exception) -> Dict[str, Any]:
    return {
        "query_id": query.id,
        "query_text": query.text,
        "source_id": query.source_id,
        "source_type": query.source_type.value if query.source_type else None,
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }


def _final_status(
    *,
    query_count: int,
    discovered_count: int,
    errors: Tuple[Dict[str, Any], ...],
) -> str:
    if errors and (query_count == len(errors) or discovered_count == 0):
        return "failed"
    if errors:
        return "partial"
    return "finished"


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
    "DISCOVERY_RESEARCH_REACT_SYSTEM_PROMPT",
    "DiscoveryResearchAgent",
    "DiscoveryResearchAgentResult",
    "PREPARE_DISCOVERY_QUERIES_TOOL_NAME",
    "PrepareDiscoveryQueriesTool",
    "RUN_DISCOVERY_QUERIES_TOOL_NAME",
    "RunDiscoveryQueriesTool",
    "build_discovery_research_agent_prompt",
    "create_discovery_llm_from_env",
]
