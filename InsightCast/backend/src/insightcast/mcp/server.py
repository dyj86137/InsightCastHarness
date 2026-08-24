"""Small stdio MCP server backed by the InsightCast REST API.

The server intentionally contains no pipeline logic.  It translates MCP
``tools/list`` and ``tools/call`` requests into calls to the already-running
InsightCast FastAPI application, so the web UI and Agent integrations share
the same storage and run state.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _bootstrap_paths() -> None:
    """Allow ``python path/to/server.py`` from a source checkout."""

    current = Path(__file__).resolve()
    backend_src = current.parents[2]
    workspace_root = current.parents[6]
    harness_src = workspace_root / "myHarness" / "myHarness-V2" / "src"
    for path in (backend_src, harness_src):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_paths()


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "insightcast"
SERVER_VERSION = "0.1.0"


class InsightCastAPIError(RuntimeError):
    """Raised when the backing InsightCast API cannot complete a request."""


class InsightCastAPIClient:
    """Minimal async HTTP client using only Python's standard library."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get(
                "INSIGHTCAST_API_BASE_URL", "http://127.0.0.1:8766"
            )
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("INSIGHTCAST_MCP_HTTP_TIMEOUT", "30")
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._request_sync(method, path, query=query, body=body),
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]],
        body: Optional[Mapping[str, Any]],
    ) -> Any:
        params = {
            str(key): str(value)
            for key, value in (query or {}).items()
            if value is not None and value != ""
        }
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        payload = None
        headers = {"Accept": "application/json"}
        if body is not None:
            payload = json.dumps(dict(body), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=payload, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            detail = raw
            try:
                parsed = json.loads(raw)
                detail = parsed.get("detail", parsed) if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                pass
            raise InsightCastAPIError(
                f"InsightCast API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise InsightCastAPIError(
                f"InsightCast API is unavailable at {self.base_url}: {exc}"
            ) from exc


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    *,
    required: Iterable[str] = (),
    read_only: bool = True,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": read_only},
    }


TOOLS = [
    _tool(
        "start_monitoring_run",
        "Start one asynchronous InsightCast discovery and analysis pipeline run.",
        {
            "resume_run_id": {"type": "string", "description": "Optional failed run to resume."},
            "transcript_fetch_limit": {"type": "integer", "minimum": 1},
            "fail_fast": {"type": "boolean", "default": False},
        },
        read_only=False,
    ),
    _tool(
        "get_run_status",
        "Get the status, current stage, counts, and error for a monitoring run.",
        {"run_id": {"type": "string"}},
        required=("run_id",),
    ),
    _tool(
        "get_latest_brief",
        "Get the latest ready daily brief, or a brief for an explicit ISO date.",
        {"brief_date": {"type": "string", "description": "ISO date, for example 2026-08-21."}},
    ),
    _tool(
        "get_brief_detail",
        "Get one daily brief and its related interview details.",
        {"brief_key": {"type": "string", "description": "Brief id or ISO date."}},
        required=("brief_key",),
    ),
    _tool(
        "list_interviews",
        "List classified InsightCast interviews with optional source/person/status filters.",
        {
            "status": {"type": "string"},
            "source_id": {"type": "string"},
            "person_id": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
        },
    ),
    _tool(
        "get_dashboard",
        "Get daily monitoring counts and the five most recent runs.",
        {"dashboard_date": {"type": "string", "description": "ISO date; defaults to today."}},
    ),
]


class InsightCastMCPServer:
    """MCP JSON-RPC dispatcher for InsightCast tools."""

    def __init__(self, api_client: Optional[InsightCastAPIClient] = None) -> None:
        self.api = api_client or InsightCastAPIClient()

    async def handle(self, message: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC message; return ``None`` for notifications."""

        method = message.get("method")
        request_id = message.get("id")
        if method == "notifications/initialized" or method == "notifications/cancelled":
            return None
        if method == "ping":
            return self._result(request_id, {})
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
        if method == "tools/list":
            return self._result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            params = message.get("params") or {}
            return await self._handle_tool_call(request_id, params)
        return self._error(request_id, -32601, f"Method not found: {method}")

    async def _handle_tool_call(
        self,
        request_id: Any,
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            return self._error(request_id, -32602, "tools/call requires a tool name and object arguments")
        if not any(tool["name"] == name for tool in TOOLS):
            return self._error(request_id, -32602, f"Unknown tool: {name}")
        try:
            data = await self._invoke(name, dict(arguments))
        except (ValueError, InsightCastAPIError) as exc:
            return self._result(request_id, self._error_content(str(exc)))
        return self._result(request_id, self._success_content(data))

    async def _invoke(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "start_monitoring_run":
            _reject_unknown(args, {"resume_run_id", "transcript_fetch_limit", "fail_fast"})
            if args.get("transcript_fetch_limit") is not None:
                _require_int(args["transcript_fetch_limit"], "transcript_fetch_limit", minimum=1)
            if args.get("fail_fast") is not None and not isinstance(args["fail_fast"], bool):
                raise ValueError("fail_fast must be a boolean")
            return await self.api.request("POST", "/api/runs", body=args)
        if name == "get_run_status":
            _require_string(args, "run_id")
            return await self.api.request("GET", f"/api/runs/{_path_arg(args['run_id'])}")
        if name == "get_latest_brief":
            _reject_unknown(args, {"brief_date"})
            if args.get("brief_date"):
                _require_string(args, "brief_date")
                return await self.api.request("GET", f"/api/briefs/{_path_arg(args['brief_date'])}")
            response = await self.api.request(
                "GET", "/api/briefs", query={"status": "ready", "limit": 1}
            )
            if response.get("items"):
                return await self.api.request("GET", f"/api/briefs/{_path_arg(response['items'][0].get('id', ''))}")
            return {"found": False, "message": "No ready InsightCast brief is available."}
        if name == "get_brief_detail":
            _require_string(args, "brief_key")
            return await self.api.request("GET", f"/api/briefs/{_path_arg(args['brief_key'])}")
        if name == "list_interviews":
            _reject_unknown(args, {"status", "source_id", "person_id", "offset", "limit"})
            query = dict(args)
            if "offset" in query:
                _require_int(query["offset"], "offset", minimum=0)
            if "limit" in query:
                _require_int(query["limit"], "limit", minimum=1, maximum=200)
            return await self.api.request("GET", "/api/interviews", query=query)
        if name == "get_dashboard":
            _reject_unknown(args, {"dashboard_date"})
            return await self.api.request("GET", "/api/dashboard", query=args)
        raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def _result(request_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _success_content(data: Any) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=2, default=str)}],
            "structuredContent": data,
            "isError": False,
        }

    @staticmethod
    def _error_content(message: str) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }


async def _serve_stdio(server: InsightCastMCPServer) -> None:
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        try:
            message = json.loads(line)
            if not isinstance(message, Mapping):
                raise ValueError("JSON-RPC message must be an object")
            response = await server.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                sys.stdout.flush()
        except Exception as exc:  # Keep the stdio protocol alive after malformed input.
            request_id = message.get("id") if isinstance(message, Mapping) else None
            response = server._error(request_id, -32600, str(exc))
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def _path_arg(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _reject_unknown(args: Mapping[str, Any], allowed: set) -> None:
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"Unknown arguments: {', '.join(unknown)}")


def _require_string(args: Mapping[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int, maximum: Optional[int] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}")


def main() -> int:
    asyncio.run(_serve_stdio(InsightCastMCPServer()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
