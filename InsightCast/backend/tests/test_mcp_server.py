from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.mcp.server import InsightCastMCPServer


class FakeAPIClient:
    def __init__(self) -> None:
        self.calls = []

    async def request(self, method, path, *, query=None, body=None):
        self.calls.append((method, path, query, body))
        if path == "/api/briefs":
            return {"items": [{"id": "brief_1"}], "total": 1}
        return {"ok": True, "method": method, "path": path, "query": query, "body": body}


class MCPServerTests(unittest.TestCase):
    def test_initialize_and_tools_list(self) -> None:
        server = InsightCastMCPServer(FakeAPIClient())
        initialize = asyncio.run(server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
        self.assertEqual(initialize["result"]["serverInfo"]["name"], "insightcast")
        tools = asyncio.run(server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        self.assertEqual(
            {item["name"] for item in tools["result"]["tools"]},
            {
                "start_monitoring_run",
                "get_run_status",
                "get_latest_brief",
                "get_brief_detail",
                "list_interviews",
                "get_dashboard",
            },
        )

    def test_tool_calls_map_to_api_routes(self) -> None:
        client = FakeAPIClient()
        server = InsightCastMCPServer(client)
        calls = [
            {"name": "start_monitoring_run", "arguments": {"fail_fast": True}},
            {"name": "get_run_status", "arguments": {"run_id": "run 1"}},
            {"name": "get_latest_brief", "arguments": {}},
            {"name": "get_brief_detail", "arguments": {"brief_key": "2026-08-21"}},
            {"name": "list_interviews", "arguments": {"limit": 5}},
            {"name": "get_dashboard", "arguments": {"dashboard_date": "2026-08-21"}},
        ]
        for index, params in enumerate(calls, start=1):
            response = asyncio.run(
                server.handle({"jsonrpc": "2.0", "id": index, "method": "tools/call", "params": params})
            )
            self.assertFalse(response["result"]["isError"], response)

        self.assertEqual(client.calls[0], ("POST", "/api/runs", None, {"fail_fast": True}))
        self.assertEqual(client.calls[1][1], "/api/runs/run%201")
        self.assertEqual(client.calls[2][1], "/api/briefs")
        self.assertEqual(client.calls[3][1], "/api/briefs/brief_1")
        self.assertEqual(client.calls[4][1], "/api/briefs/2026-08-21")
        self.assertEqual(client.calls[5], ("GET", "/api/interviews", {"limit": 5}, None))
        self.assertEqual(client.calls[6], ("GET", "/api/dashboard", {"dashboard_date": "2026-08-21"}, None))

    def test_invalid_arguments_return_mcp_tool_error(self) -> None:
        server = InsightCastMCPServer(FakeAPIClient())
        response = asyncio.run(
            server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "get_run_status", "arguments": {}},
                }
            )
        )
        self.assertTrue(response["result"]["isError"])
        self.assertIn("run_id", response["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
