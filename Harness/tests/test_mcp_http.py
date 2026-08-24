from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from tools.mcp import (
    mcp_tool_result_from_call_response,
    mcp_tool_spec_from_payload,
    parse_mcp_http_response,
)


class HttpMCPClientHelpersTests(unittest.TestCase):
    def test_parse_json_response(self) -> None:
        response = parse_mcp_http_response(
            '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
            content_type="application/json",
        )

        self.assertEqual(response["result"]["tools"], [])

    def test_parse_sse_response(self) -> None:
        response = parse_mcp_http_response(
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n',
            content_type="text/event-stream",
        )

        self.assertTrue(response["result"]["ok"])

    def test_tool_spec_uses_input_schema(self) -> None:
        spec = mcp_tool_spec_from_payload(
            {
                "name": "lemonade_transcribe_audio",
                "description": "Transcribe audio",
                "inputSchema": {
                    "type": "object",
                    "properties": {"audio_path": {"type": "string"}},
                },
            }
        )

        self.assertEqual(spec.name, "lemonade_transcribe_audio")
        self.assertIn("audio_path", spec.input_schema["properties"])

    def test_tool_call_response_content_to_result(self) -> None:
        result = mcp_tool_result_from_call_response(
            {
                "content": [{"type": "text", "text": "hello"}],
                "isError": False,
            }
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.output, "hello")


if __name__ == "__main__":
    unittest.main()
