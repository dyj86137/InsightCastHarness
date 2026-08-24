# Host Configuration

Start the InsightCast FastAPI service first. From the backend directory:

```powershell
$env:PYTHONPATH="src;..\..\..\myHarness\myHarness-V2\src"
uvicorn insightcast.api.main:app --host 127.0.0.1 --port 8766
```

The MCP process talks to that service through
`INSIGHTCAST_API_BASE_URL` (default: `http://127.0.0.1:8766`). The MCP command
can be started from any working directory because the launcher bootstraps the
source paths itself.

## Cursor

Put this in the project's `.cursor/mcp.json`, replacing the absolute path with
the checkout path on the target machine:

```json
{
  "mcpServers": {
    "insightcast": {
      "command": "python",
      "args": [
        "D:/Projects/Agent/InsightCast/V1/backend/src/insightcast/mcp/server.py"
      ],
      "env": {
        "INSIGHTCAST_API_BASE_URL": "http://127.0.0.1:8766"
      }
    }
  }
}
```

## GitHub Copilot in VS Code

Put this in `.vscode/mcp.json`:

```json
{
  "servers": {
    "insightcast": {
      "type": "stdio",
      "command": "python",
      "args": [
        "D:/Projects/Agent/InsightCast/V1/backend/src/insightcast/mcp/server.py"
      ],
      "env": {
        "INSIGHTCAST_API_BASE_URL": "http://127.0.0.1:8766"
      }
    }
  }
}
```

The exact UI and config filename can vary by Copilot host version; the MCP
server command and tool contract stay the same.
