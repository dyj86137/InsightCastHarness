# 主机配置

请先启动 InsightCast FastAPI 服务。在 backend 目录中执行：

```powershell
$env:PYTHONPATH="src;..\..\..\myHarness\myHarness-V2\src"
uvicorn insightcast.api.main:app --host 127.0.0.1 --port 8766
```

MCP 进程通过 `INSIGHTCAST_API_BASE_URL`（默认值：`http://127.0.0.1:8766`）与该服务通信。
由于启动器会自动配置源代码路径，MCP 命令可以从任意工作目录启动。

## Cursor

将以下内容放入项目的 `.cursor/mcp.json`，并将绝对路径替换为目标机器上的项目路径：

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

## VS Code 中的 GitHub Copilot

将以下内容放入 `.vscode/mcp.json`：

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

具体界面和配置文件名可能因 Copilot 宿主版本而异；MCP 服务命令和工具契约保持不变。
