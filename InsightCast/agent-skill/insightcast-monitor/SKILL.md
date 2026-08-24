---
name: insightcast-monitor
description: 查询并运行 InsightCast 行业监测服务。当用户询问最新行业动态、每日简报、已监测访谈、运行状态，或明确要求刷新监测数据时使用。
---

# InsightCast 监测器

使用 InsightCast MCP 工具，根据已持久化的监测数据回答问题。调用工具前，必须将 MCP
服务配置为 `insightcast`，并确保 InsightCast API 已运行。

## 工作流程

1. 对于最新动态相关的问题，先调用 `get_latest_brief`。
2. 如果简报不存在、对于用户要求的时间范围已经过时，或用户明确要求刷新，则调用
   `start_monitoring_run`。
3. 将返回的 `run_id` 视为异步任务。持续轮询 `get_run_status`，直到任务完成或失败；
   任务运行期间不要编造结果。
4. 任务完成后，调用 `get_latest_brief` 或 `get_brief_detail`；如果有返回标题、来源、
   URL 和日期，应在回答中引用这些信息。
5. 使用 `list_interviews` 查询指定人物、来源或状态，使用 `get_dashboard` 获取统计数
   据和最近运行摘要。

## 使用约束

- 如果已经存在合适的最新简报，不要仅为了回答问题而调用 `start_monitoring_run`。
- 用清晰的语言说明 API 或处理流水线错误，并保留返回的 `run_id`，以便用户重试或恢复任务。
- 不要将 InsightCast 返回的证据和摘要之外的内容声称为事实。
- 人物、行业和来源的配置变更不属于当前技能工具集的范围，必须获得用户的明确请求。
