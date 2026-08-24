---
name: insightcast-monitor
description: Query and run the InsightCast industry-monitoring service. Use when the user asks for the latest industry developments, a daily brief, monitored interviews, run status, or an explicit refresh of monitoring data.
---

# InsightCast Monitor

Use the InsightCast MCP tools to answer questions from the persisted monitoring
data. The MCP server must be configured as `insightcast` and the InsightCast API
must be running before calling tools.

## Workflow

1. For a question about recent developments, call `get_latest_brief` first.
2. If the brief is missing, stale for the user's requested time range, or the
   user explicitly asks to refresh, call `start_monitoring_run`.
3. Treat the returned `run_id` as asynchronous. Poll `get_run_status` until the
   run is completed or failed; do not invent results while it is running.
4. After completion, call `get_latest_brief` or `get_brief_detail` and cite the
   returned title, source, URL, and date when available.
5. Use `list_interviews` for a focused person/source/status lookup and
   `get_dashboard` for counts and recent-run summaries.

## Guardrails

- Do not call `start_monitoring_run` merely to answer a question when a suitable
  recent brief already exists.
- Explain API or pipeline errors plainly and preserve the returned `run_id` so
  the user can retry or resume it.
- Do not claim that an item is factual beyond the evidence and summaries
  returned by InsightCast.
- Configuration mutations (people, industries, and sources) are outside this
  skill's current tool set and require the user's explicit request.
