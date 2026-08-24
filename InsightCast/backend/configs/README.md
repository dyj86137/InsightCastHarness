# InsightCast V1 Configs

V1 uses editable config files for people, sources, interests, and runtime settings. These files are runtime inputs, so they live outside the Python package.

The config loader supports either a single full config file or split collection files.

Single-file example:

```powershell
python -c "import sys; sys.path[:0]=['src', '../../../myHarness/myHarness-V2/src']; from insightcast.config import load_config; print(load_config('configs/local.example.json'))"
```

Top-level fields:

- `industries`: list of `IndustryProfile` records.
- `people`: list of `Person` records.
- `sources`: list of `Source` records.
- `interests`: list of `UserInterest` records. `user_interests` is also accepted.
- `metadata`: optional config metadata.

Directory mode looks for these optional files:

- `industries.json`, `industries.yaml`, or `industries.yml`
- `people.json`, `people.yaml`, or `people.yml`
- `sources.json`, `sources.yaml`, or `sources.yml`
- `interests.json`, `interests.yaml`, or `interests.yml`

Each split collection file can be either a raw list or an object containing the matching field name.

JSON is supported without extra dependencies. YAML requires `PyYAML`.

## Local Persistence

V1 keeps the repository interface stable and supports two local storage backends:

- `json`: default, stores one JSON file per collection.
- `sqlite`: stores collection payloads in a local SQLite database.

Run discovery with SQLite persistence:

```powershell
python src\insightcast\run\discovery_run.py --storage-backend sqlite --sqlite-path data\insightcast.sqlite3
```

When `--sqlite-path` is omitted, SQLite uses `backend/data/insightcast.sqlite3`.
The SQLite backend is the recommended local persistence mode before adding the
API server and frontend, because run records, pushed interviews, daily briefs,
and duplicate-prevention state survive process restarts in one database file.

Run the full V1 daily pipeline with SQLite and Markdown output:

```powershell
python src\insightcast\run\daily_run.py --storage-backend sqlite --sqlite-path data\insightcast.sqlite3 --brief-output-dir data\briefs --trace-dir data\traces --checkpoint-dir data\checkpoints --error-log-dir data\error_logs
```

The full pipeline runs discovery, interview classification, interview promotion,
metadata transcript extraction, structured summary, ranking, and Markdown brief
generation in order.

Trace and checkpoint are enabled by default:

- Trace files: `data/traces/{run_id}.jsonl`
- Checkpoint files: `data/checkpoints/{checkpoint_id}.json`
- Error log files: `data/error_logs/{run_id}.jsonl`, written by an InsightCast myHarness Hook through the myHarness JSON logger.
- LLM trace events: `llm_requested`, `llm_responded`, and `llm_failed` with stage, task, request id, model, finish reason, and token usage where available.

Resume a failed or interrupted pipeline from its latest checkpoint:

```powershell
python src\insightcast\run\daily_run.py --storage-backend sqlite --sqlite-path data\insightcast.sqlite3 --brief-output-dir data\briefs --trace-dir data\traces --checkpoint-dir data\checkpoints --error-log-dir data\error_logs --resume-run-id run_xxx
```

Use `--no-trace` or `--no-checkpoint` only for short local experiments.

## Real Discovery Settings

YouTube full-platform discovery uses the official YouTube Data API and requires an API key:

```powershell
$env:YOUTUBE_API_KEY="your-api-key"
```

The YouTube source can override the environment variable name with:

```json
"metadata": {
  "api_key_env": "YOUTUBE_API_KEY",
  "search_scope": "platform",
  "max_results": 25,
  "order": "relevance"
}
```

Bilibili discovery and transcript fetching use a platform-specific backend route:

- `bili-cli` is preferred for structured search, video subtitles, and audio extraction.
- The public Bilibili API is the read-only fallback for search and video subtitle metadata.
- Bilibili candidates never use `yt-dlp`; Vimeo、Dailymotion、Twitch 和 PeerTube
  候选可使用通用 `yt-dlp` 回退链路。

Install the preferred backend when audio extraction is needed:

```powershell
pipx install "bilibili-cli[audio]"
bili status --yaml
```

Optional routing controls:

```powershell
$env:INSIGHTCAST_BILIBILI_BACKENDS="bili-cli,bilibili-api"
$env:INSIGHTCAST_BILIBILI_CLI="bili"
```

The selected backend is stored in candidate and transcript metadata as
`insightcast_bilibili_backend` or `backend`, so a failed or degraded run can be
diagnosed without reading command output.

## Additional video sources

The example config includes disabled adapters for Vimeo, Dailymotion, Twitch,
and PeerTube. Enable each source only after its credentials or instance URL are
configured:

- Vimeo uses the official API and requires `VIMEO_ACCESS_TOKEN` (or the
  `access_token_env` value in source metadata).
- Dailymotion uses its public Data API for video search and does not require a
  credential for the public endpoint.
- Twitch uses Helix and requires `TWITCH_CLIENT_ID` plus
  `TWITCH_ACCESS_TOKEN`. Twitch has no global VOD keyword-search endpoint, so
  `source.platform_id` must be a broadcaster ID; the adapter filters that
  channel's VODs locally using `query_match_mode` (`all`, `any`, or `none`).
- PeerTube uses the REST API of one configured instance. Set `source.url` or
  `metadata.api_base_url` to the instance root; the adapter calls
  `/api/v1/search/videos`.

All four adapters return a normalized video candidate. The transcript layer
uses `yt-dlp` for caption and audio fallback on these platforms, subject to
the video's access controls and applicable terms.

## LLM Interview Classification

The interview classifier reuses the myHarness LLM abstraction. For OpenAI, set:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:INSIGHTCAST_CLASSIFIER_LLM_PROVIDER="openai"
$env:INSIGHTCAST_CLASSIFIER_LLM_MODEL="gpt-4.1-mini"
```

`INSIGHTCAST_CLASSIFIER_LLM_API_KEY` can be used instead of `OPENAI_API_KEY` if you want a classifier-specific key.

## LLM Structured Summary

The structured summary layer also reuses the myHarness LLM abstraction:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:INSIGHTCAST_SUMMARY_LLM_PROVIDER="openai"
$env:INSIGHTCAST_SUMMARY_LLM_MODEL="gpt-4.1-mini"
```

`INSIGHTCAST_SUMMARY_LLM_API_KEY` can be used instead of `OPENAI_API_KEY` if you want a summary-specific key.
