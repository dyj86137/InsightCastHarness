<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Activity, AlertTriangle, ArrowLeft, CheckCircle2, CircleAlert, Clock3, ExternalLink, FileJson, Inbox, RefreshCw, Search, Server, XCircle } from "@lucide/vue";
import { api } from "./api";
import type { Paginated, RunStatus, TraceEvent, TraceEvents } from "./types";
import { formatDateTime, humanStatus, prettyJson, statusTone } from "./display";

const props = defineProps<{ refreshToken: number }>();

const list = ref<Paginated<RunStatus> | null>(null);
const selectedId = ref<string | null>(null);
const detail = ref<RunStatus | null>(null);
const trace = ref<TraceEvents | null>(null);
const search = ref("");
const status = ref("all");
const loading = ref(true);
const detailLoading = ref(false);
const eventsLoading = ref(false);
const error = ref<string | null>(null);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    list.value = await api.runs({ status: status.value === "all" ? undefined : status.value, limit: 100 });
    if (!selectedId.value && list.value.items[0]) selectedId.value = list.value.items[0].run_id;
    if (selectedId.value && !list.value.items.some((run) => run.run_id === selectedId.value)) selectedId.value = list.value.items[0]?.run_id ?? null;
  } catch (loadError) {
    error.value = loadError instanceof Error ? loadError.message : "无法加载运行记录";
  } finally {
    loading.value = false;
  }
}

async function loadDetail() {
  if (!selectedId.value) {
    detail.value = null;
    trace.value = null;
    return;
  }
  detailLoading.value = true;
  eventsLoading.value = true;
  try {
    const [run, events] = await Promise.all([api.run(selectedId.value), api.runEvents(selectedId.value, { limit: 200 })]);
    detail.value = run;
    trace.value = events;
  } catch {
    detail.value = null;
    trace.value = null;
  } finally {
    detailLoading.value = false;
    eventsLoading.value = false;
  }
}

onMounted(() => { void load(); });
watch(() => props.refreshToken, () => { void load(); });
watch(status, () => { void load(); });
watch(selectedId, () => { void loadDetail(); });

const filteredRuns = computed(() => {
  const value = search.value.trim().toLowerCase();
  if (!value) return list.value?.items ?? [];
  return (list.value?.items ?? []).filter((run) => [run.run_id, run.stage, run.status, run.error].filter(Boolean).join(" ").toLowerCase().includes(value));
});

function eventLabel(event: TraceEvent["event"]): string {
  const labels: Record<string, string> = {
    llm_requested: "LLM 请求",
    llm_responded: "LLM 响应",
    tool_call_started: "工具调用开始",
    tool_call_completed: "工具调用完成",
    stage_started: "阶段开始",
    stage_completed: "阶段完成",
    run_started: "运行开始",
    run_completed: "运行完成",
    run_failed: "运行失败",
  };
  return labels[event.event_type ?? ""] ?? humanStatus(event.event_type);
}

function stageItems(): Array<{ name: string; status: string; error?: string }> {
  const stages = detail.value?.metadata?.stages;
  return Array.isArray(stages) ? stages.filter((stage): stage is { name: string; status: string; error?: string } => Boolean(stage && typeof stage === "object" && "name" in stage)).map((stage) => ({ name: String(stage.name), status: String(stage.status ?? "unknown"), error: stage.error ? String(stage.error) : undefined })) : [];
}

function eventPayload(event: TraceEvent["event"]): string {
  return prettyJson(event.payload ?? event.metadata);
}
</script>

<template>
  <main class="page-content page-content-wide">
    <div class="page-heading"><div><div class="eyebrow">运行可观测性</div><h1>运行监控</h1><p>查看流程状态、阶段进度、失败原因和 trace 事件。</p></div></div>
    <div v-if="error" class="empty-state error-state"><CircleAlert :size="20" /><div><strong>无法加载运行记录</strong><p>{{ error }}</p></div><button class="button button-secondary" @click="load"><RefreshCw :size="15" /> 重试</button></div>
    <div v-else class="run-workspace">
      <section class="panel run-index">
        <div class="table-toolbar"><div class="search-field"><Search :size="16" /><input v-model="search" aria-label="搜索运行记录" placeholder="搜索 run ID、阶段或错误" /></div><select v-model="status" aria-label="筛选运行状态"><option value="all">全部状态</option><option value="queued">排队中</option><option value="running">运行中</option><option value="completed">已完成</option><option value="failed">失败</option></select><button class="icon-button" title="刷新运行记录" @click="load"><RefreshCw :size="16" /></button></div>
        <div v-if="loading" class="loading-list"><div v-for="n in 6" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="filteredRuns.length === 0" class="empty-state"><Activity :size="22" /><strong>暂无运行记录</strong><p>启动一次流程后，运行记录会显示在这里。</p></div>
        <div v-else class="run-index-list">
          <button v-for="run in filteredRuns" :key="run.run_id" class="run-index-row" :class="{ 'is-selected': selectedId === run.run_id }" @click="selectedId = run.run_id">
            <div class="run-status-mark"><span class="status-dot" :class="statusTone(run.status)" /></div><div class="run-row-main"><strong>{{ run.stage ? humanStatus(run.stage) : "流程运行" }}</strong><span>{{ run.run_id }}</span><span>{{ formatDateTime(run.started_at) }}</span></div><span class="status-badge" :class="statusTone(run.status)">{{ humanStatus(run.status) }}</span>
          </button>
        </div>
        <div v-if="list" class="table-footer"><span>{{ filteredRuns.length }} / {{ list.total }} 次运行</span><span>状态来自 SQLite 和运行进程</span></div>
      </section>

      <section class="panel run-detail">
        <div v-if="detailLoading" class="loading-list"><div v-for="n in 5" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="!detail" class="empty-state"><Activity :size="22" /><strong>请选择一次运行</strong><p>选择后，这里会显示阶段状态和 trace 事件。</p></div>
        <template v-else>
          <div class="detail-header"><button class="icon-button" title="关闭详情" @click="selectedId = null"><ArrowLeft :size="17" /></button><span class="eyebrow">运行详情</span><a class="icon-button" title="打开 trace 文件目录" :href="`file://${detail.metadata.trace_path ?? ''}`" target="_blank" rel="noreferrer"><ExternalLink :size="16" /></a></div>
          <div class="detail-content">
            <div class="run-detail-title"><div><span class="status-badge" :class="statusTone(detail.status)">{{ humanStatus(detail.status) }}</span><h2>{{ detail.run_id }}</h2><p>{{ detail.stage ? `当前阶段：${humanStatus(detail.stage)}` : "暂无阶段信息" }}</p></div><div class="run-detail-time"><span>开始 {{ formatDateTime(detail.started_at) }}</span><span>结束 {{ formatDateTime(detail.finished_at) }}</span></div></div>
            <div class="metric-grid compact-metric-grid"><div class="metric-tile metric-teal"><div class="metric-icon"><Search :size="17" /></div><div class="metric-label">发现</div><div class="metric-value">{{ detail.discovered_count }}</div></div><div class="metric-tile metric-gold"><div class="metric-icon"><CheckCircle2 :size="17" /></div><div class="metric-label">接受</div><div class="metric-value">{{ detail.accepted_count }}</div></div><div class="metric-tile metric-coral"><div class="metric-icon"><ExternalLink :size="17" /></div><div class="metric-label">推送</div><div class="metric-value">{{ detail.pushed_count }}</div></div></div>
            <div v-if="detail.error" class="run-error"><XCircle :size="18" /><div><strong>运行失败</strong><p>{{ detail.error }}</p></div></div>

            <section class="detail-section"><div class="section-label"><Server :size="14" /> 阶段进度</div><div v-if="stageItems().length" class="stage-list"><div v-for="stage in stageItems()" :key="stage.name" class="stage-row"><span class="stage-mark" :class="statusTone(stage.status)">{{ stage.status === 'finished' || stage.status === 'completed' ? '✓' : stage.status === 'failed' ? '!' : '·' }}</span><div><strong>{{ humanStatus(stage.name) }}</strong><span>{{ humanStatus(stage.status) }}</span><p v-if="stage.error">{{ stage.error }}</p></div></div></div><div v-else class="empty-inline">当前运行没有记录阶段明细。</div></section>

            <section class="detail-section"><div class="section-label"><Clock3 :size="14" /> Trace 事件 <span class="section-count">{{ trace?.items.length ?? 0 }}</span></div><div v-if="eventsLoading" class="loading-list"><div v-for="n in 3" :key="n" class="loading-row"><span /><span /><span /></div></div><div v-else-if="!trace?.items.length" class="empty-inline">暂无 trace 事件。</div><div v-else class="event-list"><article v-for="item in trace.items" :key="`${item.offset}-${item.event.id ?? item.event.event_type}`" class="event-row"><div class="event-marker"><span>{{ item.event.step ?? item.offset }}</span></div><div class="event-body"><div class="event-topline"><strong>{{ eventLabel(item.event) }}</strong><span>{{ formatDateTime(item.event.created_at) }}</span></div><p>{{ item.event.message || "暂无事件说明" }}</p><div class="event-meta"><span>{{ item.event.level || "info" }}</span><span>offset {{ item.offset }}</span></div><details v-if="item.event.payload || item.event.metadata" class="event-payload"><summary><FileJson :size="13" /> 查看 payload</summary><pre>{{ eventPayload(item.event) }}</pre></details></div></article></div></section>

            <details class="metadata-block"><summary>查看运行 metadata</summary><pre>{{ prettyJson(detail.metadata) }}</pre></details>
          </div>
        </template>
      </section>
    </div>
  </main>
</template>
