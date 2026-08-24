<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import type { Socket } from "socket.io-client";
import {
  Archive,
  Activity,
  ArrowLeft,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  Copy,
  ExternalLink,
  FileText,
  Inbox,
  LayoutDashboard,
  ListChecks,
  LoaderCircle,
  Menu,
  Newspaper,
  Play,
  Plus,
  RefreshCw,
  Mic2,
  Radio,
  Search,
  Server,
  Sparkles,
  Users,
  X,
  XCircle,
} from "@lucide/vue";
import { api } from "./api";
import {
  createRealtimeSocket,
  normalizeKeywords,
  type IndustryUpdate,
} from "./realtime";
import PersonsPage from "./PersonsPage.vue";
import SourcesPage from "./SourcesPage.vue";
import InterviewsPage from "./InterviewsPage.vue";
import RunsPage from "./RunsPage.vue";
import type {
  BriefData,
  BriefItem,
  BriefResponse,
  BriefSummary,
  Candidate,
  DashboardData,
  Interview,
  Paginated,
  RunStatus,
} from "./types";

type View = "dashboard" | "candidates" | "briefs" | "persons" | "sources" | "interviews" | "runs";
type ReviewAction = "accept" | "reject" | "duplicate" | "archive";

const navItems = [
  { id: "dashboard" as View, label: "总览", icon: LayoutDashboard },
  { id: "candidates" as View, label: "候选审核", icon: ListChecks },
  { id: "briefs" as View, label: "每日日报", icon: Newspaper },
  { id: "persons" as View, label: "人物库", icon: Users },
  { id: "sources" as View, label: "来源管理", icon: Radio },
  { id: "interviews" as View, label: "访谈详情", icon: Mic2 },
  { id: "runs" as View, label: "运行监控", icon: Activity },
];

const statusLabels: Record<string, string> = {
  accept: "接受",
  accepted: "已接受",
  archive: "归档",
  archived: "已归档",
  classified: "已分类",
  classification: "分类",
  completed: "已完成",
  discovered: "已发现",
  discovery: "发现",
  duplicate: "重复",
  failed: "失败",
  finished: "已完成",
  queued: "排队中",
  ranking: "排序",
  ready: "已就绪",
  rejected: "已拒绝",
  running: "运行中",
  summary_ready: "摘要已生成",
  summarization: "摘要生成",
  transcript_fetch: "获取字幕",
  pushed: "已推送",
  reject: "拒绝",
  interview: "访谈",
  commentary: "评论内容",
  short_clip: "短片",
  news: "新闻",
  podcast: "播客",
  video: "视频",
};

function initialView(): View {
  const value = window.location.hash.replace("#", "");
  return ["candidates", "briefs", "persons", "sources", "interviews", "runs"].includes(value) ? value as View : "dashboard";
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function formatDate(value?: string | null): string {
  if (!value) return "暂无日期";
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" }).format(new Date(value));
}

function formatDateTime(value?: string | null): string {
  if (!value) return "尚未开始";
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null) return "--";
  const minutes = Math.floor(seconds / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)}h ${minutes % 60}m` : `${minutes}m`;
}

function humanStatus(value?: string | null): string {
  if (!value) return "未知";
  if (statusLabels[value]) return statusLabels[value];
  return value.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function statusTone(value?: string | null): string {
  if (!value) return "neutral";
  if (["accepted", "ready", "pushed", "finished", "completed"].includes(value)) return "positive";
  if (["rejected", "failed", "duplicate"].includes(value)) return "negative";
  if (["classified", "running", "queued", "summary_ready"].includes(value)) return "warning";
  return "neutral";
}

const view = ref<View>(initialView());
const refreshToken = ref(0);
const toast = ref<string | null>(null);
const isRunning = ref(false);
const apiOnline = ref<boolean | null>(null);
const realtimeOnline = ref(false);
const keywordInput = ref("");
const subscribedKeywords = ref<string[]>(loadSavedKeywords());
let realtimeSocket: Socket | null = null;

function navigate(nextView: View) {
  window.location.hash = nextView;
  view.value = nextView;
}

async function runPipeline() {
  isRunning.value = true;
  try {
    const result = await api.createRun();
    toast.value = `Run ${result.run_id} queued`;
    refreshToken.value += 1;
  } catch (error) {
    toast.value = error instanceof Error ? error.message : "无法启动流程";
  } finally {
    isRunning.value = false;
  }
}

watch(toast, (value) => {
  if (!value) return;
  window.setTimeout(() => {
    if (toast.value === value) toast.value = null;
  }, 4200);
});

const onHashChange = () => { view.value = initialView(); };

function loadSavedKeywords(): string[] {
  try {
    const saved = JSON.parse(window.localStorage.getItem("insightcast.keyword-subscriptions") ?? "[]");
    return Array.isArray(saved)
      ? normalizeKeywords(saved.filter((item): item is string => typeof item === "string"))
      : [];
  } catch {
    return [];
  }
}

function saveKeywords() {
  window.localStorage.setItem("insightcast.keyword-subscriptions", JSON.stringify(subscribedKeywords.value));
}

function syncKeywordSubscription() {
  realtimeSocket?.emit("subscribe_keywords", { keywords: subscribedKeywords.value });
}

function addKeywords() {
  const additions = keywordInput.value.split(/[,，\n]/);
  const next = normalizeKeywords([...subscribedKeywords.value, ...additions]);
  if (next.length === subscribedKeywords.value.length) return;
  subscribedKeywords.value = next;
  keywordInput.value = "";
  saveKeywords();
  syncKeywordSubscription();
}

function removeKeyword(keyword: string) {
  subscribedKeywords.value = subscribedKeywords.value.filter((item) => item !== keyword);
  saveKeywords();
  syncKeywordSubscription();
}

function handleIndustryUpdate(update: IndustryUpdate) {
  toast.value = `订阅动态：${update.title}`;
  refreshToken.value += 1;
}

onMounted(() => {
  window.addEventListener("hashchange", onHashChange);
  api.health().then(() => { apiOnline.value = true; }).catch(() => { apiOnline.value = false; });
  realtimeSocket = createRealtimeSocket();
  realtimeSocket.on("connect", () => {
    realtimeOnline.value = true;
    syncKeywordSubscription();
  });
  realtimeSocket.on("disconnect", () => { realtimeOnline.value = false; });
  realtimeSocket.on("subscription_updated", (payload: { keywords?: string[] }) => {
    if (!Array.isArray(payload?.keywords)) return;
    subscribedKeywords.value = normalizeKeywords(payload.keywords);
    saveKeywords();
  });
  realtimeSocket.on("industry_update", handleIndustryUpdate);
});

onUnmounted(() => {
  window.removeEventListener("hashchange", onHashChange);
  realtimeSocket?.disconnect();
  realtimeSocket = null;
});

// Dashboard data
const dashboardDate = ref(today());
const dashboard = ref<DashboardData | null>(null);
const reviewQueue = ref<Candidate[]>([]);
const dashboardLoading = ref(true);
const dashboardError = ref<string | null>(null);

async function loadDashboard() {
  dashboardLoading.value = true;
  dashboardError.value = null;
  try {
    const [dashboardResult, candidatesResult] = await Promise.all([
      api.dashboard(dashboardDate.value),
      api.candidates({ limit: 200 }),
    ]);
    dashboard.value = dashboardResult;
    reviewQueue.value = candidatesResult.items.filter((candidate) => ["discovered", "classified"].includes(candidate.status)).slice(0, 6);
  } catch (error) {
    dashboardError.value = error instanceof Error ? error.message : "请求失败";
  } finally {
    dashboardLoading.value = false;
  }
}

watch([view, dashboardDate, refreshToken], () => {
  if (view.value === "dashboard") void loadDashboard();
}, { immediate: true });

const latestRun = computed(() => dashboard.value?.recent_runs?.[0]);

// Candidate review data
const candidateResult = ref<Paginated<Candidate> | null>(null);
const selectedCandidate = ref<Candidate | null>(null);
const candidateStatus = ref("all");
const candidateSearch = ref("");
const candidateLoading = ref(true);
const candidateBusy = ref(false);
const candidateError = ref<string | null>(null);
const duplicateTarget = ref("");

async function loadCandidates() {
  candidateLoading.value = true;
  candidateError.value = null;
  try {
    const result = await api.candidates({ status: candidateStatus.value === "all" ? undefined : candidateStatus.value, limit: 200 });
    candidateResult.value = result;
    if (selectedCandidate.value) {
      const refreshed = result.items.find((item) => item.id === selectedCandidate.value?.id);
      if (refreshed) selectedCandidate.value = refreshed;
    }
  } catch (error) {
    candidateError.value = error instanceof Error ? error.message : "请求失败";
  } finally {
    candidateLoading.value = false;
  }
}

watch([view, candidateStatus, refreshToken], () => {
  if (view.value === "candidates") void loadCandidates();
}, { immediate: true });

const filteredCandidates = computed(() => {
  const value = candidateSearch.value.trim().toLowerCase();
  if (!value) return candidateResult.value?.items ?? [];
  return (candidateResult.value?.items ?? []).filter((candidate) => [
    candidate.title,
    candidate.source_name,
    ...(candidate.detected_person_names ?? []),
  ].filter(Boolean).join(" ").toLowerCase().includes(value));
});

async function selectCandidate(candidate: Candidate) {
  selectedCandidate.value = candidate;
  duplicateTarget.value = "";
  try {
    const detail = await api.candidate(candidate.id);
    selectedCandidate.value = detail.data;
  } catch (error) {
    toast.value = error instanceof Error ? error.message : "无法加载候选内容";
  }
}

async function reviewCandidate(action: ReviewAction) {
  const selected = selectedCandidate.value;
  if (!selected) return;
  if (action === "duplicate" && !duplicateTarget.value) {
    toast.value = "请选择这条记录对应的重复候选";
    return;
  }
  candidateBusy.value = true;
  try {
    const response = await api.reviewCandidate(selected.id, action, duplicateTarget.value || undefined);
    selectedCandidate.value = response.data;
    await loadCandidates();
    toast.value = `候选已${humanStatus(action)}`;
  } catch (error) {
    toast.value = error instanceof Error ? error.message : "审核失败";
  } finally {
    candidateBusy.value = false;
  }
}

// Daily brief data
const briefList = ref<Paginated<BriefSummary> | null>(null);
const selectedBriefId = ref<string | null>(null);
const briefDetail = ref<BriefResponse | null>(null);
const briefLoading = ref(true);
const briefDetailLoading = ref(false);
const briefError = ref<string | null>(null);

async function loadBriefList() {
  briefLoading.value = true;
  briefError.value = null;
  try {
    const result = await api.briefs({ limit: 50 });
    briefList.value = result;
    if (!selectedBriefId.value && result.items[0]) selectedBriefId.value = result.items[0].id;
  } catch (error) {
    briefError.value = error instanceof Error ? error.message : "请求失败";
  } finally {
    briefLoading.value = false;
  }
}

async function loadBriefDetail() {
  if (!selectedBriefId.value) {
    briefDetail.value = null;
    return;
  }
  briefDetailLoading.value = true;
  try {
    briefDetail.value = await api.brief(selectedBriefId.value);
  } catch {
    briefDetail.value = null;
  } finally {
    briefDetailLoading.value = false;
  }
}

watch([view, refreshToken], () => {
  if (view.value === "briefs") void loadBriefList();
}, { immediate: true });
watch(selectedBriefId, () => { void loadBriefDetail(); });

const briefData = computed<BriefData | null>(() => briefDetail.value?.data ?? null);
const briefInterviews = computed(() => briefData.value?.interviews ?? []);

function sectionItems(section: string): BriefItem[] {
  return (briefData.value?.items ?? []).filter((item) => item.section === section);
}

function findInterview(id: string): Interview | undefined {
  return briefInterviews.value.find((interview) => interview.id === id);
}

function briefOpportunities(interview?: Interview): string[] {
  const opportunities = interview?.summary?.potential_opportunities;
  return Array.isArray(opportunities) ? opportunities.slice(0, 3) : [];
}
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand-lockup">
        <div class="brand-mark"><Sparkles :size="17" /></div>
        <div><strong>InsightCast</strong><span>洞察工作台</span></div>
      </div>

      <div class="sidebar-section-label">工作区</div>
      <nav class="main-nav" aria-label="主导航">
        <button v-for="item in navItems" :key="item.id" class="nav-item" :class="{ 'is-active': view === item.id }" @click="navigate(item.id)">
          <component :is="item.icon" :size="17" />
          <span>{{ item.label }}</span>
          <ChevronRight v-if="view === item.id" class="nav-chevron" :size="15" />
        </button>
      </nav>

      <div class="sidebar-footer">
        <div class="connection-state" :class="{ online: apiOnline === true, offline: apiOnline === false }">
          <span class="connection-dot" />
          <span>{{ apiOnline === true ? "API 已连接" : apiOnline === false ? "API 未连接" : "正在检查 API" }}</span>
        </div>
        <div class="connection-state" :class="{ online: realtimeOnline, offline: !realtimeOnline }">
          <span class="connection-dot" />
          <span>{{ realtimeOnline ? "实时订阅已连接" : "实时订阅重连中" }}</span>
        </div>
        <span class="version-label">V1 本地工作区</span>
      </div>
    </aside>

    <div class="main-area">
      <header class="topbar">
        <div class="mobile-brand"><Menu :size="20" /><strong>InsightCast</strong></div>
        <div class="topbar-spacer" />
        <div class="subscription-control">
          <div class="subscription-input"><Radio :size="15" /><input v-model="keywordInput" aria-label="订阅关键词" placeholder="订阅关键词" @keyup.enter="addKeywords" /><button class="icon-button" title="添加订阅关键词" @click="addKeywords"><Plus :size="15" /></button></div>
          <div v-if="subscribedKeywords.length" class="subscription-chips"><button v-for="keyword in subscribedKeywords" :key="keyword" class="subscription-chip" :title="`取消订阅 ${keyword}`" @click="removeKeyword(keyword)"><span>{{ keyword }}</span><X :size="12" /></button></div>
        </div>
        <button class="button button-primary" title="运行流程" :disabled="isRunning" @click="runPipeline">
          <LoaderCircle v-if="isRunning" class="spin" :size="16" /><Play v-else :size="16" />
          {{ isRunning ? "启动中" : "运行流程" }}
        </button>
      </header>

      <main v-if="view === 'dashboard'" class="page-content">
        <div class="page-heading">
          <div><div class="eyebrow">洞察工作台</div><h1>今日概览</h1><p>集中查看发现、审核和日报交付情况。</p></div>
          <label class="date-control"><span>查看日期</span><input v-model="dashboardDate" type="date" /></label>
        </div>

        <div v-if="dashboardError" class="empty-state error-state">
          <CircleAlert :size="20" /><div><strong>无法加载当前页面</strong><p>{{ dashboardError }}</p></div>
          <button class="button button-secondary" @click="loadDashboard"><RefreshCw :size="15" /> 重试</button>
        </div>
        <div v-else-if="dashboardLoading"><div class="loading-list"><div v-for="n in 3" :key="n" class="loading-row"><span /><span /><span /></div></div></div>
        <template v-else-if="dashboard">
          <section class="metric-grid">
            <div class="metric-tile metric-teal"><div class="metric-icon"><Search :size="19" /></div><div class="metric-label">发现数量</div><div class="metric-value">{{ dashboard.discovered_count }}</div><div class="metric-detail">{{ formatDate(dashboard.dashboard_date) }}</div></div>
            <div class="metric-tile metric-coral"><div class="metric-icon"><ArrowUpRight :size="19" /></div><div class="metric-label">已推送</div><div class="metric-value">{{ dashboard.pushed_count }}</div><div class="metric-detail">已发布到日报</div></div>
            <div class="metric-tile metric-gold"><div class="metric-icon"><ListChecks :size="19" /></div><div class="metric-label">待审核</div><div class="metric-value">{{ dashboard.pending_review_count }}</div><div class="metric-detail">已发现或已分类</div></div>
            <div class="metric-tile metric-ink"><div class="metric-icon"><Server :size="19" /></div><div class="metric-label">最近运行</div><div class="metric-value">{{ latestRun ? humanStatus(latestRun.status) : "--" }}</div><div class="metric-detail">{{ latestRun?.stage ? humanStatus(latestRun.stage) : "暂无运行记录" }}</div></div>
          </section>

          <section class="dashboard-grid">
            <div class="panel panel-table">
              <div class="panel-header"><div class="panel-title"><span class="panel-title-icon"><Clock3 :size="17" /></span><h2>最近运行</h2></div><button class="text-button" @click="navigate('candidates')">查看审核队列 <ArrowUpRight :size="14" /></button></div>
              <div v-if="dashboard.recent_runs.length === 0" class="empty-state"><Inbox :size="22" /><strong>暂无运行记录</strong><p>启动一次流程后，这里会显示运行记录。</p></div>
              <div v-else class="run-list">
                <div v-for="run in dashboard.recent_runs" :key="run.run_id" class="run-row">
                  <div class="run-status-mark"><span class="status-dot" :class="statusTone(run.status)" /></div>
                  <div class="run-row-main"><strong>{{ run.stage ? humanStatus(run.stage) : "流程运行" }}</strong><span>{{ run.run_id }} / {{ formatDateTime(run.started_at) }}</span></div>
                  <div class="run-counts"><span>{{ run.discovered_count }} 条发现</span><span>{{ run.pushed_count }} 条推送</span></div>
                  <span class="status-badge" :class="statusTone(run.status)">{{ humanStatus(run.status) }}</span>
                </div>
              </div>
            </div>

            <div class="panel panel-table">
              <div class="panel-header"><div class="panel-title"><span class="panel-title-icon"><ListChecks :size="17" /></span><h2>审核队列</h2></div><button class="text-button" @click="navigate('candidates')">打开队列 <ArrowUpRight :size="14" /></button></div>
              <div v-if="reviewQueue.length === 0" class="empty-state"><Inbox :size="22" /><strong>队列为空</strong><p>没有等待审核的已发现或已分类候选。</p></div>
              <div v-else class="queue-list">
                <button v-for="candidate in reviewQueue" :key="candidate.id" class="queue-row" @click="navigate('candidates')">
                  <div class="queue-row-main"><strong>{{ candidate.title }}</strong><span>{{ candidate.source_name || humanStatus(candidate.source_type) }} / {{ candidate.detected_person_names?.join(', ') || '未匹配到人物' }}</span></div>
                  <span class="status-badge" :class="statusTone(candidate.status)">{{ humanStatus(candidate.status) }}</span>
                </button>
              </div>
            </div>
          </section>
        </template>
      </main>

      <main v-if="view === 'candidates'" class="page-content page-content-wide">
        <div class="page-heading"><div><div class="eyebrow">候选审核</div><h1>处理发现队列</h1><p>查看模型判断，保留有效信号或将其移出队列。</p></div></div>
        <div v-if="candidateError" class="empty-state error-state">
          <CircleAlert :size="20" /><div><strong>无法加载当前页面</strong><p>{{ candidateError }}</p></div>
          <button class="button button-secondary" @click="loadCandidates"><RefreshCw :size="15" /> 重试</button>
        </div>
        <div v-else class="review-workspace">
          <section class="panel candidate-table-panel">
            <div class="table-toolbar">
              <div class="search-field"><Search :size="16" /><input v-model="candidateSearch" aria-label="搜索候选" placeholder="搜索标题、来源或人物" /></div>
              <select v-model="candidateStatus" aria-label="筛选候选状态"><option value="all">全部状态</option><option value="discovered">已发现</option><option value="classified">已分类</option><option value="accepted">已接受</option><option value="rejected">已拒绝</option><option value="duplicate">重复</option><option value="archived">已归档</option></select>
              <button class="icon-button" title="刷新候选" @click="loadCandidates"><RefreshCw :size="16" /></button>
            </div>
            <div class="candidate-table-head"><span>候选内容</span><span>模型判断</span><span>发布时间</span><span>状态</span></div>
            <div v-if="candidateLoading" class="loading-list"><div v-for="n in 6" :key="n" class="loading-row"><span /><span /><span /></div></div>
            <div v-else-if="filteredCandidates.length === 0" class="empty-state"><Inbox :size="22" /><strong>未找到候选内容</strong><p>请尝试其他状态或搜索条件。</p></div>
            <div v-else class="candidate-list">
              <button v-for="candidate in filteredCandidates" :key="candidate.id" class="candidate-row" :class="{ 'is-selected': selectedCandidate?.id === candidate.id }" @click="selectCandidate(candidate)">
                <div class="candidate-primary"><strong>{{ candidate.title }}</strong><span>{{ candidate.source_name || humanStatus(candidate.source_type) }} / {{ candidate.detected_person_names?.join(', ') || '未匹配到人物' }}</span></div>
                <div class="candidate-decision"><template v-if="candidate.interview_decision"><strong>{{ Math.round(candidate.interview_decision.confidence * 100) }}%</strong><span>{{ humanStatus(candidate.interview_decision.content_type) }}</span></template><span v-else class="muted">暂无判断</span></div>
                <span class="candidate-date">{{ formatDate(candidate.published_at) }}</span>
                <span class="status-badge" :class="statusTone(candidate.status)">{{ humanStatus(candidate.status) }}</span>
              </button>
            </div>
            <div v-if="candidateResult" class="table-footer"><span>{{ filteredCandidates.length }} / {{ candidateResult.total }} 条候选</span><span>已从 SQLite 更新</span></div>
          </section>

          <aside class="panel candidate-detail" :class="{ 'is-open': selectedCandidate }">
            <template v-if="selectedCandidate">
              <div class="detail-header"><button class="icon-button" title="关闭详情" @click="selectedCandidate = null"><ArrowLeft :size="17" /></button><span class="eyebrow">候选详情</span><a class="icon-button" title="打开原始链接" :href="selectedCandidate.url" target="_blank" rel="noreferrer"><ExternalLink :size="16" /></a></div>
              <div class="detail-content">
                <span class="status-badge" :class="statusTone(selectedCandidate.status)">{{ humanStatus(selectedCandidate.status) }}</span>
                <h2>{{ selectedCandidate.title }}</h2>
                <div class="detail-meta"><span>{{ selectedCandidate.source_name || humanStatus(selectedCandidate.source_type) }}</span><span>{{ formatDuration(selectedCandidate.duration_seconds) }}</span><span>{{ formatDate(selectedCandidate.published_at) }}</span></div>
                <p v-if="selectedCandidate.description" class="detail-description">{{ selectedCandidate.description }}</p>

                <div class="detail-section"><div class="section-label">模型判断</div>
                  <template v-if="selectedCandidate.interview_decision"><div class="decision-score"><strong>{{ Math.round(selectedCandidate.interview_decision.confidence * 100) }}%</strong><span>{{ humanStatus(selectedCandidate.interview_decision.content_type) }}</span></div><p>{{ selectedCandidate.interview_decision.reason || "暂无判断理由。" }}</p><div class="fact-list"><div><span>是否访谈</span><strong>{{ selectedCandidate.interview_decision.is_interview ? "是" : "否" }}</strong></div><div><span>目标人物在场</span><strong>{{ selectedCandidate.interview_decision.target_person_present ? "是" : "否" }}</strong></div><div><span>是否权威来源</span><strong>{{ selectedCandidate.interview_decision.is_original_or_authoritative_source ? "是" : "否" }}</strong></div></div></template>
                  <div v-else class="empty-state"><Inbox :size="22" /><strong>暂无模型判断</strong><p>只有在分类判断完成后，才能人工接受这个候选。</p></div>
                </div>

                <div class="detail-section"><div class="section-label">审核操作</div><div class="review-actions"><button class="button button-positive" :disabled="candidateBusy" @click="reviewCandidate('accept')"><Check :size="15" /> 接受</button><button class="button button-danger" :disabled="candidateBusy" @click="reviewCandidate('reject')"><XCircle :size="15" /> 拒绝</button><button class="button button-secondary" :disabled="candidateBusy" @click="reviewCandidate('archive')"><Archive :size="15" /> 归档</button></div><div class="duplicate-control"><select v-model="duplicateTarget" aria-label="重复候选目标"><option value="">选择重复目标</option><option v-for="candidate in filteredCandidates.filter((item) => item.id !== selectedCandidate?.id)" :key="candidate.id" :value="candidate.id">{{ candidate.title }}</option></select><button class="button button-secondary" :disabled="candidateBusy || !duplicateTarget" @click="reviewCandidate('duplicate')"><Copy :size="15" /> 标记重复</button></div></div>
              </div>
            </template>
            <div v-else class="empty-state"><Inbox :size="22" /><strong>请选择候选内容</strong><p>选择后，这里会显示分类依据和可用的审核操作。</p></div>
          </aside>
        </div>
      </main>

      <main v-if="view === 'briefs'" class="page-content page-content-wide">
        <div class="page-heading"><div><div class="eyebrow">日报输出</div><h1>每日日报</h1><p>按排名阅读重要内容，每条内容都附有对应背景。</p></div></div>
        <div v-if="briefError" class="empty-state error-state"><CircleAlert :size="20" /><div><strong>无法加载当前页面</strong><p>{{ briefError }}</p></div><button class="button button-secondary" @click="loadBriefList"><RefreshCw :size="15" /> 重试</button></div>
        <div v-else-if="briefLoading"><div class="loading-list"><div v-for="n in 4" :key="n" class="loading-row"><span /><span /><span /></div></div></div>
        <div v-else class="brief-workspace">
          <section class="panel brief-index"><div class="panel-header"><div class="panel-title"><span class="panel-title-icon"><Newspaper :size="17" /></span><h2>日报归档</h2></div><button class="icon-button" title="刷新日报" @click="loadBriefList"><RefreshCw :size="16" /></button></div><div v-if="!briefList || briefList.items.length === 0" class="empty-state"><Inbox :size="22" /><strong>暂无日报</strong><p>流程完成并生成日报后，内容会显示在这里。</p></div><div v-else class="brief-list"><button v-for="brief in briefList.items" :key="brief.id" class="brief-list-row" :class="{ 'is-selected': selectedBriefId === brief.id }" @click="selectedBriefId = brief.id"><div><span class="brief-date">{{ formatDate(brief.brief_date) }}</span><strong>{{ brief.title }}</strong></div><div class="brief-row-meta"><span>{{ brief.items?.length ?? 0 }} 条内容</span><span class="status-badge" :class="statusTone(brief.status)">{{ humanStatus(brief.status) }}</span></div></button></div></section>

          <section class="panel brief-reader">
            <div v-if="briefDetailLoading" class="loading-list"><div v-for="n in 4" :key="n" class="loading-row"><span /><span /><span /></div></div>
            <div v-else-if="!briefData" class="empty-state"><Inbox :size="22" /><strong>请选择一份日报</strong><p>从左侧归档中选择日期，打开日报阅读视图。</p></div>
            <template v-else>
              <div class="brief-reader-header"><div><div class="eyebrow">{{ formatDate(briefData.brief_date) }}</div><h2>{{ briefData.title }}</h2><div class="brief-reader-meta"><span class="status-badge" :class="statusTone(briefData.status)">{{ humanStatus(briefData.status) }}</span><span>{{ briefData.items?.length ?? 0 }} 条排名内容</span></div></div><a v-if="briefData.markdown_path" class="button button-secondary" :href="`file://${briefData.markdown_path}`" target="_blank" rel="noreferrer"><ExternalLink :size="15" /> 打开 Markdown</a></div>
              <div class="brief-sections">
                <section v-if="sectionItems('must_read').length" class="brief-section"><div class="brief-section-heading"><Sparkles :size="16" /><h3>必读</h3><span class="section-count">{{ sectionItems('must_read').length }}</span></div><article v-for="item in sectionItems('must_read')" :key="`${item.interview_id}-${item.rank}`" class="brief-item-row"><div class="brief-item-rank">{{ String(item.rank).padStart(2, '0') }}</div><div class="brief-item-content"><div class="brief-item-topline"><span>{{ item.person_names?.join(' / ') || findInterview(item.interview_id)?.person_names?.join(' / ') || '未匹配到人物' }}</span><span>{{ Math.round((item.score ?? 0) * 100) }} 分</span></div><h4>{{ item.title }}</h4><p v-if="findInterview(item.interview_id)?.summary?.summary">{{ findInterview(item.interview_id)?.summary?.summary }}</p><div v-if="briefOpportunities(findInterview(item.interview_id)).length" class="opportunity-row"><span v-for="opportunity in briefOpportunities(findInterview(item.interview_id))" :key="opportunity">{{ opportunity }}</span></div><div class="brief-item-footer"><span>{{ item.reason || '根据相关性和来源质量排序。' }}</span><a :href="item.url" target="_blank" rel="noreferrer">原始来源 <ExternalLink :size="13" /></a></div></div></article></section>
                <section v-if="sectionItems('worth_watching').length" class="brief-section"><div class="brief-section-heading"><FileText :size="16" /><h3>值得关注</h3><span class="section-count">{{ sectionItems('worth_watching').length }}</span></div><article v-for="item in sectionItems('worth_watching')" :key="`${item.interview_id}-${item.rank}`" class="brief-item-row"><div class="brief-item-rank">{{ String(item.rank).padStart(2, '0') }}</div><div class="brief-item-content"><div class="brief-item-topline"><span>{{ item.person_names?.join(' / ') || findInterview(item.interview_id)?.person_names?.join(' / ') || '未匹配到人物' }}</span><span>{{ Math.round((item.score ?? 0) * 100) }} 分</span></div><h4>{{ item.title }}</h4><p v-if="findInterview(item.interview_id)?.summary?.summary">{{ findInterview(item.interview_id)?.summary?.summary }}</p><div v-if="briefOpportunities(findInterview(item.interview_id)).length" class="opportunity-row"><span v-for="opportunity in briefOpportunities(findInterview(item.interview_id))" :key="opportunity">{{ opportunity }}</span></div><div class="brief-item-footer"><span>{{ item.reason || '根据相关性和来源质量排序。' }}</span><a :href="item.url" target="_blank" rel="noreferrer">原始来源 <ExternalLink :size="13" /></a></div></div></article></section>
                <section v-if="sectionItems('archived').length" class="brief-section"><div class="brief-section-heading"><Archive :size="16" /><h3>已归档</h3><span class="section-count">{{ sectionItems('archived').length }}</span></div><article v-for="item in sectionItems('archived')" :key="`${item.interview_id}-${item.rank}`" class="brief-item-row"><div class="brief-item-rank">{{ String(item.rank).padStart(2, '0') }}</div><div class="brief-item-content"><div class="brief-item-topline"><span>{{ item.person_names?.join(' / ') || findInterview(item.interview_id)?.person_names?.join(' / ') || '未匹配到人物' }}</span><span>{{ Math.round((item.score ?? 0) * 100) }} 分</span></div><h4>{{ item.title }}</h4><p v-if="findInterview(item.interview_id)?.summary?.summary">{{ findInterview(item.interview_id)?.summary?.summary }}</p><div v-if="briefOpportunities(findInterview(item.interview_id)).length" class="opportunity-row"><span v-for="opportunity in briefOpportunities(findInterview(item.interview_id))" :key="opportunity">{{ opportunity }}</span></div><div class="brief-item-footer"><span>{{ item.reason || '根据相关性和来源质量排序。' }}</span><a :href="item.url" target="_blank" rel="noreferrer">原始来源 <ExternalLink :size="13" /></a></div></div></article></section>
              </div>
            </template>
          </section>
        </div>
      </main>

      <PersonsPage v-if="view === 'persons'" :refresh-token="refreshToken" @toast="toast = $event" />
      <SourcesPage v-if="view === 'sources'" :refresh-token="refreshToken" @toast="toast = $event" />
      <InterviewsPage v-if="view === 'interviews'" :refresh-token="refreshToken" />
      <RunsPage v-if="view === 'runs'" :refresh-token="refreshToken" />
    </div>

    <div v-if="toast" class="toast" role="status"><CheckCircle2 :size="17" /><span>{{ toast }}</span><button class="icon-button icon-button-inverse" title="Dismiss" @click="toast = null"><X :size="15" /></button></div>
  </div>
</template>
