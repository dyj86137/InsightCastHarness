<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { ArrowLeft, CircleAlert, ExternalLink, FileText, Inbox, RefreshCw, Search, Sparkles } from "@lucide/vue";
import { api } from "./api";
import type { Interview, Paginated } from "./types";
import { formatDate, formatDuration, humanStatus, percent, statusTone } from "./display";

const props = defineProps<{ refreshToken: number }>();

const list = ref<Paginated<Interview> | null>(null);
const selectedInterview = ref<Interview | null>(null);
const selectedId = ref<string | null>(null);
const search = ref("");
const status = ref("all");
const loading = ref(true);
const detailLoading = ref(false);
const error = ref<string | null>(null);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    list.value = await api.interviews({ status: status.value === "all" ? undefined : status.value, limit: 200 });
    if (!selectedId.value && list.value.items[0]) selectedId.value = list.value.items[0].id;
    if (selectedId.value && !list.value.items.some((interview) => interview.id === selectedId.value)) selectedId.value = list.value.items[0]?.id ?? null;
  } catch (loadError) {
    error.value = loadError instanceof Error ? loadError.message : "无法加载访谈列表";
  } finally {
    loading.value = false;
  }
}

async function loadDetail() {
  if (!selectedId.value) {
    selectedInterview.value = null;
    return;
  }
  detailLoading.value = true;
  try {
    const response = await api.interview(selectedId.value);
    selectedInterview.value = response.data;
  } catch {
    selectedInterview.value = null;
  } finally {
    detailLoading.value = false;
  }
}

onMounted(() => { void load(); });
watch(() => props.refreshToken, () => { void load(); });
watch(status, () => { void load(); });
watch(selectedId, () => { void loadDetail(); });

const filteredInterviews = computed(() => {
  const value = search.value.trim().toLowerCase();
  if (!value) return list.value?.items ?? [];
  return (list.value?.items ?? []).filter((interview) => [interview.title, interview.source_name, ...interview.person_names].filter(Boolean).join(" ").toLowerCase().includes(value));
});

function summaryText(key: string): string {
  const value = selectedInterview.value?.summary?.[key];
  return typeof value === "string" ? value : "";
}

function summaryList(key: string): string[] {
  const value = selectedInterview.value?.summary?.[key];
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

function transcriptValue(key: string): string {
  const value = selectedInterview.value?.transcript?.[key];
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") return value.toLocaleString("zh-CN");
  return typeof value === "string" ? value : "--";
}
</script>

<template>
  <main class="page-content page-content-wide">
    <div class="page-heading"><div><div class="eyebrow">内容资产</div><h1>访谈详情</h1><p>查看已通过筛选的访谈、摘要、字幕状态和关联候选。</p></div></div>
    <div v-if="error" class="empty-state error-state"><CircleAlert :size="20" /><div><strong>无法加载访谈列表</strong><p>{{ error }}</p></div><button class="button button-secondary" @click="load"><RefreshCw :size="15" /> 重试</button></div>
    <div v-else class="interview-workspace">
      <section class="panel interview-index">
        <div class="table-toolbar"><div class="search-field"><Search :size="16" /><input v-model="search" aria-label="搜索访谈" placeholder="搜索标题、来源或人物" /></div><select v-model="status" aria-label="筛选访谈状态"><option value="all">全部状态</option><option value="new">新建</option><option value="transcript_pending">等待字幕</option><option value="transcript_ready">字幕已就绪</option><option value="summary_ready">摘要已生成</option><option value="push_ready">待推送</option><option value="pushed">已推送</option><option value="archived">已归档</option><option value="failed">失败</option></select><button class="icon-button" title="刷新访谈" @click="load"><RefreshCw :size="16" /></button></div>
        <div v-if="loading" class="loading-list"><div v-for="n in 6" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="filteredInterviews.length === 0" class="empty-state"><Inbox :size="22" /><strong>暂无访谈</strong><p>完成访谈筛选和转录后，内容会显示在这里。</p></div>
        <div v-else class="interview-list">
          <button v-for="interview in filteredInterviews" :key="interview.id" class="interview-row" :class="{ 'is-selected': selectedId === interview.id }" @click="selectedId = interview.id">
            <div class="resource-primary"><strong>{{ interview.title }}</strong><span>{{ interview.source_name || "未知来源" }} / {{ interview.person_names?.join(', ') || "未匹配人物" }}</span></div>
            <div class="interview-row-meta"><span>{{ formatDate(interview.published_at) }}</span><span>{{ interview.candidate_count ?? 0 }} 个候选</span></div>
            <span class="status-badge" :class="statusTone(interview.status)">{{ humanStatus(interview.status) }}</span>
          </button>
        </div>
        <div v-if="list" class="table-footer"><span>{{ filteredInterviews.length }} / {{ list.total }} 条访谈</span><span>已从 SQLite 更新</span></div>
      </section>

      <section class="panel interview-detail">
        <div v-if="detailLoading" class="loading-list"><div v-for="n in 5" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="!selectedInterview" class="empty-state"><FileText :size="22" /><strong>请选择一条访谈</strong><p>选择后，这里会显示摘要、字幕状态和关联候选。</p></div>
        <template v-else>
          <div class="detail-header"><button class="icon-button" title="关闭详情" @click="selectedId = null"><ArrowLeft :size="17" /></button><span class="eyebrow">访谈详情</span><a class="icon-button" title="打开原始链接" :href="selectedInterview.url" target="_blank" rel="noreferrer"><ExternalLink :size="16" /></a></div>
          <div class="detail-content interview-reader">
            <span class="status-badge" :class="statusTone(selectedInterview.status)">{{ humanStatus(selectedInterview.status) }}</span>
            <h2>{{ selectedInterview.title }}</h2>
            <div class="detail-meta"><span>{{ selectedInterview.source_name || "未知来源" }}</span><span>{{ humanStatus(selectedInterview.type) }}</span><span>{{ formatDuration(selectedInterview.duration_seconds) }}</span><span>{{ formatDate(selectedInterview.published_at) }}</span></div>
            <div class="interview-score-grid"><div><span>重要性</span><strong>{{ percent(selectedInterview.importance_score) }}</strong></div><div><span>新颖度</span><strong>{{ percent(selectedInterview.novelty_score) }}</strong></div><div><span>相关性</span><strong>{{ percent(selectedInterview.relevance_score) }}</strong></div></div>

            <section class="detail-section"><div class="section-label"><Sparkles :size="14" /> 摘要</div><template v-if="selectedInterview.summary"><p v-if="summaryText('summary')" class="summary-lead">{{ summaryText('summary') }}</p><div v-for="section in [['key_points', '核心观点'], ['potential_opportunities', '关注机会'], ['industry_judgements', '行业判断'], ['mentioned_companies', '提及公司'], ['mentioned_products', '提及产品']]" :key="section[0]" class="insight-group"><h4>{{ section[1] }}</h4><ul v-if="summaryList(section[0]).length"><li v-for="item in summaryList(section[0])" :key="item">{{ item }}</li></ul><span v-else class="muted">暂无记录</span></div></template><div v-else class="empty-inline">暂无摘要，等待后续处理。</div></section>

            <section class="detail-section"><div class="section-label"><FileText :size="14" /> 字幕状态</div><div v-if="selectedInterview.transcript" class="fact-list transcript-facts"><div><span>状态</span><strong>{{ humanStatus(selectedInterview.transcript.status as string) }}</strong></div><div><span>来源</span><strong>{{ humanStatus(selectedInterview.transcript.source as string) }}</strong></div><div><span>文本长度</span><strong>{{ transcriptValue('text_chars') }} 字符</strong></div><div><span>片段数量</span><strong>{{ transcriptValue('segment_count') }}</strong></div><div><span>包含文本</span><strong>{{ transcriptValue('has_text') }}</strong></div></div><div v-else class="empty-inline">暂无字幕记录。</div></section>

            <section class="detail-section"><div class="section-label">关联候选</div><div v-if="selectedInterview.candidates?.length" class="linked-candidates"><a v-for="candidate in selectedInterview.candidates" :key="candidate.id" class="linked-candidate" :href="candidate.url" target="_blank" rel="noreferrer"><div><strong>{{ candidate.title }}</strong><span>{{ candidate.source_name || "未知来源" }}</span></div><span class="status-badge" :class="statusTone(candidate.status)">{{ humanStatus(candidate.status) }}</span></a></div><div v-else class="empty-inline">暂无关联候选。</div></section>
          </div>
        </template>
      </section>
    </div>
  </main>
</template>
