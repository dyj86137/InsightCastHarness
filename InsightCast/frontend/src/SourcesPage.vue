<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { CircleAlert, ExternalLink, Globe2, Plus, RefreshCw, Save, Search, Trash2, X } from "@lucide/vue";
import { api } from "./api";
import type { Paginated, Source } from "./types";
import { formatDate, percent, splitList } from "./display";

const props = defineProps<{ refreshToken: number }>();
const emit = defineEmits<{ toast: [message: string] }>();

const sourceTypes = [
  ["youtube", "YouTube"], ["bilibili", "Bilibili"], ["vimeo", "Vimeo"], ["dailymotion", "Dailymotion"],
  ["twitch", "Twitch"], ["peertube", "PeerTube"], ["podcast_rss", "播客 RSS"],
  ["rss", "媒体 RSS"], ["website", "网站"], ["manual", "手动录入"], ["other", "其他"],
];
const industryOptions = [
  ["ai", "人工智能"], ["ecommerce", "电子商务"], ["retail", "零售"],
  ["consumer", "消费"], ["finance", "金融"], ["cloud", "云计算"],
  ["enterprise_software", "企业软件"], ["media", "媒体"], ["healthcare", "医疗健康"],
  ["auto", "汽车"], ["energy", "能源"], ["other", "其他"],
];

interface SourceForm {
  name: string;
  type: string;
  url: string;
  platform_id: string;
  authority: number;
  enabled: boolean;
  languages: string;
  industries: string[];
}

function emptyForm(): SourceForm {
  return { name: "", type: "other", url: "", platform_id: "", authority: 0.5, enabled: true, languages: "", industries: [] };
}

function formFromSource(source: Source): SourceForm {
  return { name: source.name, type: source.type, url: source.url ?? "", platform_id: source.platform_id ?? "", authority: source.authority, enabled: source.enabled, languages: source.languages.join(", "), industries: [...source.industries] };
}

const list = ref<Paginated<Source> | null>(null);
const selected = ref<Source | null>(null);
const form = ref<SourceForm>(emptyForm());
const filter = ref("all");
const search = ref("");
const loading = ref(true);
const busy = ref(false);
const error = ref<string | null>(null);
const isCreating = ref(false);

async function load() {
  loading.value = true;
  error.value = null;
  try {
    list.value = await api.sources({ enabled: filter.value === "all" ? undefined : filter.value === "enabled" });
    if (selected.value) {
      const refreshed = list.value.items.find((source) => source.id === selected.value?.id);
      if (refreshed) {
        selected.value = refreshed;
        if (!isCreating.value) form.value = formFromSource(refreshed);
      }
    }
  } catch (loadError) {
    error.value = loadError instanceof Error ? loadError.message : "无法加载来源列表";
  } finally {
    loading.value = false;
  }
}

onMounted(() => { void load(); });
watch(() => props.refreshToken, () => { void load(); });
watch(filter, () => { void load(); });

const filteredSources = computed(() => {
  const value = search.value.trim().toLowerCase();
  if (!value) return list.value?.items ?? [];
  return (list.value?.items ?? []).filter((source) => [source.name, source.type, source.url, source.platform_id, ...source.languages].filter(Boolean).join(" ").toLowerCase().includes(value));
});

function newSource() {
  selected.value = null;
  isCreating.value = true;
  form.value = emptyForm();
}

function editSource(source: Source) {
  selected.value = source;
  isCreating.value = false;
  form.value = formFromSource(source);
}

function closeEditor() {
  selected.value = null;
  isCreating.value = false;
  form.value = emptyForm();
}

async function save() {
  if (!form.value.name.trim()) {
    emit("toast", "请输入来源名称");
    return;
  }
  busy.value = true;
  const payload = {
    name: form.value.name.trim(),
    type: form.value.type,
    url: form.value.url.trim() || null,
    platform_id: form.value.platform_id.trim() || null,
    authority: Number(form.value.authority),
    enabled: form.value.enabled,
    languages: splitList(form.value.languages),
    industries: form.value.industries,
    metadata: selected.value?.metadata ?? {},
  };
  try {
    const saved = isCreating.value || !selected.value
      ? await api.createSource(payload)
      : await api.updateSource(selected.value.id, payload);
    selected.value = saved;
    isCreating.value = false;
    form.value = formFromSource(saved);
    await load();
    emit("toast", "来源信息已保存");
  } catch (saveError) {
    emit("toast", saveError instanceof Error ? saveError.message : "保存来源失败");
  } finally {
    busy.value = false;
  }
}

async function toggleEnabled(source: Source) {
  try {
    const updated = await api.updateSource(source.id, { enabled: !source.enabled });
    if (selected.value?.id === updated.id) {
      selected.value = updated;
      form.value = formFromSource(updated);
    }
    await load();
    emit("toast", updated.enabled ? "来源已启用" : "来源已停用");
  } catch (toggleError) {
    emit("toast", toggleError instanceof Error ? toggleError.message : "更新来源状态失败");
  }
}

async function removeSource() {
  if (!selected.value || !window.confirm(`确定删除来源“${selected.value.name}”吗？`)) return;
  busy.value = true;
  try {
    await api.deleteSource(selected.value.id);
    closeEditor();
    await load();
    emit("toast", "来源已删除");
  } catch (deleteError) {
    emit("toast", deleteError instanceof Error ? deleteError.message : "删除来源失败");
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <main class="page-content page-content-wide">
    <div class="page-heading">
      <div><div class="eyebrow">基础配置</div><h1>来源管理</h1><p>维护 YouTube、RSS 和其他内容来源的权威性与启用状态。</p></div>
      <button class="button button-primary" @click="newSource"><Plus :size="16" /> 新增来源</button>
    </div>

    <div v-if="error" class="empty-state error-state"><CircleAlert :size="20" /><div><strong>无法加载来源列表</strong><p>{{ error }}</p></div><button class="button button-secondary" @click="load"><RefreshCw :size="15" /> 重试</button></div>
    <div v-else class="resource-workspace">
      <section class="panel resource-list-panel">
        <div class="table-toolbar"><div class="search-field"><Search :size="16" /><input v-model="search" aria-label="搜索来源" placeholder="搜索名称、平台或地址" /></div><select v-model="filter" aria-label="筛选来源"><option value="all">全部来源</option><option value="enabled">仅看已启用</option><option value="disabled">仅看已停用</option></select><button class="icon-button" title="刷新来源" @click="load"><RefreshCw :size="16" /></button></div>
        <div class="resource-table-head"><span>来源</span><span>类型 / 地址</span><span>权威性</span><span>状态</span></div>
        <div v-if="loading" class="loading-list"><div v-for="n in 6" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="filteredSources.length === 0" class="empty-state"><Globe2 :size="22" /><strong>暂无来源</strong><p>新增来源后，它们会出现在这里。</p></div>
        <div v-else class="resource-list">
          <button v-for="source in filteredSources" :key="source.id" class="resource-row" :class="{ 'is-selected': selected?.id === source.id }" @click="editSource(source)">
            <div class="resource-primary"><strong>{{ source.name }}</strong><span>{{ source.platform_id || source.languages.join(', ') || "未设置平台信息" }}</span></div>
            <div class="resource-secondary"><strong>{{ source.type }}</strong><span>{{ source.url || "未设置地址" }}</span></div>
            <span class="resource-score">{{ percent(source.authority) }}</span>
            <span class="status-badge" :class="source.enabled ? 'positive' : 'negative'">{{ source.enabled ? "已启用" : "已停用" }}</span>
          </button>
        </div>
        <div v-if="list" class="table-footer"><span>{{ filteredSources.length }} / {{ list.total }} 个来源</span><span>已从 SQLite 更新</span></div>
      </section>

      <aside class="panel resource-detail">
        <template v-if="isCreating || selected">
          <div class="detail-header"><button class="icon-button" title="关闭编辑" @click="closeEditor"><X :size="17" /></button><span class="eyebrow">{{ isCreating ? "新增来源" : "编辑来源" }}</span><button v-if="selected && !isCreating" class="icon-button icon-button-danger" title="删除来源" @click="removeSource"><Trash2 :size="16" /></button></div>
          <form class="detail-content resource-form" @submit.prevent="save">
            <div class="form-field"><label for="source-name">来源名称 <span>*</span></label><input id="source-name" v-model="form.name" required placeholder="例如：OpenAI YouTube Channel" /></div>
            <div class="form-grid"><div class="form-field"><label for="source-type">来源类型</label><select id="source-type" v-model="form.type"><option v-for="[value, label] in sourceTypes" :key="value" :value="value">{{ label }}</option></select></div><div class="form-field"><label for="source-platform">平台 ID</label><input id="source-platform" v-model="form.platform_id" placeholder="频道或节目 ID" /></div></div>
            <div class="form-field"><label for="source-url">来源地址</label><input id="source-url" v-model="form.url" type="url" placeholder="https://..." /></div>
            <div class="form-field"><div class="form-label-row"><label for="source-authority">权威性</label><strong>{{ percent(form.authority) }}</strong></div><input id="source-authority" v-model.number="form.authority" type="range" min="0" max="1" step="0.05" /></div>
            <div class="form-field"><label for="source-languages">语言</label><input id="source-languages" v-model="form.languages" placeholder="例如：en, zh，用逗号分隔" /></div>
            <div class="form-field"><label for="source-industries">关注行业</label><select id="source-industries" v-model="form.industries" multiple size="4"><option v-for="[value, label] in industryOptions" :key="value" :value="value">{{ label }}</option></select></div>
            <label class="check-field"><input v-model="form.enabled" type="checkbox" /><span>启用该来源的内容发现</span></label>
            <div class="form-actions"><button class="button button-primary" type="submit" :disabled="busy"><Save :size="15" /> 保存来源</button><button class="button button-secondary" type="button" @click="closeEditor">取消</button><a v-if="selected?.url" class="button button-secondary" :href="selected.url" target="_blank" rel="noreferrer"><ExternalLink :size="15" /> 打开地址</a></div>
          </form>
        </template>
        <div v-else class="empty-state"><Globe2 :size="22" /><strong>请选择来源</strong><p>选择来源后，可以编辑配置或调整监控状态。</p></div>
      </aside>
    </div>
  </main>
</template>
