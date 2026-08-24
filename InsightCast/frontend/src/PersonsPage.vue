<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Check, CircleAlert, Plus, RefreshCw, Save, Search, Trash2, UserRound, X } from "@lucide/vue";
import { api } from "./api";
import type { Paginated, Person } from "./types";
import { formatDate, percent, splitList } from "./display";

const props = defineProps<{ refreshToken: number }>();
const emit = defineEmits<{ toast: [message: string] }>();

const industryOptions = [
  ["ai", "人工智能"], ["ecommerce", "电子商务"], ["retail", "零售"],
  ["consumer", "消费"], ["finance", "金融"], ["cloud", "云计算"],
  ["enterprise_software", "企业软件"], ["media", "媒体"], ["healthcare", "医疗健康"],
  ["auto", "汽车"], ["energy", "能源"], ["other", "其他"],
];

interface PersonForm {
  name: string;
  display_name: string;
  aliases: string;
  companies: string;
  title: string;
  industries: string[];
  importance: number;
  enabled: boolean;
  notes: string;
}

function emptyForm(): PersonForm {
  return { name: "", display_name: "", aliases: "", companies: "", title: "", industries: [], importance: 0.5, enabled: true, notes: "" };
}

function formFromPerson(person: Person): PersonForm {
  return {
    name: person.name,
    display_name: person.display_name ?? "",
    aliases: person.aliases.join(", "),
    companies: person.companies.join(", "),
    title: person.title ?? "",
    industries: [...person.industries],
    importance: person.importance,
    enabled: person.enabled,
    notes: person.notes ?? "",
  };
}

const list = ref<Paginated<Person> | null>(null);
const selected = ref<Person | null>(null);
const form = ref<PersonForm>(emptyForm());
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
    list.value = await api.persons({ enabled: filter.value === "all" ? undefined : filter.value === "enabled" });
    if (selected.value) {
      const refreshed = list.value.items.find((person) => person.id === selected.value?.id);
      if (refreshed) {
        selected.value = refreshed;
        if (!isCreating.value) form.value = formFromPerson(refreshed);
      }
    }
  } catch (loadError) {
    error.value = loadError instanceof Error ? loadError.message : "无法加载人物列表";
  } finally {
    loading.value = false;
  }
}

onMounted(() => { void load(); });
watch(() => props.refreshToken, () => { void load(); });
watch(filter, () => { void load(); });

const filteredPeople = computed(() => {
  const value = search.value.trim().toLowerCase();
  if (!value) return list.value?.items ?? [];
  return (list.value?.items ?? []).filter((person) => [
    person.name, person.display_name, person.title, ...person.aliases, ...person.companies,
  ].filter(Boolean).join(" ").toLowerCase().includes(value));
});

function newPerson() {
  selected.value = null;
  isCreating.value = true;
  form.value = emptyForm();
}

function editPerson(person: Person) {
  selected.value = person;
  isCreating.value = false;
  form.value = formFromPerson(person);
}

function closeEditor() {
  selected.value = null;
  isCreating.value = false;
  form.value = emptyForm();
}

async function save() {
  if (!form.value.name.trim()) {
    emit("toast", "请输入人物姓名");
    return;
  }
  busy.value = true;
  const payload = {
    name: form.value.name.trim(),
    display_name: form.value.display_name.trim() || null,
    aliases: splitList(form.value.aliases),
    companies: splitList(form.value.companies),
    title: form.value.title.trim() || null,
    industries: form.value.industries,
    importance: Number(form.value.importance),
    enabled: form.value.enabled,
    notes: form.value.notes.trim() || null,
    metadata: selected.value?.metadata ?? {},
  };
  try {
    const saved = isCreating.value || !selected.value
      ? await api.createPerson(payload)
      : await api.updatePerson(selected.value.id, payload);
    selected.value = saved;
    isCreating.value = false;
    form.value = formFromPerson(saved);
    await load();
    emit("toast", "人物信息已保存");
  } catch (saveError) {
    emit("toast", saveError instanceof Error ? saveError.message : "保存人物失败");
  } finally {
    busy.value = false;
  }
}

async function toggleEnabled(person: Person) {
  try {
    const updated = await api.updatePerson(person.id, { enabled: !person.enabled });
    if (selected.value?.id === updated.id) {
      selected.value = updated;
      form.value = formFromPerson(updated);
    }
    await load();
    emit("toast", updated.enabled ? "人物已启用" : "人物已停用");
  } catch (toggleError) {
    emit("toast", toggleError instanceof Error ? toggleError.message : "更新人物状态失败");
  }
}

async function removePerson() {
  if (!selected.value || !window.confirm(`确定删除人物“${selected.value.name}”吗？`)) return;
  busy.value = true;
  try {
    await api.deletePerson(selected.value.id);
    closeEditor();
    await load();
    emit("toast", "人物已删除");
  } catch (deleteError) {
    emit("toast", deleteError instanceof Error ? deleteError.message : "删除人物失败");
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <main class="page-content page-content-wide">
    <div class="page-heading">
      <div><div class="eyebrow">基础配置</div><h1>人物库管理</h1><p>维护关注人物、别名、所属公司和重要性权重。</p></div>
      <button class="button button-primary" @click="newPerson"><Plus :size="16" /> 新增人物</button>
    </div>

    <div v-if="error" class="empty-state error-state"><CircleAlert :size="20" /><div><strong>无法加载人物库</strong><p>{{ error }}</p></div><button class="button button-secondary" @click="load"><RefreshCw :size="15" /> 重试</button></div>
    <div v-else class="resource-workspace">
      <section class="panel resource-list-panel">
        <div class="table-toolbar"><div class="search-field"><Search :size="16" /><input v-model="search" aria-label="搜索人物" placeholder="搜索姓名、公司或别名" /></div><select v-model="filter" aria-label="筛选人物"><option value="all">全部人物</option><option value="enabled">仅看已启用</option><option value="disabled">仅看已停用</option></select><button class="icon-button" title="刷新人物" @click="load"><RefreshCw :size="16" /></button></div>
        <div class="resource-table-head"><span>人物</span><span>公司 / 职位</span><span>重要性</span><span>状态</span></div>
        <div v-if="loading" class="loading-list"><div v-for="n in 6" :key="n" class="loading-row"><span /><span /><span /></div></div>
        <div v-else-if="filteredPeople.length === 0" class="empty-state"><UserRound :size="22" /><strong>暂无人物</strong><p>新增人物后，他们会出现在这里。</p></div>
        <div v-else class="resource-list">
          <button v-for="person in filteredPeople" :key="person.id" class="resource-row" :class="{ 'is-selected': selected?.id === person.id }" @click="editPerson(person)">
            <div class="resource-primary"><strong>{{ person.display_name || person.name }}</strong><span>{{ person.name }}<template v-if="person.aliases.length"> / {{ person.aliases.join(', ') }}</template></span></div>
            <div class="resource-secondary"><strong>{{ person.companies.join(', ') || "--" }}</strong><span>{{ person.title || "未填写职位" }}</span></div>
            <span class="resource-score">{{ percent(person.importance) }}</span>
            <span class="status-badge" :class="person.enabled ? 'positive' : 'negative'">{{ person.enabled ? "已启用" : "已停用" }}</span>
          </button>
        </div>
        <div v-if="list" class="table-footer"><span>{{ filteredPeople.length }} / {{ list.total }} 位人物</span><span>已从 SQLite 更新</span></div>
      </section>

      <aside class="panel resource-detail">
        <template v-if="isCreating || selected">
          <div class="detail-header"><button class="icon-button" title="关闭编辑" @click="closeEditor"><X :size="17" /></button><span class="eyebrow">{{ isCreating ? "新增人物" : "编辑人物" }}</span><button v-if="selected && !isCreating" class="icon-button icon-button-danger" title="删除人物" @click="removePerson"><Trash2 :size="16" /></button></div>
          <form class="detail-content resource-form" @submit.prevent="save">
            <div class="form-field"><label for="person-name">姓名 <span>*</span></label><input id="person-name" v-model="form.name" required placeholder="例如：Sam Altman" /></div>
            <div class="form-grid"><div class="form-field"><label for="person-display">显示名称</label><input id="person-display" v-model="form.display_name" placeholder="日报中显示的名称" /></div><div class="form-field"><label for="person-title">职位</label><input id="person-title" v-model="form.title" placeholder="例如：CEO" /></div></div>
            <div class="form-field"><label for="person-companies">公司</label><input id="person-companies" v-model="form.companies" placeholder="多个值用逗号分隔" /></div>
            <div class="form-field"><label for="person-aliases">别名</label><input id="person-aliases" v-model="form.aliases" placeholder="英文名或其他写法，用逗号分隔" /></div>
            <div class="form-field"><label for="person-industries">关注行业</label><select id="person-industries" v-model="form.industries" multiple size="4"><option v-for="[value, label] in industryOptions" :key="value" :value="value">{{ label }}</option></select></div>
            <div class="form-field"><div class="form-label-row"><label for="person-importance">重要性</label><strong>{{ percent(form.importance) }}</strong></div><input id="person-importance" v-model.number="form.importance" type="range" min="0" max="1" step="0.05" /></div>
            <label class="check-field"><input v-model="form.enabled" type="checkbox" /><span>启用该人物的发现和监控</span></label>
            <div class="form-field"><label for="person-notes">备注</label><textarea id="person-notes" v-model="form.notes" rows="4" placeholder="补充人物背景或筛选要求" /></div>
            <div class="form-actions"><button class="button button-primary" type="submit" :disabled="busy"><Save :size="15" /> 保存人物</button><button class="button button-secondary" type="button" @click="closeEditor">取消</button></div>
          </form>
        </template>
        <div v-else class="empty-state"><UserRound :size="22" /><strong>请选择人物</strong><p>选择人物后，可以编辑配置或调整监控状态。</p></div>
      </aside>
    </div>
  </main>
</template>
