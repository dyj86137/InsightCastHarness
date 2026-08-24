const labels: Record<string, string> = {
  accept: "接受",
  accepted: "已接受",
  archive: "归档",
  archived: "已归档",
  bilibili: "Bilibili",
  dailymotion: "Dailymotion",
  classified: "已分类",
  classification: "分类",
  completed: "已完成",
  commentary: "评论内容",
  discovered: "已发现",
  discovery: "发现",
  duplicate: "重复",
  failed: "失败",
  finished: "已完成",
  interview: "访谈",
  new: "新建",
  news_clip: "新闻片段",
  podcast_interview: "播客访谈",
  podcast_rss: "播客 RSS",
  peertube: "PeerTube",
  pushed: "已推送",
  push_ready: "待推送",
  queued: "排队中",
  ranking: "排序",
  ready: "已就绪",
  reject: "拒绝",
  rejected: "已拒绝",
  running: "运行中",
  sent: "已发送",
  short_clip: "短片",
  speech: "演讲",
  summary_ready: "摘要已生成",
  summarization: "摘要生成",
  transcript_pending: "等待字幕",
  transcript_ready: "字幕已就绪",
  transcript_fetch: "获取字幕",
  twitch: "Twitch",
  unavailable: "不可用",
  pending: "处理中",
  partial: "部分内容",
  unknown: "未知",
  video: "视频",
  vimeo: "Vimeo",
  youtube: "YouTube",
};

export function humanStatus(value?: string | null): string {
  if (!value) return "未知";
  if (labels[value]) return labels[value];
  return value.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

export function statusTone(value?: string | null): string {
  if (!value) return "neutral";
  if (["accepted", "ready", "pushed", "finished", "completed", "enabled", "sent"].includes(value)) return "positive";
  if (["rejected", "failed", "duplicate", "disabled"].includes(value)) return "negative";
  if (["classified", "running", "queued", "summary_ready", "transcript_pending", "pending"].includes(value)) return "warning";
  return "neutral";
}

export function formatDate(value?: string | null): string {
  if (!value) return "暂无日期";
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" }).format(new Date(value));
}

export function formatDateTime(value?: string | null): string {
  if (!value) return "尚未开始";
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

export function formatDuration(seconds?: number | null): string {
  if (seconds === undefined || seconds === null) return "--";
  const minutes = Math.floor(seconds / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分钟` : `${minutes} 分钟`;
}

export function splitList(value: string): string[] {
  return value.split(/[，,\n]/).map((item) => item.trim()).filter(Boolean);
}

export function prettyJson(value: unknown): string {
  if (value === undefined || value === null) return "暂无数据";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function percent(value?: number | null): string {
  return value === undefined || value === null ? "--" : `${Math.round(value * 100)}%`;
}
