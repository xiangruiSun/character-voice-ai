export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return "0 秒";
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes.toFixed(minutes < 10 ? 1 : 0)} 分钟`;
  return `${(minutes / 60).toFixed(1)} 小时`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function formatElapsed(fromIso: string | null, toIso?: string | null): string {
  if (!fromIso) return "—";
  const end = toIso ? new Date(toIso).getTime() : Date.now();
  const seconds = Math.max(0, Math.round((end - new Date(fromIso).getTime()) / 1000));
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m ? `${m} 分 ${s} 秒` : `${s} 秒`;
}

export function formatRelative(iso: string | null): string {
  if (!iso) return "从未";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  return new Date(iso).toLocaleDateString("zh-CN");
}

export const STYLE_LABELS: Record<string, string> = {
  neutral: "平静", soft: "温柔", happy: "开心", sad: "难过", angry: "生气",
  serious: "严肃", excited: "激动", surprised: "惊讶", teasing: "调侃", whisper: "低语",
};

export function styleLabel(style: string): string {
  return STYLE_LABELS[style] ?? style;
}

export function prettyModel(model: string): string {
  const [family, size] = model.split(":");
  const name = family.charAt(0).toUpperCase() + family.slice(1);
  return size ? `${name} ${size.toUpperCase()}` : name;
}
