"use client";

import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  CircleSlash,
  Clock,
  Loader2,
  type LucideIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";

type Tone = "ok" | "warn" | "bad" | "info" | "muted";

const TONES: Record<Tone, string> = {
  ok: "text-ok bg-ok/10",
  warn: "text-warn bg-warn/10",
  bad: "text-bad bg-bad/10",
  info: "text-info bg-info/10",
  muted: "text-muted-foreground bg-muted",
};

/** A status is always icon + words + colour — never colour alone. */
export function StatusBadge({ tone, icon: Icon, label, spin, className }: {
  tone: Tone; icon: LucideIcon; label: string; spin?: boolean; className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap", TONES[tone], className)}>
      <Icon className={cn("size-3.5", spin && "animate-spin")} aria-hidden />
      {label}
    </span>
  );
}

export function ConnectionStatusBadge({ status }: { status: string }) {
  switch (status) {
    case "connected": return <StatusBadge tone="ok" icon={CheckCircle2} label="已连接" />;
    case "testing": return <StatusBadge tone="info" icon={Loader2} spin label="测试中" />;
    case "error": return <StatusBadge tone="bad" icon={AlertCircle} label="连接失败" />;
    case "disabled": return <StatusBadge tone="muted" icon={CircleSlash} label="已停用" />;
    default: return <StatusBadge tone="muted" icon={CircleDashed} label="未测试" />;
  }
}

export function VoicePackStatusBadge({ status }: { status: string }) {
  switch (status) {
    case "ready": return <StatusBadge tone="ok" icon={CheckCircle2} label="可以训练" />;
    case "needs_review": return <StatusBadge tone="warn" icon={Clock} label="待审核" />;
    case "processing": return <StatusBadge tone="info" icon={Loader2} spin label="处理中" />;
    case "uploading": return <StatusBadge tone="muted" icon={CircleDashed} label="待处理" />;
    case "error": return <StatusBadge tone="bad" icon={AlertCircle} label="处理失败" />;
    default: return <StatusBadge tone="muted" icon={CircleDashed} label="未上传" />;
  }
}

export function TrainingStatusBadge({ status }: { status: string }) {
  switch (status) {
    case "completed": return <StatusBadge tone="ok" icon={CheckCircle2} label="已完成" />;
    case "failed": return <StatusBadge tone="bad" icon={AlertCircle} label="失败" />;
    case "cancelled": return <StatusBadge tone="muted" icon={CircleSlash} label="已取消" />;
    case "cancel_requested": return <StatusBadge tone="warn" icon={Loader2} spin label="正在取消" />;
    case "queued": return <StatusBadge tone="muted" icon={Clock} label="排队中" />;
    default: return <StatusBadge tone="info" icon={Loader2} spin label="进行中" />;
  }
}

export function CharacterStatusBadge({ status }: { status: string }) {
  return status === "ready"
    ? <StatusBadge tone="ok" icon={CheckCircle2} label="可以对话" />
    : <StatusBadge tone="warn" icon={AlertCircle} label="配置不完整" />;
}
