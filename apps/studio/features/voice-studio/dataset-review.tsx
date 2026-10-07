"use client";

import { AlertTriangle, ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { PlayButton, StatTile } from "@/components/shared/blocks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { errorText } from "@/lib/api";
import { formatDuration, STYLE_LABELS } from "@/lib/format";
import { useSamples, useVoicePackMutations } from "@/lib/queries";
import type { AudioSample, VoicePack } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE = 25;
const STYLE_ITEMS = Object.entries(STYLE_LABELS).map(([value, label]) => ({ value, label }));

export function DatasetSummaryTiles({ pack }: { pack: VoicePack }) {
  const s = pack.summary;
  if (!s) return null;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="上传" value={formatDuration(s.uploaded_s)} />
        <StatTile label="可用" value={formatDuration(s.usable_s)} hint={`${s.approved_clips} 个片段`} />
        <StatTile label="已排除" value={formatDuration(s.rejected_s)} />
        <StatTile label="片段总数" value={String(s.clips)} />
      </div>
      {s.potential_issues.length > 0 && (
        <div className="rounded-xl border p-4">
          <p className="mb-2 flex items-center gap-2 text-sm font-medium">
            <AlertTriangle className="size-4 text-warn" aria-hidden />可能存在的问题
            <span className="font-normal text-muted-foreground">（自动检测，仅供参考）</span>
          </p>
          <ul className="flex flex-wrap gap-2 text-sm">
            {s.potential_issues.map((i) => (
              <li key={i.issue} className="rounded-full bg-warn/10 px-2.5 py-0.5 text-warn">{i.count} 个片段{i.issue}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SampleRow({ sample, packId }: { sample: AudioSample; packId: string }) {
  const m = useVoicePackMutations(packId);
  const [text, setText] = useState(sample.transcript);
  const [local, setLocal] = useState(sample);

  async function save(patch: Partial<Pick<AudioSample, "transcript" | "style" | "approved">>) {
    const before = local;
    setLocal({ ...local, ...patch });                       // optimistic
    try {
      setLocal(await m.updateSample.mutateAsync({ id: sample.id, patch }));
    } catch (e) {
      setLocal(before);
      if (patch.transcript !== undefined) setText(before.transcript);
      const { message, hint } = errorText(e);
      toast.error(message, { description: hint ?? undefined });
    }
  }

  return (
    <li className={cn("grid gap-3 px-4 py-3 md:grid-cols-[auto_1fr_auto] md:items-center", !local.approved && "bg-muted/40")}>
      <div className="flex items-center gap-3">
        <PlayButton src={sample.audio_url} label={`播放：${sample.transcript || sample.source_file}`} />
        <span className="w-12 text-xs text-muted-foreground tabular-nums">{sample.duration_s.toFixed(1)} 秒</span>
      </div>
      <div className="min-w-0 space-y-1">
        <Input value={text} onChange={(e) => setText(e.target.value)} aria-label="这段语音的文字"
               placeholder="（没有识别出文字，请填写）"
               onBlur={() => { if (text.trim() !== local.transcript) void save({ transcript: text.trim() }); }}
               onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
               className={cn(!local.approved && "text-muted-foreground")} />
        <div className="flex flex-wrap gap-1.5 text-xs text-muted-foreground">
          <span className="truncate">{sample.source_file}</span>
          {local.transcript_source === "provided" && <span>· 自带文字稿</span>}
          {local.transcript_source === "human" && <span>· 已人工修改</span>}
          {sample.issues.map((issue) => <span key={issue} className="text-warn">· 可能{issue.replace(/^可能/, "")}</span>)}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Select items={STYLE_ITEMS} value={local.style} onValueChange={(v) => v && void save({ style: v })}>
          <SelectTrigger size="sm" className="w-24" aria-label="语气风格"><SelectValue /></SelectTrigger>
          <SelectContent>{STYLE_ITEMS.map((s) => <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>)}</SelectContent>
        </Select>
        <label className="flex items-center gap-2 text-sm">
          <Switch checked={local.approved} onCheckedChange={(v) => void save({ approved: v })}
                  aria-label={local.approved ? "已纳入训练，点击排除" : "已排除，点击纳入训练"} />
          <span className="w-8">{local.approved ? "纳入" : "排除"}</span>
        </label>
      </div>
    </li>
  );
}

export function DatasetReview({ pack }: { pack: VoicePack }) {
  const samples = useSamples(pack.id, pack.status !== "processing");
  const [filter, setFilter] = useState("all");
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const list = samples.data ?? [];
    if (filter === "approved") return list.filter((s) => s.approved);
    if (filter === "excluded") return list.filter((s) => !s.approved);
    if (filter === "issues") return list.filter((s) => s.issues.length > 0);
    return list;
  }, [samples.data, filter]);
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  const current = Math.min(page, pages - 1);

  if (samples.isPending) return <div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-16" />)}</div>;
  const counts = {
    all: samples.data?.length ?? 0,
    approved: samples.data?.filter((s) => s.approved).length ?? 0,
    excluded: samples.data?.filter((s) => !s.approved).length ?? 0,
    issues: samples.data?.filter((s) => s.issues.length > 0).length ?? 0,
  };

  return (
    <div className="space-y-3">
      <Tabs value={filter} onValueChange={(v) => { setFilter(String(v)); setPage(0); }}>
        <TabsList>
          <TabsTrigger value="all">全部 {counts.all}</TabsTrigger>
          <TabsTrigger value="approved">已纳入 {counts.approved}</TabsTrigger>
          <TabsTrigger value="excluded">已排除 {counts.excluded}</TabsTrigger>
          <TabsTrigger value="issues">可能有问题 {counts.issues}</TabsTrigger>
        </TabsList>
      </Tabs>
      {filtered.length === 0 ? (
        <p className="rounded-xl border px-4 py-10 text-center text-sm text-muted-foreground">这里没有片段。</p>
      ) : (
        <ul className="divide-y rounded-xl border" aria-label="语音片段">
          {filtered.slice(current * PAGE, (current + 1) * PAGE).map((s) => <SampleRow key={s.id} sample={s} packId={pack.id} />)}
        </ul>
      )}
      {pages > 1 && (
        <div className="flex items-center justify-end gap-2 text-sm">
          <Button size="icon-sm" variant="outline" disabled={current === 0} onClick={() => setPage(current - 1)} aria-label="上一页"><ChevronLeft aria-hidden /></Button>
          <span className="tabular-nums text-muted-foreground">{current + 1} / {pages}</span>
          <Button size="icon-sm" variant="outline" disabled={current >= pages - 1} onClick={() => setPage(current + 1)} aria-label="下一页"><ChevronRight aria-hidden /></Button>
        </div>
      )}
    </div>
  );
}
