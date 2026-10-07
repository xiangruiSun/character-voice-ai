"use client";

import { useQueryClient } from "@tanstack/react-query";
import { ChevronDown, Loader2, RotateCcw, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { ErrorAlert } from "@/components/shared/blocks";
import { TrainingStatusBadge } from "@/components/shared/status";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { errorText } from "@/lib/api";
import { formatDuration, formatElapsed } from "@/lib/format";
import { keys, useTrainingMutations, useTrainingOptions } from "@/lib/queries";
import type { TrainingJob, VoicePack } from "@/lib/types";
import { cn } from "@/lib/utils";

const TERMINAL = ["completed", "failed", "cancelled"];
const EVENT_LABEL: Record<string, string> = {
  job_queued: "已加入队列", dataset_validating: "开始检查数据", dataset_preparing: "开始准备数据",
  training_started: "开始训练", epoch_completed: "完成一轮训练", checkpoint_saved: "保存模型",
  evaluation_started: "开始评估", completed: "训练完成", failed: "训练失败",
  cancel_requested: "请求取消", cancelled: "已取消",
};

/** Live job state over Server-Sent Events; TanStack's cache is the single copy. */
export function useJobStream(jobId: string | null, initial?: TrainingJob) {
  const client = useQueryClient();
  const [streamed, setJob] = useState<TrainingJob | undefined>();
  const job = streamed && streamed.id === jobId ? streamed : initial;
  // Depend on whether the job is finished, not on `initial` itself: the job list refetches
  // every few seconds, and a new object each time would reopen the stream each time.
  const startedFinished = !!initial && TERMINAL.includes(initial.status);
  useEffect(() => {
    if (!jobId || startedFinished) return;
    const source = new EventSource(`/api/training-jobs/${jobId}/events`);
    source.addEventListener("job", (event) => {
      const next = JSON.parse((event as MessageEvent).data) as TrainingJob;
      setJob(next);
      client.setQueryData(keys.job(jobId), next);
    });
    source.addEventListener("end", () => source.close());
    return () => source.close();
  }, [jobId, startedFinished, client]);

  // However the end is learned (stream or list refetch), the voice model it produced and
  // the pack's state must be re-read, or the workflow cannot move on to the test step.
  const finished = !!job && TERMINAL.includes(job.status);
  const wasActive = useRef(!startedFinished);
  useEffect(() => {
    if (!finished || !wasActive.current) return;
    wasActive.current = false;
    void client.invalidateQueries({ queryKey: keys.jobs });
    void client.invalidateQueries({ queryKey: keys.voices });
    void client.invalidateQueries({ queryKey: keys.packs });
  }, [finished, client]);
  return job;
}

export function TrainingSetup({ pack }: { pack: VoicePack }) {
  const options = useTrainingOptions();
  const m = useTrainingMutations();
  const engine = options.data?.[0];
  const [preset, setPreset] = useState("balanced");
  const [advanced, setAdvanced] = useState<Record<string, number>>({});
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  if (!engine) return <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" aria-hidden />正在读取训练选项…</p>;
  const params = { ...(engine.presets.find((p) => p.key === preset)?.params ?? {}), ...advanced };

  async function start() {
    setError(null);
    try {
      await m.start.mutateAsync({ voice_pack_id: pack.id, preset, advanced });
      toast.success("训练已加入队列", { description: "可以离开这个页面，训练会在后台继续。" });
    } catch (e) { setError(errorText(e)); }
  }

  return (
    <div className="space-y-6">
      {!engine.available && <ErrorAlert title="训练引擎不可用" hint={engine.unavailable_reason} />}
      <div>
        <p className="mb-3 text-sm font-medium">训练方案</p>
        <div className="grid gap-3 md:grid-cols-3" role="radiogroup" aria-label="训练方案">
          {engine.presets.map((p) => (
            <button key={p.key} type="button" role="radio" aria-checked={preset === p.key}
                    onClick={() => { setPreset(p.key); setAdvanced({}); }}
                    className={cn("rounded-xl border p-4 text-left transition-colors focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
                                  preset === p.key ? "border-primary bg-primary/5" : "hover:bg-muted/50")}>
              <span className="block font-medium">{p.label}</span>
              <span className="mt-1 block text-sm text-muted-foreground">{p.description}</span>
            </button>
          ))}
        </div>
      </div>

      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ChevronDown className="size-4" aria-hidden />高级设置
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4 grid gap-4 md:grid-cols-3">
          {engine.advanced.map((o) => (
            <div key={o.key} className="grid gap-1.5">
              <Label htmlFor={`adv-${o.key}`}>{o.label}</Label>
              <Input id={`adv-${o.key}`} type="number" min={o.minimum} max={o.maximum}
                     step={o.kind === "int" ? 1 : 0.0001} value={params[o.key] ?? o.default}
                     onChange={(e) => setAdvanced((a) => ({ ...a, [o.key]: Number(e.target.value) }))} />
              <span className="text-xs text-muted-foreground">{o.hint}</span>
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm text-muted-foreground">
        <span>引擎：{engine.label}（{engine.base_model}）</span>
        <span>数据：{formatDuration(pack.usable_duration_s)}</span>
      </div>
      {error && <ErrorAlert title={error.message} hint={error.hint} />}
      <Button size="lg" onClick={start} disabled={m.start.isPending || !engine.available}>
        {m.start.isPending && <Loader2 className="animate-spin" aria-hidden />}开始训练
      </Button>
    </div>
  );
}

export function TrainingProgress({ initial, pack }: { initial: TrainingJob; pack?: VoicePack }) {
  const job = useJobStream(initial.id, initial) ?? initial;
  const m = useTrainingMutations();
  const [confirm, setConfirm] = useState(false);
  const [, tick] = useState(0);
  useEffect(() => {
    if (TERMINAL.includes(job.status)) return;
    const timer = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, [job.status]);

  const percent = Math.round(job.progress * 100);
  const active = !TERMINAL.includes(job.status);
  const losses = Object.entries(job.metrics).slice(0, 6);

  async function act(action: () => Promise<unknown>, message: string) {
    try { await action(); toast.success(message); }
    catch (e) { const { message: msg, hint } = errorText(e); toast.error(msg, { description: hint ?? undefined }); }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold">训练 {pack?.character_name ?? job.voice_pack_name}</h3>
        <TrainingStatusBadge status={job.status} />
      </div>

      {job.status === "failed" ? (
        <div className="space-y-3">
          <ErrorAlert title={`训练已停止：${job.error_message ?? "未知原因"}`} hint={job.error_hint} />
          <Button onClick={() => act(() => m.retry.mutateAsync(job.id), "已重新加入队列")} disabled={m.retry.isPending}>
            <RotateCcw aria-hidden />重试
          </Button>
        </div>
      ) : (
        <div className="space-y-2" role="status" aria-live="polite">
          <Progress value={percent} aria-label="训练进度" />
          <div className="flex flex-wrap justify-between gap-2 text-sm">
            <span className="font-medium">{job.current_stage}</span>
            <span className="tabular-nums text-muted-foreground">{percent}%</span>
          </div>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
        <div><dt className="text-muted-foreground">轮次</dt><dd className="tabular-nums">{job.total_epochs ? `${job.current_epoch} / ${job.total_epochs}` : "—"}</dd></div>
        <div><dt className="text-muted-foreground">数据</dt><dd>{pack ? formatDuration(pack.usable_duration_s) : "—"}</dd></div>
        <div><dt className="text-muted-foreground">引擎</dt><dd>{job.engine === "gpt_sovits" ? "GPT-SoVITS" : job.engine}</dd></div>
        <div><dt className="text-muted-foreground">用时</dt><dd className="tabular-nums">{formatElapsed(job.started_at ?? job.created_at, job.completed_at)}</dd></div>
      </dl>

      {active && (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" onClick={() => setConfirm(true)} disabled={job.status === "cancel_requested"}>
            <Square aria-hidden />取消训练
          </Button>
          <span className="text-xs text-muted-foreground">可以离开这个页面，训练会在后台继续。</span>
        </div>
      )}
      {job.status === "cancelled" && (
        <Button variant="outline" onClick={() => act(() => m.retry.mutateAsync(job.id), "已重新加入队列")}><RotateCcw aria-hidden />重新开始</Button>
      )}

      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ChevronDown className="size-4" aria-hidden />训练详情
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-4 text-sm">
          <p className="text-muted-foreground">方案：{job.preset} · 参数：{Object.entries(job.config).map(([k, v]) => `${k}=${v}`).join("，")}</p>
          {losses.length > 0 && (
            <p className="font-mono text-xs">{losses.map(([k, v]) => `${k}: ${Number(v).toFixed(3)}`).join("   ")}</p>
          )}
          <ol className="max-h-56 space-y-1 overflow-y-auto font-mono text-xs text-muted-foreground">
            {[...job.events].reverse().map((e, i) => (
              <li key={i}>{new Date(e.at).toLocaleTimeString("zh-CN")}  {EVENT_LABEL[e.type] ?? e.type}
                {e.epoch ? ` ${e.epoch}/${e.total_epochs}` : ""}{typeof e.message === "string" ? ` · ${e.message}` : ""}</li>
            ))}
          </ol>
          {job.error_detail && (
            <pre className="max-h-64 overflow-auto rounded-lg bg-muted p-3 text-xs whitespace-pre-wrap">{job.error_detail}</pre>
          )}
        </CollapsibleContent>
      </Collapsible>

      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>取消训练？</DialogTitle>
            <DialogDescription>已经完成的训练进度会被丢弃。之后可以用同样的设置重新开始。</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />}>继续训练</DialogClose>
            <Button variant="destructive" onClick={() => { setConfirm(false); void act(() => m.cancel.mutateAsync(job.id), "正在取消训练"); }}>
              取消训练
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
