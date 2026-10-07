"use client";

import { ArrowLeft, ArrowRight, Loader2, Users } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { toast } from "sonner";

import { PageContainer } from "@/components/layout/app-shell";
import { ErrorAlert, PageHeader, Stepper, LinkButton } from "@/components/shared/blocks";
import { VoicePackStatusBadge } from "@/components/shared/status";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { DatasetReview, DatasetSummaryTiles } from "@/features/voice-studio/dataset-review";
import { TrainingProgress, TrainingSetup } from "@/features/voice-studio/training";
import { Uploader } from "@/features/voice-studio/uploader";
import { VoiceTester } from "@/features/voice-studio/voice-tester";
import { errorText } from "@/lib/api";
import { useTrainingJobs, useVoiceModels, useVoicePack, useVoicePackMutations } from "@/lib/queries";
import { ACTIVE, STEPS, stepFor } from "@/features/voice-studio/workflow";

function Async({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<Skeleton className="h-96 rounded-xl" />}>{children}</Suspense>;
}

function Workflow() {
  const id = useSearchParams().get("id");
  const pack = useVoicePack(id);
  const jobs = useTrainingJobs(id ?? undefined);
  const models = useVoiceModels();
  const m = useVoicePackMutations(id ?? undefined);
  // A manual choice holds only while the server-derived step stays where it was chosen.
  const [manual, setManual] = useState<{ step: number; at: number } | null>(null);
  const [rights, setRights] = useState(false);

  const job = jobs.data?.[0];
  const model = useMemo(() => {
    const own = models.data?.filter((v) => v.voice_pack_id === id) ?? [];
    return own.find((v) => v.id === job?.voice_model_id) ?? own[0];
  }, [models.data, id, job]);
  const derived = pack.data ? stepFor(pack.data, job, model) : 0;
  const current = manual && manual.at === derived ? manual.step : derived;
  const setStep = (step: number) => setManual({ step, at: derived });

  if (!id) return <ErrorAlert title="没有指定声音包" />;
  if (pack.isPending) return <Skeleton className="h-96 rounded-xl" />;
  if (pack.isError) return <ErrorAlert title={errorText(pack.error).message} hint={errorText(pack.error).hint} onRetry={() => pack.refetch()} />;
  const p = pack.data;

  async function run(action: () => Promise<unknown>, success: string) {
    try { await action(); toast.success(success); }
    catch (e) { const { message, hint } = errorText(e); toast.error(message, { description: hint ?? undefined }); }
  }

  const reachable = (index: number) => index <= derived;

  return (
    <>
      <PageHeader title={p.name} description={`角色：${p.character_name} · 中文`} actions={<VoicePackStatusBadge status={p.status} />} />
      <Stepper steps={STEPS.map((label, i) => ({ key: label, label, disabled: !reachable(i) }))}
               current={current} onSelect={(i) => setStep(i)} />

      {current === 0 && (
        <section className="space-y-6" aria-label="上传">
          <Uploader pack={p} onUploaded={() => pack.refetch()} />
          <div className="flex justify-end">
            <Button onClick={() => run(() => m.prepare.mutateAsync(p.id), "开始处理语音")}
                    disabled={p.total_files === 0 || m.prepare.isPending}>
              {m.prepare.isPending && <Loader2 className="animate-spin" aria-hidden />}开始处理<ArrowRight aria-hidden />
            </Button>
          </div>
        </section>
      )}

      {current === 1 && (
        <section className="space-y-6" aria-label="检查">
          {p.status === "processing" ? (
            <div className="space-y-3 rounded-xl border p-6" role="status" aria-live="polite">
              <p className="flex items-center gap-2 font-medium"><Loader2 className="size-4 animate-spin" aria-hidden />{p.progress.label ?? "正在处理"}…</p>
              <Progress value={null} aria-label="处理进度" />
              <p className="text-sm text-muted-foreground">正在检查格式、整理片段、识别文字并分析音质。可以离开这个页面，处理会在后台继续。</p>
            </div>
          ) : p.status === "error" ? (
            <ErrorAlert title="处理语音时出错" hint={p.last_error} retryLabel="重新处理"
                        onRetry={() => run(() => m.prepare.mutateAsync(p.id), "重新开始处理")} />
          ) : (
            <>
              <DatasetSummaryTiles pack={p} />
              <DatasetReview pack={p} />
              <div className="flex flex-wrap justify-between gap-2">
                <Button variant="ghost" onClick={() => setStep(0)}><ArrowLeft aria-hidden />继续上传</Button>
                <Button onClick={() => setStep(2)}>下一步<ArrowRight aria-hidden /></Button>
              </div>
            </>
          )}
        </section>
      )}

      {current === 2 && (
        <section className="space-y-6" aria-label="准备">
          <DatasetSummaryTiles pack={p} />
          {p.summary && !p.summary.ready_to_train && (
            <ErrorAlert title="可用数据还不够"
                        hint={`至少需要 ${p.summary.requirements.min_clips} 个片段、${p.summary.requirements.min_seconds} 秒可用语音。可以上传更多，或重新纳入被排除的片段。`} />
          )}
          <label className="flex items-start gap-3 rounded-xl border p-4 text-sm">
            <Checkbox checked={p.rights_confirmed || rights} disabled={p.rights_confirmed}
                      onCheckedChange={(v) => setRights(!!v)} aria-label="确认有权使用这些音频" />
            <span>我确认自己拥有或已获得使用这些音频进行声音训练的相应权利或许可。</span>
          </label>
          <div className="flex flex-wrap justify-between gap-2">
            <Button variant="ghost" onClick={() => setStep(1)}><ArrowLeft aria-hidden />返回检查</Button>
            <Button disabled={!(p.rights_confirmed || rights) || !p.summary?.ready_to_train || m.approve.isPending}
                    onClick={() => run(async () => {
                      if (!p.rights_confirmed) await m.confirmRights.mutateAsync(p.id);
                      if (p.status !== "ready") await m.approve.mutateAsync(p.id);
                      setStep(3);
                    }, "数据集已确认")}>
              {m.approve.isPending && <Loader2 className="animate-spin" aria-hidden />}确认数据集<ArrowRight aria-hidden />
            </Button>
          </div>
        </section>
      )}

      {current === 3 && (
        <section aria-label="训练">
          {job && (ACTIVE.includes(job.status) || job.status === "failed")
            ? <TrainingProgress initial={job} pack={p} />
            : <TrainingSetup pack={p} />}
        </section>
      )}

      {current === 4 && model && (
        <section className="space-y-6" aria-label="试听">
          <div>
            <h3 className="text-lg font-semibold">角色声音已就绪</h3>
            <p className="text-sm text-muted-foreground">{model.name}：多试几句不同语气的话，再决定是否通过。</p>
          </div>
          <VoiceTester model={model} />
          {model.approved && <div className="flex justify-end"><Button onClick={() => setStep(5)}>下一步<ArrowRight aria-hidden /></Button></div>}
        </section>
      )}

      {current === 5 && model && (
        <section className="flex flex-col items-center gap-4 rounded-xl border px-6 py-12 text-center" aria-label="创建角色">
          <Users className="size-8 text-muted-foreground" aria-hidden />
          <div>
            <p className="text-lg font-semibold">用「{model.name}」创建角色</p>
            <p className="text-sm text-muted-foreground">选择对话模型、写下性格设定，就可以和 {p.character_name} 对话了。</p>
          </div>
          <LinkButton size="lg" href={`/characters?new=1&voice=${model.id}&name=${encodeURIComponent(p.character_name)}`}>
            创建角色<ArrowRight aria-hidden />
          </LinkButton>
        </section>
      )}
    </>
  );
}

export default function VoicePackPage() {
  return (
    <PageContainer wide>
      <LinkButton href="/voice-studio" variant="ghost" size="sm" className="mb-4 -ml-2">
        <ArrowLeft aria-hidden />声音工坊
      </LinkButton>
      <Async><Workflow /></Async>
    </PageContainer>
  );
}
