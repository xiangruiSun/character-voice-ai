"use client";

import { AudioLines, Mic2, Plus, Timer } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { PageContainer } from "@/components/layout/app-shell";
import { EmptyState, ErrorAlert, LinkButton, PageHeader } from "@/components/shared/blocks";
import { StatusBadge, TrainingStatusBadge, VoicePackStatusBadge } from "@/components/shared/status";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CreatePackDialog } from "@/features/voice-studio/create-pack-dialog";
import { VoiceTester } from "@/features/voice-studio/voice-tester";
import { errorText } from "@/lib/api";
import { formatDuration, formatRelative, styleLabel } from "@/lib/format";
import { useTrainingJobs, useVoiceModels, useVoicePacks } from "@/lib/queries";
import type { VoiceModel } from "@/lib/types";
import { CheckCircle2 } from "lucide-react";

function Loading() {
  return <div className="grid gap-4 md:grid-cols-2">{[0, 1].map((i) => <Skeleton key={i} className="h-36 rounded-xl" />)}</div>;
}

function PacksTab({ onCreate }: { onCreate: () => void }) {
  const packs = useVoicePacks();
  if (packs.isPending) return <Loading />;
  if (packs.isError) return <ErrorAlert title={errorText(packs.error).message} onRetry={() => packs.refetch()} />;
  if (!packs.data.length) {
    return <EmptyState icon={AudioLines} title="还没有声音包" description="上传一个角色的中文语音，训练出属于这个角色的声音。"
                       action={<Button onClick={onCreate}><Plus aria-hidden />创建声音</Button>} />;
  }
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {packs.data.map((p) => (
        <Link key={p.id} href={`/voice-studio/pack?id=${p.id}`} className="rounded-xl focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none">
          <Card className="h-full transition-colors hover:border-primary/40">
            <CardHeader className="flex items-start justify-between gap-3 space-y-0">
              <div className="min-w-0">
                <CardTitle className="truncate text-base">{p.name}</CardTitle>
                <p className="text-sm text-muted-foreground">{p.character_name}</p>
              </div>
              <VoicePackStatusBadge status={p.status} />
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              {p.status === "processing" ? (
                <p className="text-info">{p.progress.label ?? "处理中"}…</p>
              ) : (
                <p>{p.total_files} 个音频 · {formatDuration(p.total_duration_s)}
                  {p.summary && p.summary.clips > 0 ? ` · 可用 ${formatDuration(p.summary.usable_s)}` : ""}</p>
              )}
              <p className="text-xs">更新于 {formatRelative(p.updated_at)}</p>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}

function ModelsTab() {
  const models = useVoiceModels();
  const [testing, setTesting] = useState<VoiceModel | null>(null);
  if (models.isPending) return <Loading />;
  if (models.isError) return <ErrorAlert title={errorText(models.error).message} onRetry={() => models.refetch()} />;
  if (!models.data.length) {
    return <EmptyState icon={Mic2} title="还没有训练好的声音" description="声音包审核完成并训练后，会出现在这里。" />;
  }
  return (
    <>
      <div className="grid gap-4 md:grid-cols-2">
        {models.data.map((v) => (
          <Card key={v.id}>
            <CardHeader className="flex items-start justify-between gap-3 space-y-0">
              <div className="min-w-0">
                <CardTitle className="truncate text-base">{v.name}</CardTitle>
                <p className="text-sm text-muted-foreground">{v.engine === "gpt_sovits" ? "GPT-SoVITS" : v.engine} · {v.base_model} · 第 {v.version} 版</p>
              </div>
              {v.approved ? <StatusBadge tone="ok" icon={CheckCircle2} label="已审核" />
                          : <StatusBadge tone="warn" icon={Timer} label="待试听" />}
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p className="text-muted-foreground">风格：{v.styles.map(styleLabel).join("、") || "平静"}</p>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setTesting(v)}>试听</Button>
                <LinkButton size="sm" variant="ghost" href={`/characters?new=1&voice=${v.id}`}>
                  用这个声音创建角色
                </LinkButton>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      <Dialog open={!!testing} onOpenChange={(open) => !open && setTesting(null)}>
        <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader><DialogTitle>试听：{testing?.name}</DialogTitle></DialogHeader>
          {testing && <VoiceTester model={testing} />}
        </DialogContent>
      </Dialog>
    </>
  );
}

function TrainingTab() {
  const jobs = useTrainingJobs();
  if (jobs.isPending) return <Loading />;
  if (jobs.isError) return <ErrorAlert title={errorText(jobs.error).message} onRetry={() => jobs.refetch()} />;
  if (!jobs.data.length) return <EmptyState icon={Timer} title="还没有训练任务" description="在声音包的「训练」步骤开始第一次训练。" />;
  return (
    <ul className="divide-y rounded-xl border">
      {jobs.data.map((j) => (
        <li key={j.id}>
          <Link href={`/voice-studio/pack?id=${j.voice_pack_id}`} className="flex flex-wrap items-center gap-4 px-4 py-3 hover:bg-muted/40">
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{j.voice_pack_name ?? j.voice_pack_id}</p>
              <p className="text-xs text-muted-foreground">{j.current_stage} · {formatRelative(j.created_at)}</p>
            </div>
            {!["completed", "failed", "cancelled"].includes(j.status) && (
              <Progress value={Math.round(j.progress * 100)} className="w-32" aria-label="训练进度" />
            )}
            <TrainingStatusBadge status={j.status} />
          </Link>
        </li>
      ))}
    </ul>
  );
}

export default function VoiceStudioPage() {
  const [creating, setCreating] = useState(false);
  return (
    <PageContainer wide>
      <PageHeader title="声音工坊" description="上传角色语音，检查数据，训练并试听角色的声音。"
                  actions={<Button onClick={() => setCreating(true)}><Plus aria-hidden />创建声音</Button>} />
      <Tabs defaultValue="packs">
        <TabsList className="mb-4">
          <TabsTrigger value="packs">声音包</TabsTrigger>
          <TabsTrigger value="models">声音模型</TabsTrigger>
          <TabsTrigger value="training">训练任务</TabsTrigger>
        </TabsList>
        <TabsContent value="packs"><PacksTab onCreate={() => setCreating(true)} /></TabsContent>
        <TabsContent value="models"><ModelsTab /></TabsContent>
        <TabsContent value="training"><TrainingTab /></TabsContent>
      </Tabs>
      <CreatePackDialog open={creating} onOpenChange={setCreating} />
    </PageContainer>
  );
}
