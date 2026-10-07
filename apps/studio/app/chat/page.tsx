"use client";

import { ArrowRight, AudioLines, Bot, CheckCircle2, Circle, Users } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { ErrorAlert, LinkButton } from "@/components/shared/blocks";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { ChatView } from "@/features/chat/chat-view";
import { errorText } from "@/lib/api";
import { useCharacters, useOverview } from "@/lib/queries";

function FirstRun() {
  const overview = useOverview();
  const o = overview.data;
  const steps = [
    { done: !!o?.model_connections, title: "连接一个 AI 模型", text: "本机的 Qwen，或你自己的 API。", href: "/models", icon: Bot },
    { done: !!o?.voice_models, title: "创建一个声音", text: "上传角色的中文语音，训练出角色的声音。", href: "/voice-studio", icon: AudioLines },
    { done: !!o?.characters, title: "创建你的角色", text: "给角色选好模型、声音和性格。", href: "/characters", icon: Users },
  ];
  const next = steps.findIndex((s) => !s.done);
  return (
    <div className="mx-auto max-w-xl px-4 py-16">
      <h1 className="text-2xl font-semibold tracking-tight">欢迎来到 Character AI Studio</h1>
      <p className="mt-2 text-muted-foreground">三步之后，就可以和你的角色对话了。</p>
      <ol className="mt-8 space-y-3">
        {steps.map((s, i) => (
          <li key={s.title} className="flex items-center gap-4 rounded-xl border p-4">
            {s.done ? <CheckCircle2 className="size-5 text-ok" aria-label="已完成" /> : <Circle className="size-5 text-muted-foreground" aria-label="未完成" />}
            <div className="min-w-0 flex-1">
              <p className="font-medium">{i + 1}. {s.title}</p>
              <p className="text-sm text-muted-foreground">{s.text}</p>
            </div>
            {!s.done && (
              <LinkButton href={s.href} size="sm" variant={i === next ? "default" : "outline"}>
                去完成<ArrowRight aria-hidden />
              </LinkButton>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function ChatInner() {
  const params = useSearchParams();
  const router = useRouter();
  const characters = useCharacters();

  if (characters.isPending) {
    return <div className="mx-auto max-w-3xl space-y-4 p-6"><Skeleton className="h-12" /><Skeleton className="h-64" /></div>;
  }
  if (characters.isError) {
    const { message, hint } = errorText(characters.error);
    return <div className="mx-auto max-w-xl p-6"><ErrorAlert title={message} hint={hint} onRetry={() => characters.refetch()} /></div>;
  }
  const ready = characters.data.filter((c) => c.status === "ready");
  if (!ready.length) return <FirstRun />;
  const wanted = params.get("character");
  const character = ready.find((c) => c.id === wanted) ?? ready[0];
  const items = ready.map((c) => ({ value: c.id, label: c.name }));

  return (
    <div className="relative h-full">
      {ready.length > 1 && (
        <div className="absolute top-4 right-4 z-10">
          <Select items={items} value={character.id} onValueChange={(v) => v && router.replace(`/chat?character=${v}`)}>
            <SelectTrigger size="sm" aria-label="切换角色"><SelectValue /></SelectTrigger>
            <SelectContent align="end">{items.map((i) => <SelectItem key={i.value} value={i.value}>{i.label}</SelectItem>)}</SelectContent>
          </Select>
        </div>
      )}
      <ChatView key={character.id} character={character} />
    </div>
  );
}

export default function ChatPage() {
  return <Suspense fallback={null}><ChatInner /></Suspense>;
}
