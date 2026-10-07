"use client";

import { ArrowLeft, ArrowRight, ImagePlus, Loader2, Send } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { ErrorAlert, Stepper, LinkButton } from "@/components/shared/blocks";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { errorText, uploadFiles } from "@/lib/api";
import { prettyModel, styleLabel } from "@/lib/format";
import { useCharacterMutations, useConnections, useVoiceModels } from "@/lib/queries";
import type { Character, CharacterInput } from "@/lib/types";

const STEPS = ["身份", "大脑", "声音", "预览"];
const DEFAULT_MODEL = "__default__";

function blank(): CharacterInput {
  return { name: "", description: "", system_prompt: "", greeting: "", model_connection_id: null,
           voice_model_id: null, default_voice_style: "neutral" };
}
const PROMPT_TEMPLATE = "性格：\n说话方式：\n和用户的关系：\n不要做的事：";

export function Avatar({ name, url, size = 40 }: { name: string; url?: string | null; size?: number }) {
  return url ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={url} alt="" width={size} height={size} className="shrink-0 rounded-full object-cover" style={{ width: size, height: size }} />
  ) : (
    <span aria-hidden className="flex shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-300 to-fuchsia-300 font-semibold text-white"
          style={{ width: size, height: size, fontSize: size * 0.42 }}>{(name || "?").slice(0, 1)}</span>
  );
}

export function CharacterWizard({ open, onOpenChange, existing, preset, startAt = 0 }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  existing?: Character;
  preset?: { voice?: string | null; name?: string | null };
  startAt?: number;
}) {
  const router = useRouter();
  const connections = useConnections();
  const voices = useVoiceModels();
  const m = useCharacterMutations();
  // The parent remounts this component (key) for every open, so props seed the state.
  const [step, setStep] = useState(startAt);
  const [form, setForm] = useState<CharacterInput>(() => existing ? {
    name: existing.name, description: existing.description, system_prompt: existing.system_prompt,
    greeting: existing.greeting, model_connection_id: existing.model_connection_id,
    voice_model_id: existing.voice_model_id, default_voice_style: existing.default_voice_style,
  } : { ...blank(), name: preset?.name ?? "", voice_model_id: preset?.voice ?? null,
        system_prompt: preset?.name ? `你是${preset.name}。\n${PROMPT_TEMPLATE}` : "" });
  const [avatar, setAvatar] = useState<File | null>(null);
  const [preview, setPreview] = useState<{ text: string; reply?: string; audio?: string; ms?: string } | null>(null);
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);
  const avatarInput = useRef<HTMLInputElement>(null);


  const set = <K extends keyof CharacterInput>(k: K, v: CharacterInput[K]) => setForm((f) => ({ ...f, [k]: v }));
  const defaultConn = connections.data?.find((c) => c.is_default);
  const usableVoices = voices.data?.filter((v) => v.status === "ready") ?? [];
  const voice = usableVoices.find((v) => v.id === form.voice_model_id);
  const connectionItems = [
    { value: DEFAULT_MODEL, label: defaultConn ? `跟随默认模型（${defaultConn.name}）` : "跟随默认模型" },
    ...(connections.data ?? []).filter((c) => c.status !== "disabled").map((c) => ({ value: c.id, label: `${c.name} · ${prettyModel(c.model_name)}` })),
  ];
  const voiceItems = usableVoices.map((v) => ({ value: v.id, label: `${v.name}${v.approved ? "" : "（未审核）"}` }));
  const styleItems = (voice?.styles ?? ["neutral"]).map((s) => ({ value: s, label: styleLabel(s) }));

  const canNext = [!!form.name.trim(), true, !!form.voice_model_id, true][step];

  async function runPreview() {
    const text = preview?.text?.trim() || "你好，介绍一下你自己吧。";
    setError(null);
    setPreview({ text });
    try {
      const r = await m.preview.mutateAsync({
        text, name: form.name, system_prompt: form.system_prompt, model_connection_id: form.model_connection_id,
        voice_model_id: form.voice_model_id, voice_style: form.default_voice_style,
      });
      setPreview({ text, reply: r.reply, audio: r.audio_url,
                   ms: `思考 ${(r.llm_ms / 1000).toFixed(1)} 秒${r.tts_ms ? ` · 语音 ${(r.tts_ms / 1000).toFixed(1)} 秒` : ""}` });
    } catch (e) { setError(errorText(e)); setPreview({ text }); }
  }

  async function save() {
    setError(null);
    try {
      const saved = existing
        ? await m.update.mutateAsync({ id: existing.id, patch: { ...form, use_default_model: form.model_connection_id === null } })
        : await m.create.mutateAsync(form);
      if (avatar) await uploadFiles(`/api/characters/${saved.id}/avatar`, [avatar], "file", () => {});
      toast.success(existing ? "已保存" : `角色「${saved.name}」已创建`, {
        action: { label: "开始对话", onClick: () => router.push(`/chat?character=${saved.id}`) },
      });
      onOpenChange(false);
    } catch (e) { setError(errorText(e)); }
  }

  const saving = m.create.isPending || m.update.isPending;
  const localAvatar = useMemo(() => (avatar ? URL.createObjectURL(avatar) : null), [avatar]);
  const avatarUrl = localAvatar ?? existing?.avatar_url;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92dvh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{existing ? `编辑 ${existing.name}` : "创建角色"}</DialogTitle>
          <DialogDescription>角色 = 身份 + 大脑（对话模型与性格）+ 声音。</DialogDescription>
        </DialogHeader>
        <Stepper steps={STEPS.map((label, i) => ({ key: label, label, disabled: i > step && !canNext }))}
                 current={step} onSelect={(i) => (i < step || canNext) && setStep(i)} />

        {step === 0 && (
          <div className="grid gap-4">
            <div className="flex items-center gap-4">
              <Avatar name={form.name} url={avatarUrl} size={64} />
              <Button type="button" variant="outline" size="sm" onClick={() => avatarInput.current?.click()}>
                <ImagePlus aria-hidden />{avatarUrl ? "更换头像" : "上传头像"}
              </Button>
              <input ref={avatarInput} type="file" accept="image/png,image/jpeg,image/webp,image/gif" className="hidden"
                     onChange={(e) => setAvatar(e.target.files?.[0] ?? null)} />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="ch-name">角色名称</Label>
              <Input id="ch-name" value={form.name} onChange={(e) => set("name", e.target.value)} maxLength={100} autoFocus />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="ch-desc">简介（可选）</Label>
              <Input id="ch-desc" value={form.description} onChange={(e) => set("description", e.target.value)} maxLength={500} placeholder="一句话介绍这个角色" />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="ch-greet">开场白（可选）</Label>
              <Input id="ch-greet" value={form.greeting} onChange={(e) => set("greeting", e.target.value)} maxLength={500} placeholder="打开对话时角色说的第一句话" />
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="grid gap-4">
            <div className="grid gap-2">
              <Label>对话模型</Label>
              {connections.data?.length === 0 ? (
                <ErrorAlert title="还没有连接任何模型" hint="请先在「模型」页面添加一个模型连接。" />
              ) : (
                <Select items={connectionItems} value={form.model_connection_id ?? DEFAULT_MODEL}
                        onValueChange={(v) => set("model_connection_id", v === DEFAULT_MODEL || !v ? null : v)}>
                  <SelectTrigger className="w-full" aria-label="对话模型"><SelectValue /></SelectTrigger>
                  <SelectContent>{connectionItems.map((c) => <SelectItem key={c.value} value={c.value}>{c.label}</SelectItem>)}</SelectContent>
                </Select>
              )}
              <span className="text-xs text-muted-foreground">选择「跟随默认模型」时，以后更换默认模型，这个角色也会跟着变。</span>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="ch-prompt">性格与设定</Label>
              <Textarea id="ch-prompt" value={form.system_prompt} onChange={(e) => set("system_prompt", e.target.value)}
                        rows={9} maxLength={8000} placeholder={`你是……\n${PROMPT_TEMPLATE}`} />
              <span className="text-xs text-muted-foreground">写给模型看的角色设定：性格、说话方式、和用户的关系、禁忌。</span>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="grid gap-4">
            <div className="grid gap-2">
              <Label>声音</Label>
              {usableVoices.length === 0 ? (
                <ErrorAlert title="还没有训练好的声音" hint="请先在「声音工坊」上传语音并完成训练。" />
              ) : (
                <Select items={voiceItems} value={form.voice_model_id}
                        onValueChange={(v) => { set("voice_model_id", v); set("default_voice_style", "neutral"); }}>
                  <SelectTrigger className="w-full" aria-label="声音"><SelectValue placeholder="选择一个声音" /></SelectTrigger>
                  <SelectContent>{voiceItems.map((v) => <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>)}</SelectContent>
                </Select>
              )}
            </div>
            {voice && (
              <div className="grid gap-2">
                <Label>默认语气</Label>
                <Select items={styleItems} value={form.default_voice_style} onValueChange={(v) => v && set("default_voice_style", v)}>
                  <SelectTrigger className="w-40" aria-label="默认语气"><SelectValue /></SelectTrigger>
                  <SelectContent>{styleItems.map((s) => <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>)}</SelectContent>
                </Select>
                <span className="text-xs text-muted-foreground">对话中会根据每句话的情绪自动选择合适的语气。</span>
              </div>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="grid gap-4">
            <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void runPreview(); }}>
              <Input value={preview?.text ?? ""} onChange={(e) => setPreview({ text: e.target.value })}
                     placeholder="你好，介绍一下你自己吧。" aria-label="试着对角色说一句话" />
              <Button type="submit" disabled={m.preview.isPending}>
                {m.preview.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Send aria-hidden />}试一试
              </Button>
            </form>
            <div className="min-h-24 rounded-xl border bg-muted/30 p-4" aria-live="polite">
              {m.preview.isPending ? (
                <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" aria-hidden />正在思考并生成语音…</p>
              ) : preview?.reply ? (
                <div className="flex gap-3">
                  <Avatar name={form.name} url={avatarUrl} size={32} />
                  <div className="min-w-0 space-y-2">
                    <p className="whitespace-pre-wrap">{preview.reply}</p>
                    {preview.audio && <audio controls autoPlay src={preview.audio} className="h-9 w-full max-w-sm" />}
                    <p className="text-xs text-muted-foreground">{preview.ms}</p>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">对角色说一句话，听听回答和声音。</p>
              )}
            </div>
          </div>
        )}

        {error && <ErrorAlert title={error.message} hint={error.hint} />}

        <div className="flex flex-wrap justify-between gap-2 pt-2">
          {step > 0 ? <Button variant="ghost" onClick={() => setStep(step - 1)}><ArrowLeft aria-hidden />上一步</Button> : <span />}
          <div className="flex gap-2">
            {step < 3 && <Button variant={existing ? "outline" : "default"} onClick={() => setStep(step + 1)} disabled={!canNext}>下一步<ArrowRight aria-hidden /></Button>}
            {(step === 3 || existing) && (
              <Button onClick={save} disabled={saving || !form.name.trim() || !form.voice_model_id}>
                {saving && <Loader2 className="animate-spin" aria-hidden />}{existing ? "保存" : "创建角色"}
              </Button>
            )}
          </div>
        </div>
        {!existing && step === 2 && usableVoices.length === 0 && (
          <LinkButton href="/voice-studio" variant="link">去声音工坊</LinkButton>
        )}
      </DialogContent>
    </Dialog>
  );
}
