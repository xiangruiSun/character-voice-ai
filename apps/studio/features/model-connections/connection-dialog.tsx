"use client";

import { ChevronDown, Cloud, Globe, Loader2, Monitor, RefreshCw } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { ErrorAlert } from "@/components/shared/blocks";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { api, errorText } from "@/lib/api";
import { useConnectionMutations } from "@/lib/queries";
import type { ConnectionDraft, ConnectionKind, ModelConnection, ProbeResult, ProviderType } from "@/lib/types";
import { cn } from "@/lib/utils";

const KINDS: { kind: ConnectionKind; title: string; description: string; icon: typeof Monitor }[] = [
  { kind: "local", title: "本机模型", description: "Ollama、vLLM、LM Studio 等运行在这台电脑上的模型", icon: Monitor },
  { kind: "public_api", title: "公开 API", description: "使用服务商的 API Key，例如 OpenAI 兼容接口", icon: Cloud },
  { kind: "custom", title: "自定义端点", description: "其他任何 OpenAI 兼容的推理服务地址", icon: Globe },
];

const LOCAL_PROVIDERS: { value: ProviderType; label: string; url: string }[] = [
  { value: "ollama", label: "Ollama", url: "http://localhost:11434" },
  { value: "openai_compatible", label: "OpenAI 兼容本地服务（vLLM / LM Studio）", url: "http://localhost:8001/v1" },
];

interface FormState {
  name: string;
  provider_type: ProviderType;
  base_url: string;
  model_name: string;
  api_key: string;
  temperature: number;
  max_tokens: number;
  stream: boolean;
  organization: string;
  project: string;
}

function initialForm(kind: ConnectionKind, existing?: ModelConnection): FormState {
  if (existing) {
    return {
      name: existing.name, provider_type: existing.provider_type, base_url: existing.base_url,
      model_name: existing.model_name, api_key: "",
      temperature: existing.generation_defaults.temperature ?? 0.7,
      max_tokens: existing.generation_defaults.max_tokens ?? 400,
      stream: existing.generation_defaults.stream ?? true,
      organization: existing.extra.organization ?? "", project: existing.extra.project ?? "",
    };
  }
  const local = kind === "local";
  return {
    name: local ? "本地模型" : kind === "public_api" ? "云端模型" : "自定义模型",
    provider_type: local ? "ollama" : "openai_compatible",
    base_url: local ? "http://localhost:11434" : kind === "public_api" ? "https://api.openai.com/v1" : "",
    model_name: local ? "qwen3:4b" : "", api_key: "",
    temperature: 0.7, max_tokens: 1024, stream: true, organization: "", project: "",
  };
}

export function ConnectionDialog({ open, onOpenChange, existing }: {
  open: boolean; onOpenChange: (open: boolean) => void; existing?: ModelConnection;
}) {
  // The parent remounts this dialog (key) for every open, so props seed the state.
  const editing = !!existing;
  const [kind, setKind] = useState<ConnectionKind | null>(existing?.kind ?? null);
  const [form, setForm] = useState<FormState>(() => initialForm(existing?.kind ?? "local", existing));
  const [models, setModels] = useState<string[] | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);
  const m = useConnectionMutations();

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((f) => ({ ...f, [key]: value }));
    setProbe(null);
  };
  const needsKey = kind !== "local";
  const isOpenAI = form.base_url.includes("api.openai.com");

  function draft(): ConnectionDraft {
    const extra: Record<string, string> = {};
    if (form.organization) extra.organization = form.organization;
    if (form.project) extra.project = form.project;
    return {
      name: form.name.trim(), kind: kind ?? "local", provider_type: form.provider_type,
      base_url: form.base_url.trim(), model_name: form.model_name.trim(),
      api_key: form.api_key.trim() || null,
      generation_defaults: { temperature: form.temperature, max_tokens: form.max_tokens, stream: form.stream },
      extra,
    };
  }

  const valid = form.name.trim() && form.base_url.trim() && form.model_name.trim() &&
                (!needsKey || editing || form.api_key.trim() || kind === "custom");

  async function discover() {
    setLoadingModels(true); setError(null);
    try {
      const result = await api.post<{ models: string[] }>("/api/model-connections/models", {
        provider_type: form.provider_type, base_url: form.base_url, api_key: form.api_key || null,
        connection_id: existing?.id ?? null,
      });
      setModels(result.models);
      if (!result.models.length) setError({ message: "服务器没有返回可用模型", hint: form.provider_type === "ollama" ? "在终端运行：ollama pull qwen3:4b" : null });
    } catch (e) { setError(errorText(e)); setModels(null); }
    finally { setLoadingModels(false); }
  }

  async function test() {
    setError(null); setProbe(null);
    try {
      if (editing && !form.api_key) {
        // Test the stored key: save non-secret edits first, then test the saved connection.
        await m.update.mutateAsync({ id: existing.id, patch: patchFor() });
        const result = await m.test.mutateAsync(existing.id);
        const [message, hint] = (result.last_error ?? "").split("｜");
        setProbe({ ok: result.status === "connected", latency_ms: result.latency_ms,
                   message: result.status === "connected" ? "连接成功" : message, hint: hint ?? null });
      } else {
        setProbe(await m.probe.mutateAsync(draft()));
      }
    } catch (e) { setError(errorText(e)); }
  }

  function patchFor() {
    const d = draft();
    return { name: d.name, base_url: d.base_url, model_name: d.model_name,
             generation_defaults: d.generation_defaults, extra: d.extra,
             ...(form.api_key ? { api_key: form.api_key } : {}) };
  }

  async function save() {
    setError(null);
    try {
      const saved = editing
        ? await m.update.mutateAsync({ id: existing.id, patch: patchFor() })
        : await m.create.mutateAsync(draft());
      const tested = await m.test.mutateAsync(saved.id);
      if (tested.status === "connected") toast.success(`「${tested.name}」已连接`);
      else toast.warning(`已保存「${tested.name}」，但连接测试未通过`);
      onOpenChange(false);
    } catch (e) { setError(errorText(e)); }
  }

  const testing = m.probe.isPending || (m.test.isPending && !m.create.isPending && !m.update.isPending);
  const saving = m.create.isPending || m.update.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{editing ? "编辑模型连接" : kind ? "连接模型" : "你的模型在哪里？"}</DialogTitle>
          <DialogDescription>
            {kind ? "后端会代为调用这个模型；API Key 只保存在后端，并加密存储。" : "选择模型的运行方式。"}
          </DialogDescription>
        </DialogHeader>

        {!kind ? (
          <div className="grid gap-3" role="radiogroup" aria-label="模型位置">
            {KINDS.map(({ kind: k, title, description, icon: Icon }) => (
              <button key={k} type="button" role="radio" aria-checked={false}
                      onClick={() => { setKind(k); setForm(initialForm(k)); }}
                      className="flex items-start gap-3 rounded-xl border p-4 text-left transition-colors hover:border-primary/50 hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none">
                <Icon className="mt-0.5 size-5 text-muted-foreground" aria-hidden />
                <span>
                  <span className="block font-medium">{title}</span>
                  <span className="block text-sm text-muted-foreground">{description}</span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <form className="grid gap-4" onSubmit={(e) => { e.preventDefault(); void save(); }}>
            <div className="grid gap-2">
              <Label htmlFor="mc-name">连接名称</Label>
              <Input id="mc-name" value={form.name} onChange={(e) => set("name", e.target.value)} maxLength={100} required />
            </div>

            {kind === "local" && (
              <div className="grid gap-2">
                <Label>运行方式</Label>
                <Select items={LOCAL_PROVIDERS.map((p) => ({ value: p.value, label: p.label }))}
                        value={form.provider_type}
                        onValueChange={(value) => {
                          const provider = LOCAL_PROVIDERS.find((p) => p.value === value);
                          if (provider) setForm((f) => ({ ...f, provider_type: provider.value, base_url: provider.url }));
                          setModels(null); setProbe(null);
                        }}>
                  <SelectTrigger className="w-full" aria-label="运行方式"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {LOCAL_PROVIDERS.map((p) => <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="grid gap-2">
              <Label htmlFor="mc-url">{kind === "local" ? "服务地址" : "API 地址（Base URL）"}</Label>
              <Input id="mc-url" value={form.base_url} onChange={(e) => set("base_url", e.target.value)}
                     placeholder="https://…/v1" inputMode="url" autoComplete="off" required />
            </div>

            {needsKey && (
              <div className="grid gap-2">
                <Label htmlFor="mc-key">API Key{kind === "custom" ? "（如需要）" : ""}</Label>
                <Input id="mc-key" type="password" value={form.api_key} onChange={(e) => set("api_key", e.target.value)}
                       autoComplete="new-password" spellCheck={false}
                       placeholder={existing?.api_key_hint ? `已保存 ${existing.api_key_hint}（留空保持不变）` : "sk-…"} />
              </div>
            )}

            <div className="grid gap-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="mc-model">模型</Label>
                <Button type="button" variant="ghost" size="xs" onClick={discover} disabled={!form.base_url || loadingModels}>
                  {loadingModels ? <Loader2 className="animate-spin" aria-hidden /> : <RefreshCw aria-hidden />}
                  获取模型列表
                </Button>
              </div>
              <Input id="mc-model" value={form.model_name} onChange={(e) => set("model_name", e.target.value)}
                     placeholder={kind === "local" ? "qwen3:4b" : "模型名称"} autoComplete="off" required />
              {models && models.length > 0 && (
                <div className="flex flex-wrap gap-1.5" aria-label="可用模型">
                  {models.slice(0, 40).map((name) => (
                    <button key={name} type="button" onClick={() => set("model_name", name)}
                            className={cn("rounded-full border px-2.5 py-0.5 text-xs transition-colors hover:bg-muted",
                                          name === form.model_name && "border-primary bg-primary/10")}>
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <Collapsible>
              <CollapsibleTrigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
                <ChevronDown className="size-4" aria-hidden /> 高级设置
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 grid gap-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="grid gap-2">
                    <Label htmlFor="mc-temp">温度</Label>
                    <Input id="mc-temp" type="number" step="0.1" min={0} max={2} value={form.temperature}
                           onChange={(e) => set("temperature", Number(e.target.value))} />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="mc-max">最大回复长度（tokens）</Label>
                    <Input id="mc-max" type="number" min={16} max={8192} value={form.max_tokens}
                           onChange={(e) => set("max_tokens", Number(e.target.value))} />
                  </div>
                </div>
                <label className="flex items-center justify-between gap-3 text-sm">
                  流式输出
                  <Switch checked={form.stream} onCheckedChange={(v) => set("stream", v)} aria-label="流式输出" />
                </label>
                {isOpenAI && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="grid gap-2">
                      <Label htmlFor="mc-org">Organization（可选）</Label>
                      <Input id="mc-org" value={form.organization} onChange={(e) => set("organization", e.target.value)} />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="mc-proj">Project（可选）</Label>
                      <Input id="mc-proj" value={form.project} onChange={(e) => set("project", e.target.value)} />
                    </div>
                  </div>
                )}
              </CollapsibleContent>
            </Collapsible>

            <div role="status" aria-live="polite">
              {testing && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" aria-hidden />正在测试连接…</p>}
              {probe && (probe.ok ? (
                <Alert className="border-ok/30 bg-ok/5"><AlertTitle className="text-ok">连接成功</AlertTitle>
                  <AlertDescription>{probe.latency_ms != null ? `响应时间 ${Math.round(probe.latency_ms)} ms` : "模型可以正常使用"}</AlertDescription></Alert>
              ) : <ErrorAlert title={probe.message} hint={probe.hint} />)}
            </div>
            {error && <ErrorAlert title={error.message} hint={error.hint} />}

            <DialogFooter className="gap-2 sm:justify-between">
              {!editing ? <Button type="button" variant="ghost" onClick={() => setKind(null)}>上一步</Button> : <span />}
              <div className="flex gap-2">
                <Button type="button" variant="outline" onClick={test} disabled={!valid || testing || saving}>
                  {testing && <Loader2 className="animate-spin" aria-hidden />}测试连接
                </Button>
                <Button type="submit" disabled={!valid || saving}>
                  {saving && <Loader2 className="animate-spin" aria-hidden />}{editing ? "保存" : "保存并连接"}
                </Button>
              </div>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}
