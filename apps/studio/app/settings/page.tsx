"use client";

import { AlertCircle, CheckCircle2, Loader2, RefreshCw } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState, useSyncExternalStore } from "react";

import { PageContainer } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/shared/blocks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { loadChatSettings, saveChatSettings, serverChatSettings, subscribeChatSettings, type ChatSettings } from "@/lib/chat-settings";
import { useHealth } from "@/lib/queries";

const SERVICES: { key: string; label: string; down: string }[] = [
  { key: "backend", label: "后端", down: "后端服务未运行" },
  { key: "database", label: "数据库", down: "无法访问数据库" },
  { key: "redis", label: "任务队列（Redis / Memurai）", down: "训练和数据处理暂不可用" },
  { key: "worker", label: "训练进程", down: "训练和数据处理暂不可用：请启动训练进程" },
  { key: "llm", label: "默认对话模型", down: "默认对话模型不可用" },
  { key: "voice_engine", label: "语音引擎（GPT-SoVITS）", down: "语音生成暂不可用" },
];

const THEMES = [{ value: "system", label: "跟随系统" }, { value: "light", label: "浅色" }, { value: "dark", label: "深色" }];

export default function SettingsPage() {
  const health = useHealth(10000);
  const { theme, setTheme } = useTheme();
  const settings = useSyncExternalStore(subscribeChatSettings, loadChatSettings, serverChatSettings);
  const [mics, setMics] = useState<{ value: string; label: string }[]>([{ value: "", label: "系统默认" }]);

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then((devices) => {
      const inputs = devices.filter((d) => d.kind === "audioinput" && d.deviceId && d.deviceId !== "default");
      setMics([{ value: "", label: "系统默认" }, ...inputs.map((d, i) => ({ value: d.deviceId, label: d.label || `麦克风 ${i + 1}` }))]);
    }).catch(() => {});
  }, []);
  const update = (patch: Partial<ChatSettings>) => saveChatSettings({ ...settings, ...patch });

  return (
    <PageContainer>
      <PageHeader title="设置" />
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">对话</CardTitle>
            <CardDescription>只保存在这个浏览器里。</CardDescription>
          </CardHeader>
          {settings && (
            <CardContent className="grid gap-5">
              <label className="flex items-center justify-between gap-4 text-sm">
                自动播放角色语音
                <Switch checked={settings.autoplay} onCheckedChange={(v) => update({ autoplay: v })} aria-label="自动播放角色语音" />
              </label>
              <div className="grid gap-2">
                <Label>音量</Label>
                <Slider value={[settings.volume * 100]} max={100} step={5} aria-label="音量"
                        onValueChange={(v) => update({ volume: (Array.isArray(v) ? v[0] : v) / 100 })} />
              </div>
              <div className="grid gap-2">
                <Label>麦克风</Label>
                <Select items={mics} value={settings.micId} onValueChange={(v) => update({ micId: v ?? "" })}>
                  <SelectTrigger className="w-full" aria-label="麦克风"><SelectValue /></SelectTrigger>
                  <SelectContent>{mics.map((m) => <SelectItem key={m.value || "default"} value={m.value}>{m.label}</SelectItem>)}</SelectContent>
                </Select>
                <span className="text-xs text-muted-foreground">在对话中授权过麦克风后，这里会列出具体设备名称。</span>
              </div>
              <div className="grid gap-2">
                <Label>外观</Label>
                <Select items={THEMES} value={theme ?? "system"} onValueChange={(v) => v && setTheme(v)}>
                  <SelectTrigger className="w-40" aria-label="外观"><SelectValue /></SelectTrigger>
                  <SelectContent>{THEMES.map((t) => <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>)}</SelectContent>
                </Select>
              </div>
            </CardContent>
          )}
        </Card>

        <Card>
          <CardHeader className="flex items-start justify-between gap-3 space-y-0">
            <div>
              <CardTitle className="text-base">服务状态</CardTitle>
              <CardDescription>这些服务都运行在这台电脑上。</CardDescription>
            </div>
            <Button size="sm" variant="outline" onClick={() => health.refetch()} disabled={health.isFetching}>
              {health.isFetching ? <Loader2 className="animate-spin" aria-hidden /> : <RefreshCw aria-hidden />}刷新
            </Button>
          </CardHeader>
          <CardContent>
            <ul className="divide-y">
              {SERVICES.map((s) => {
                const check = health.isError ? (s.key === "backend" ? { ok: false } : undefined) : health.data?.checks[s.key];
                return (
                  <li key={s.key} className="flex items-center gap-3 py-2.5 text-sm">
                    {!check ? <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label="检查中" />
                      : check.ok ? <CheckCircle2 className="size-4 text-ok" aria-label="正常" />
                      : <AlertCircle className="size-4 text-bad" aria-label="异常" />}
                    <span className="flex-1">{s.label}{check?.detail && check.ok ? <span className="text-muted-foreground"> · {check.detail}</span> : null}</span>
                    <span className={check?.ok ? "text-ok" : "text-muted-foreground"}>
                      {!check ? "检查中" : check.ok ? "正常" : (check.detail && s.key === "llm" ? check.detail : s.down)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  );
}
