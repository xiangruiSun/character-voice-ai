"use client";

import { Bot, Loader2, MoreHorizontal, Plus, Star } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { PageContainer } from "@/components/layout/app-shell";
import { EmptyState, ErrorAlert, PageHeader } from "@/components/shared/blocks";
import { ConnectionStatusBadge } from "@/components/shared/status";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { ConnectionDialog } from "@/features/model-connections/connection-dialog";
import { errorText } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useConnectionMutations, useConnections } from "@/lib/queries";
import type { ModelConnection } from "@/lib/types";

const PROVIDER_LABEL: Record<string, string> = { ollama: "Ollama", openai_compatible: "OpenAI 兼容" };
const KIND_LABEL: Record<string, string> = { local: "本机", public_api: "公开 API", custom: "自定义端点" };

function ConnectionCard({ connection, onEdit }: { connection: ModelConnection; onEdit: () => void }) {
  const m = useConnectionMutations();
  const [message, hint] = (connection.last_error ?? "").split("｜");

  async function run(action: () => Promise<unknown>, success?: string) {
    try { await action(); if (success) toast.success(success); }
    catch (e) { const { message: msg, hint: h } = errorText(e); toast.error(msg, { description: h ?? undefined }); }
  }

  const testing = m.test.isPending;
  return (
    <Card>
      <CardHeader className="flex items-start justify-between gap-3 space-y-0">
        <div className="min-w-0 space-y-1">
          <CardTitle className="flex items-center gap-2 text-base">
            <span className="truncate">{connection.name}</span>
            {connection.is_default && <Badge variant="secondary" className="gap-1"><Star className="size-3" aria-hidden />默认</Badge>}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {PROVIDER_LABEL[connection.provider_type]} · {KIND_LABEL[connection.kind]}
          </p>
        </div>
        <ConnectionStatusBadge status={testing ? "testing" : connection.status} />
      </CardHeader>
      <CardContent className="space-y-1 text-sm">
        <p className="font-medium">{connection.model_name}</p>
        <p className="truncate text-muted-foreground" title={connection.base_url}>{connection.base_url}</p>
        {connection.api_key_hint && (
          <p className="text-xs text-muted-foreground">API Key <span className="font-mono">{connection.api_key_hint}</span></p>
        )}
        <p className="text-xs text-muted-foreground">
          {connection.status === "connected" && connection.latency_ms != null
            ? `响应 ${Math.round(connection.latency_ms)} ms · 测试于 ${formatRelative(connection.last_tested_at)}`
            : " "}
        </p>
        {connection.status === "error" && message && (
          <ErrorAlert title={message} hint={hint} className="mt-3" />
        )}
      </CardContent>
      <CardFooter className="gap-2">
        <Button size="sm" variant="outline" disabled={testing}
                onClick={() => run(async () => {
                  const r = await m.test.mutateAsync(connection.id);
                  if (r.status === "connected") toast.success(`「${r.name}」连接正常`);
                  else toast.error(`「${r.name}」连接失败`);
                })}>
          {testing && <Loader2 className="animate-spin" aria-hidden />}测试
        </Button>
        <Button size="sm" variant="ghost" onClick={onEdit}>配置</Button>
        <div className="flex-1" />
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`更多操作：${connection.name}`} />}>
            <MoreHorizontal aria-hidden />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {!connection.is_default && (
              <DropdownMenuItem onClick={() => run(() => m.setDefault.mutateAsync(connection.id), "已设为默认对话模型")}>
                设为默认模型
              </DropdownMenuItem>
            )}
            <DropdownMenuItem onClick={() => run(() => m.update.mutateAsync({ id: connection.id, patch: { disabled: connection.status !== "disabled" } }))}>
              {connection.status === "disabled" ? "启用" : "停用"}
            </DropdownMenuItem>
            <DropdownMenuItem variant="destructive" onClick={() => run(() => m.remove.mutateAsync(connection.id), "已删除")}>
              删除
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </CardFooter>
    </Card>
  );
}

export default function ModelsPage() {
  const connections = useConnections();
  const [dialog, setDialog] = useState<{ open: boolean; existing?: ModelConnection; nonce: number }>({ open: false, nonce: 0 });

  const add = <Button onClick={() => setDialog((d) => ({ open: true, nonce: d.nonce + 1 }))}><Plus aria-hidden />添加模型连接</Button>;
  return (
    <PageContainer wide>
      <PageHeader title="模型" description="连接为角色提供思考能力的大语言模型：本机模型或你自己的 API。" actions={connections.data?.length ? add : undefined} />
      {connections.isPending ? (
        <div className="grid gap-4 md:grid-cols-2">{[0, 1].map((i) => <Skeleton key={i} className="h-48 rounded-xl" />)}</div>
      ) : connections.isError ? (
        <ErrorAlert title={errorText(connections.error).message} hint={errorText(connections.error).hint} onRetry={() => connections.refetch()} />
      ) : connections.data.length === 0 ? (
        <EmptyState icon={Bot} title="还没有连接任何模型" description="连接一个本机模型（例如 Ollama 上的 Qwen）或你的 API，角色才能思考和回答。" action={add} />
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {connections.data.map((c) => <ConnectionCard key={c.id} connection={c} onEdit={() => setDialog((d) => ({ open: true, existing: c, nonce: d.nonce + 1 }))} />)}
        </div>
      )}
      <ConnectionDialog key={dialog.nonce} open={dialog.open} existing={dialog.existing}
                        onOpenChange={(open) => setDialog((d) => ({ ...d, open }))} />
    </PageContainer>
  );
}
