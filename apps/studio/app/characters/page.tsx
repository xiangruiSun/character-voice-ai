"use client";

import { MessageCircle, MoreHorizontal, Plus, Users } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { toast } from "sonner";

import { PageContainer } from "@/components/layout/app-shell";
import { EmptyState, ErrorAlert, LinkButton, PageHeader } from "@/components/shared/blocks";
import { CharacterStatusBadge } from "@/components/shared/status";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { Avatar, CharacterWizard } from "@/features/characters/character-wizard";
import { errorText } from "@/lib/api";
import { useCharacterMutations, useCharacters } from "@/lib/queries";
import type { Character } from "@/lib/types";

function CharactersInner() {
  const params = useSearchParams();
  const router = useRouter();
  const characters = useCharacters();
  const m = useCharacterMutations();
  // /characters?new=1&voice=…&name=… arrives from Voice Studio's last step.
  type WizardState = { open: boolean; existing?: Character; startAt?: number; nonce: number;
                       preset?: { voice?: string | null; name?: string | null } };
  const [wizard, setWizard] = useState<WizardState>(() =>
    params.get("new") ? { open: true, nonce: 1, preset: { voice: params.get("voice"), name: params.get("name") } }
                      : { open: false, nonce: 0 });
  useEffect(() => { if (params.get("new")) router.replace("/characters"); }, [params, router]);
  // A fresh key per open: the wizard starts from its props every time.
  const openWizard = (w: Omit<WizardState, "nonce">) => setWizard((prev) => ({ ...w, nonce: prev.nonce + 1 }));

  const create = <Button onClick={() => openWizard({ open: true })}><Plus aria-hidden />创建角色</Button>;

  return (
    <>
      <PageHeader title="角色" description="角色由身份、对话模型和声音组成。" actions={characters.data?.length ? create : undefined} />
      {characters.isPending ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-52 rounded-xl" />)}</div>
      ) : characters.isError ? (
        <ErrorAlert title={errorText(characters.error).message} onRetry={() => characters.refetch()} />
      ) : characters.data.length === 0 ? (
        <EmptyState icon={Users} title="还没有角色" description="选择一个对话模型和一个训练好的声音，就能创建第一个角色。" action={create} />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {characters.data.map((c) => (
            <Card key={c.id} className="flex flex-col">
              <CardHeader className="flex items-start gap-3 space-y-0">
                <Avatar name={c.name} url={c.avatar_url} size={48} />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="truncate font-semibold">{c.name}</p>
                  <CharacterStatusBadge status={c.status} />
                </div>
              </CardHeader>
              <CardContent className="flex-1 space-y-2 text-sm">
                {c.description && <p className="line-clamp-2 text-muted-foreground">{c.description}</p>}
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <dt className="text-muted-foreground">大脑</dt>
                  <dd className="truncate">{c.model_label ?? "—"}{c.uses_default_model && c.model_label ? "（默认）" : ""}</dd>
                  <dt className="text-muted-foreground">声音</dt>
                  <dd className="truncate">{c.voice_label ?? "—"}</dd>
                </dl>
                {c.problems.length > 0 && <p className="text-xs text-warn">{c.problems.join("；")}</p>}
              </CardContent>
              <CardFooter className="gap-2">
                <LinkButton size="sm" disabled={c.status !== "ready"} href={`/chat?character=${c.id}`}>
                  <MessageCircle aria-hidden />对话
                </LinkButton>
                <Button size="sm" variant="outline" onClick={() => openWizard({ open: true, existing: c })}>编辑</Button>
                <Button size="sm" variant="ghost" onClick={() => openWizard({ open: true, existing: c, startAt: 3 })}>测试</Button>
                <div className="flex-1" />
                <DropdownMenu>
                  <DropdownMenuTrigger render={<Button size="icon-sm" variant="ghost" aria-label={`更多操作：${c.name}`} />}>
                    <MoreHorizontal aria-hidden />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem variant="destructive" onClick={async () => {
                      try { await m.remove.mutateAsync(c.id); toast.success(`已删除「${c.name}」`); }
                      catch (e) { toast.error(errorText(e).message); }
                    }}>删除角色</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
      <CharacterWizard key={wizard.nonce} open={wizard.open} existing={wizard.existing} preset={wizard.preset} startAt={wizard.startAt}
                       onOpenChange={(open) => setWizard((w) => ({ ...w, open }))} />
    </>
  );
}

export default function CharactersPage() {
  return (
    <PageContainer wide>
      <Suspense fallback={<Skeleton className="h-64 rounded-xl" />}><CharactersInner /></Suspense>
    </PageContainer>
  );
}
