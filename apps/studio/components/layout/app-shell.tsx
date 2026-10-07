"use client";

import { AudioLines, Bot, Menu, MessageCircle, Settings, Sparkles, Users } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/chat", label: "对话", icon: MessageCircle },
  { href: "/characters", label: "角色", icon: Users },
  { href: "/voice-studio", label: "声音工坊", icon: AudioLines },
  { href: "/models", label: "模型", icon: Bot },
];

function normalize(path: string | null): string {
  return (path ?? "/").replace(/\/+$/, "") || "/";
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const path = normalize(usePathname());
  const item = (href: string, label: string, Icon: typeof MessageCircle) => {
    const active = path === href || path.startsWith(href + "/");
    return (
      <Link key={href} href={href} onClick={onNavigate} aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              active ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                     : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
            )}>
        <Icon className="size-4" aria-hidden />
        {label}
      </Link>
    );
  };
  return (
    <nav className="flex flex-1 flex-col gap-1" aria-label="主导航">
      {NAV.map((n) => item(n.href, n.label, n.icon))}
      <div className="flex-1" />
      {item("/settings", "设置", Settings)}
    </nav>
  );
}

/** Backend reachability only — the microphone and voice states live in Chat. */
function BackendStatus() {
  const health = useHealth(20000);
  const llm = health.data?.checks.llm;
  const state = health.isError ? "down" : !health.data ? "checking" : llm?.ok ? "ok" : "degraded";
  const label = { ok: "本地 AI 已连接", degraded: "对话模型未就绪", down: "后端未连接", checking: "正在连接…" }[state];
  return (
    <Link href="/settings" className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-muted-foreground hover:bg-sidebar-accent/60">
      <span aria-hidden className={cn("size-2 rounded-full",
        state === "ok" && "bg-ok", state === "degraded" && "bg-warn",
        state === "down" && "border-2 border-bad", state === "checking" && "animate-pulse bg-muted-foreground")} />
      {label}
    </Link>
  );
}

function Brand() {
  return (
    <Link href="/chat" className="flex items-center gap-2 px-3 py-1 font-semibold">
      <span className="flex size-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <Sparkles className="size-4" aria-hidden />
      </span>
      Character AI Studio
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <aside className="hidden w-60 shrink-0 flex-col gap-4 border-r bg-sidebar p-3 md:flex">
        <Brand />
        <NavLinks />
        <BackendStatus />
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b px-3 py-2 md:hidden">
          <Button variant="ghost" size="icon" onClick={() => setOpen(true)} aria-label="打开导航">
            <Menu aria-hidden />
          </Button>
          <Brand />
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="left" className="flex w-64 flex-col gap-4 bg-sidebar p-3">
          <SheetTitle className="sr-only">导航</SheetTitle>
          <Brand />
          <NavLinks onNavigate={() => setOpen(false)} />
          <BackendStatus />
        </SheetContent>
      </Sheet>
    </div>
  );
}

export function PageContainer({ children, wide }: { children: ReactNode; wide?: boolean }) {
  return <div className={cn("mx-auto w-full px-4 py-6 md:px-8 md:py-8", wide ? "max-w-6xl" : "max-w-4xl")}>{children}</div>;
}
