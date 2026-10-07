"use client";

import { AlertCircle, Check, Pause, Play, type LucideIcon } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import Link from "next/link";

import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function PageHeader({ title, description, actions }: {
  title: string; description?: string; actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4 pb-6">
      <div className="min-w-0 space-y-1">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, description, action }: {
  icon: LucideIcon; title: string; description?: string; action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-16 text-center">
      <div className="mb-4 rounded-full bg-muted p-3"><Icon className="size-6 text-muted-foreground" aria-hidden /></div>
      <p className="font-medium">{title}</p>
      {description && <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorAlert({ title, hint, onRetry, retryLabel = "重试", className }: {
  title: string; hint?: string | null; onRetry?: () => void; retryLabel?: string; className?: string;
}) {
  return (
    <Alert variant="destructive" className={className} role="alert">
      <AlertCircle aria-hidden />
      <AlertTitle>{title}</AlertTitle>
      {(hint || onRetry) && (
        <AlertDescription className="flex flex-wrap items-center gap-3">
          {hint && <span>{hint}</span>}
          {onRetry && <Button size="sm" variant="outline" onClick={onRetry}>{retryLabel}</Button>}
        </AlertDescription>
      )}
    </Alert>
  );
}

export function Stepper({ steps, current, onSelect }: {
  steps: { key: string; label: string; disabled?: boolean }[];
  current: number;
  onSelect?: (index: number) => void;
}) {
  return (
    <nav aria-label="进度" className="mb-8">
      <ol className="flex flex-wrap items-center gap-x-2 gap-y-3">
        {steps.map((step, index) => {
          const state = index < current ? "done" : index === current ? "current" : "todo";
          const clickable = !!onSelect && !step.disabled && index !== current;
          return (
            <li key={step.key} className="flex items-center gap-2">
              <button
                type="button"
                disabled={!clickable}
                onClick={() => clickable && onSelect?.(index)}
                aria-current={state === "current" ? "step" : undefined}
                className={cn(
                  "flex items-center gap-2 rounded-full py-1 pr-3 pl-1 text-sm transition-colors",
                  clickable && "hover:bg-muted",
                  state === "current" ? "font-medium text-foreground" : "text-muted-foreground",
                )}
              >
                <span className={cn(
                  "flex size-6 items-center justify-center rounded-full border text-xs",
                  state === "done" && "border-primary bg-primary text-primary-foreground",
                  state === "current" && "border-primary text-primary",
                )}>
                  {state === "done" ? <Check className="size-3.5" aria-hidden /> : index + 1}
                </span>
                {step.label}
                <span className="sr-only">{state === "done" ? "（已完成）" : state === "current" ? "（当前步骤）" : ""}</span>
              </button>
              {index < steps.length - 1 && <span className="h-px w-6 bg-border" aria-hidden />}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

// One clip plays at a time across the page.
let activeAudio: HTMLAudioElement | null = null;

export function PlayButton({ src, label = "播放" }: { src: string; label?: string }) {
  const [playing, setPlaying] = useState(false);
  const audio = useRef<HTMLAudioElement | null>(null);
  useEffect(() => () => { audio.current?.pause(); }, []);

  function toggle() {
    if (!audio.current) {
      audio.current = new Audio(src);
      audio.current.onended = () => setPlaying(false);
      audio.current.onpause = () => setPlaying(false);
      audio.current.onplay = () => setPlaying(true);
    }
    if (playing) { audio.current.pause(); return; }
    if (activeAudio && activeAudio !== audio.current) activeAudio.pause();
    activeAudio = audio.current;
    audio.current.currentTime = 0;
    void audio.current.play();
  }

  return (
    <Button type="button" size="icon-sm" variant="outline" onClick={toggle}
            aria-label={playing ? "暂停" : label} aria-pressed={playing}>
      {playing ? <Pause aria-hidden /> : <Play aria-hidden />}
    </Button>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/** Navigation that looks like a button but is a real link (announced as a link). */
export function LinkButton({ href, children, variant = "default", size = "default", className, disabled }: {
  href: string; children: ReactNode; className?: string; disabled?: boolean;
  variant?: "default" | "outline" | "secondary" | "ghost" | "link";
  size?: "default" | "xs" | "sm" | "lg";
}) {
  if (disabled) return <Button variant={variant} size={size} className={className} disabled>{children}</Button>;
  return <Link href={href} className={cn(buttonVariants({ variant, size }), className)}>{children}</Link>;
}
