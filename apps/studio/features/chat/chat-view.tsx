"use client";

import { ArrowUp, Loader2, Mic, MicOff, RotateCcw, Sparkles, Square, Volume2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import { Avatar } from "@/features/characters/character-wizard";
import { loadChatSettings } from "@/lib/chat-settings";
import { prettyModel } from "@/lib/format";
import type { Character } from "@/lib/types";
import { cn } from "@/lib/utils";

import { VoiceEngine, VoiceState, type VoiceStateValue } from "./voice-engine";

const BUSY: VoiceStateValue[] = [VoiceState.THINKING, VoiceState.GENERATING_SPEECH, VoiceState.SPEAKING];
const LISTENING: VoiceStateValue[] = [VoiceState.LISTENING, VoiceState.SPEECH_DETECTED];

/** What the status bar says — the same physical row, whatever is happening. */
function statusView(state: VoiceStateValue, name: string, hasTalked: boolean) {
  switch (state) {
    case VoiceState.REQUESTING_MIC_PERMISSION: return { tone: "text-info", icon: "spin", label: "正在请求麦克风权限…", hint: "请在浏览器提示中允许使用麦克风" };
    case VoiceState.LISTENING: return { tone: "text-listen", icon: "rec", label: "正在聆听", hint: "说完后点击「完成」" };
    case VoiceState.SPEECH_DETECTED: return { tone: "text-listen", icon: "rec", label: "正在聆听", hint: "听到你的声音了 · 说完点击「完成」" };
    case VoiceState.TRANSCRIBING: return { tone: "text-info", icon: "spin", label: "正在识别语音…", hint: "麦克风已关闭" };
    case VoiceState.THINKING: return { tone: "text-think", icon: "think", label: "正在思考…", hint: `${name}正在组织回复` };
    case VoiceState.GENERATING_SPEECH: return { tone: "text-think", icon: "spin", label: "正在生成语音…", hint: "文字已就绪，正在合成声音" };
    case VoiceState.SPEAKING: return { tone: "text-speak", icon: "speak", label: `${name}正在说话`, hint: "点击麦克风可以打断" };
    case VoiceState.ERROR: return { tone: "text-bad", icon: "error", label: "出现问题", hint: "" };
    default: return { tone: "text-muted-foreground", icon: "dot", label: "准备就绪", hint: hasTalked ? "点击麦克风继续对话" : "点击麦克风开始说话" };
  }
}

function StatusIcon({ icon }: { icon: string }) {
  if (icon === "spin") return <Loader2 className="size-4 animate-spin" aria-hidden />;
  if (icon === "rec") return <span className="size-2.5 animate-cv-pulse rounded-full bg-listen" aria-hidden />;
  if (icon === "think") return <Sparkles className="size-4" aria-hidden />;
  if (icon === "speak") return <Volume2 className="size-4" aria-hidden />;
  if (icon === "error") return <X className="size-4" aria-hidden />;
  return <span className="size-2 rounded-full bg-muted-foreground/60" aria-hidden />;
}

function LevelMeter({ engine }: { engine: VoiceEngine }) {
  const bars = useRef<(HTMLSpanElement | null)[]>([]);
  useEffect(() => {
    let raf = 0;
    const smooth = new Array(7).fill(0);
    const tick = () => {
      const level = engine.level();
      bars.current.forEach((bar, i) => {
        if (!bar) return;
        const shape = 0.55 + 0.45 * Math.sin((i / 6) * Math.PI);
        smooth[i] = smooth[i] * 0.55 + level * shape * (0.75 + Math.random() * 0.25) * 0.45;
        bar.style.height = `${Math.max(4, Math.round(4 + smooth[i] * 18))}px`;
      });
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [engine]);
  return (
    <span className="flex h-6 items-center gap-[3px]" aria-hidden data-testid="level-meter">
      {Array.from({ length: 7 }, (_, i) => <span key={i} ref={(el) => { bars.current[i] = el; }} className="w-[3px] rounded-full bg-listen" style={{ height: 4 }} />)}
    </span>
  );
}

function SpeakingBars() {
  return (
    <span className="flex h-6 items-center gap-[3px]" aria-hidden>
      {[0, 1, 2, 3, 4].map((i) => <span key={i} className="w-[3px] animate-cv-talk rounded-full bg-speak" style={{ animationDelay: `${i * 0.15}s` }} />)}
    </span>
  );
}

export function ChatView({ character }: { character: Character }) {
  const engine = useMemo(() => new VoiceEngine(character.id), [character.id]);
  const snap = useSyncExternalStore(engine.subscribe, engine.getSnapshot, engine.getSnapshot);
  const [text, setText] = useState("");
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    engine.configure(loadChatSettings());
    void engine.connect().then(() => engine.greet(character.greeting));
    return () => { void engine.dispose(); };
  }, [engine, character.greeting]);

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [snap.messages]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (LISTENING.includes(snap.state)) void engine.finishListening();
      else if (BUSY.includes(snap.state)) engine.interrupt();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [engine, snap.state]);

  const name = snap.characterName || character.name;
  const status = statusView(snap.state, name, snap.hasTalked);
  const listening = LISTENING.includes(snap.state);
  const ready = snap.conn === "connected";
  const typed = text.trim().length > 0;

  // Enter while still connecting must not swallow the message: it stays in the box.
  function send() { if (typed && engine.sendText(text)) setText(""); }

  return (
    <div className="flex h-full flex-col" data-voice-state={snap.state} data-mic={snap.mic}>
      {/* character */}
      <div className="mx-auto flex w-full max-w-3xl items-center gap-3 px-4 pt-5 pb-3">
        <span className={cn("rounded-full p-0.5 ring-2 ring-transparent transition-shadow",
                            snap.state === VoiceState.SPEAKING && "ring-speak", listening && "ring-listen/60")}>
          <Avatar name={name} url={character.avatar_url} size={44} />
        </span>
        <div className="min-w-0 flex-1">
          <h1 className="truncate font-semibold">{name}</h1>
          <p className="truncate text-xs text-muted-foreground">
            {[snap.modelLabel ? prettyModel(snap.modelLabel) : character.model_label, snap.voiceLabel || character.voice_label].filter(Boolean).join(" · ") || "本地角色语音"}
          </p>
        </div>
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground" aria-hidden>
          {snap.state === VoiceState.SPEAKING && <SpeakingBars />}
          {snap.conn === "connected" ? "在线" : snap.conn === "connecting" ? "正在连接…" : "未连接"}
        </span>
      </div>

      {/* conversation */}
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
        <ol className="mx-auto flex max-w-3xl flex-col gap-5 px-4 pt-2 pb-6" role="log" aria-label="对话记录">
          {snap.messages.length === 0 && (
            <li className="py-16 text-center text-sm text-muted-foreground">
              {snap.conn === "connecting" ? <span className="inline-flex items-center gap-2"><Loader2 className="size-4 animate-spin" aria-hidden />正在连接本地 AI…</span>
                : `和 ${name} 打个招呼吧：点击麦克风说话，或者直接输入文字。`}
            </li>
          )}
          {snap.messages.map((m) => m.role === "note" ? (
            <li key={m.id} className="self-center text-xs text-muted-foreground">{m.text}</li>
          ) : m.role === "user" ? (
            <li key={m.id} className="flex flex-col items-end gap-1">
              <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary/10 px-4 py-2.5 leading-relaxed whitespace-pre-wrap break-words">
                <span className="sr-only">你：</span>{m.text}
              </div>
              {m.voice && <span className="flex items-center gap-1 text-xs text-muted-foreground"><Mic className="size-3" aria-hidden />语音输入</span>}
            </li>
          ) : (
            <li key={m.id} className="flex gap-3">
              <Avatar name={name} url={character.avatar_url} size={32} />
              <div className="flex min-w-0 max-w-[80%] flex-col items-start gap-1">
                <div className="rounded-2xl rounded-tl-md border bg-card px-4 py-2.5 leading-relaxed whitespace-pre-wrap break-words">
                  <span className="sr-only">{name}：</span>
                  {m.typing && !m.text ? <span className="flex gap-1 py-1.5" aria-label="正在输入">{[0, 1, 2].map((i) => <span key={i} className="size-1.5 animate-pulse rounded-full bg-muted-foreground/60" style={{ animationDelay: `${i * 0.2}s` }} />)}</span> : m.text}
                </div>
                {m.hasAudio && (
                  <Button variant="ghost" size="xs" onClick={() => engine.replay(m.id)} aria-label="播放这条语音">
                    <Volume2 aria-hidden />播放语音
                  </Button>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* status + composer */}
      <div className="mx-auto w-full max-w-3xl px-4 pb-4">
        <div className={cn("mb-2 flex min-h-11 items-center gap-3 rounded-xl px-3 py-1.5 transition-colors",
                           listening ? "bg-listen/10" : "bg-muted")}>
          <div role="status" aria-live="polite" aria-atomic="true" className={cn("flex min-w-0 flex-1 items-center gap-2.5", status.tone)}>
            <StatusIcon icon={status.icon} />
            <span className="text-sm font-medium whitespace-nowrap">{status.label}</span>
            {status.hint && <span className="hidden truncate text-sm text-muted-foreground sm:inline">{status.hint}</span>}
          </div>
          {listening && <LevelMeter engine={engine} />}
          {snap.state === VoiceState.SPEAKING && <SpeakingBars />}
          {BUSY.includes(snap.state) && (
            <Button size="xs" variant="outline" onClick={() => engine.interrupt()} aria-label="停止回复"><Square aria-hidden />停止</Button>
          )}
          <span className={cn("flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs",
                              snap.mic === "on" ? "bg-listen text-white" : "bg-background text-muted-foreground")}>
            {snap.mic === "on" ? <Mic className="size-3.5" aria-hidden /> : <MicOff className="size-3.5" aria-hidden />}
            <span className="hidden sm:inline" data-testid="mic-chip">{snap.mic === "on" ? "麦克风开启" : snap.mic === "pending" ? "等待授权" : "麦克风已关闭"}</span>
          </span>
        </div>

        {snap.error && (
          <div role="alert" className="mb-2 flex items-center gap-3 rounded-xl border border-bad/30 bg-bad/5 px-3 py-2 text-sm">
            <span className="flex-1">{snap.error.text}</span>
            {snap.error.canRetry && <Button size="xs" variant="outline" onClick={() => engine.retry()}><RotateCcw aria-hidden />{snap.error.retryLabel ?? "重试"}</Button>}
            <Button size="icon-xs" variant="ghost" onClick={() => engine.clearError()} aria-label="关闭提示"><X aria-hidden /></Button>
          </div>
        )}

        <form className="flex items-end gap-2 rounded-3xl border bg-card p-2 pl-4 shadow-sm focus-within:border-ring"
              onSubmit={(e) => { e.preventDefault(); send(); }}>
          <label htmlFor="composer" className="sr-only">输入消息</label>
          <textarea id="composer" rows={1} value={text} onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); } }}
                    placeholder="输入消息…" className="max-h-40 min-h-10 flex-1 resize-none bg-transparent py-2 outline-none field-sizing-content" />
          {typed && !listening ? (
            <Button type="submit" size="icon-lg" className="rounded-full" disabled={!ready} aria-label="发送消息"><ArrowUp aria-hidden /></Button>
          ) : (
            <Button type="button" size="lg" disabled={!ready || snap.state === VoiceState.REQUESTING_MIC_PERMISSION || snap.state === VoiceState.TRANSCRIBING}
                    variant={listening ? "default" : "secondary"} aria-pressed={listening}
                    aria-label={listening ? "完成说话，开始识别" : BUSY.includes(snap.state) ? "打断并开始说话" : "开始说话"}
                    onClick={() => (listening ? void engine.finishListening() : void engine.startListening())}
                    className={cn("rounded-full", listening ? "animate-cv-pulse bg-listen px-4 text-white hover:bg-listen/90" : "size-9 px-0")}>
              {snap.state === VoiceState.REQUESTING_MIC_PERMISSION ? <Loader2 className="animate-spin" aria-hidden />
                : listening ? <><Square aria-hidden />完成</> : <Mic aria-hidden />}
            </Button>
          )}
        </form>
        <p className="mt-1.5 hidden text-center text-xs text-muted-foreground sm:block">Enter 发送 · Shift+Enter 换行 · Esc 停止</p>
      </div>
    </div>
  );
}
