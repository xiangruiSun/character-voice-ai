"use client";

import { CheckCircle2, Loader2, Wand2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { ErrorAlert } from "@/components/shared/blocks";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { errorText } from "@/lib/api";
import { styleLabel } from "@/lib/format";
import { useTestSentences, useVoiceModelMutations } from "@/lib/queries";
import type { SynthesisResult, VoiceModel } from "@/lib/types";

type Result = SynthesisResult & { text: string; at: number };

export function VoiceTester({ model, showApprove = true }: { model: VoiceModel; showApprove?: boolean }) {
  const sentences = useTestSentences();
  const m = useVoiceModelMutations();
  const [text, setText] = useState("");
  const [style, setStyle] = useState(model.styles.includes("neutral") ? "neutral" : model.styles[0] ?? "neutral");
  const [results, setResults] = useState<Result[]>([]);
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);
  const styleItems = (model.styles.length ? model.styles : ["neutral"]).map((s) => ({ value: s, label: styleLabel(s) }));

  async function generate(sentence = text, sentenceStyle = style) {
    if (!sentence.trim()) return;
    setError(null);
    const usable = model.styles.includes(sentenceStyle) ? sentenceStyle : style;
    try {
      const result = await m.synthesize.mutateAsync({ id: model.id, text: sentence.trim(), style: usable });
      setResults((r) => [{ ...result, text: sentence.trim(), at: Date.now() }, ...r].slice(0, 12));
    } catch (e) { setError(errorText(e)); }
  }

  return (
    <div className="space-y-5">
      {model.evaluation.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">训练后自动生成的试听</p>
          <ul className="grid gap-2 md:grid-cols-3">
            {model.evaluation.map((e) => (
              <li key={e.audio_url} className="space-y-2 rounded-xl border p-3 text-sm">
                <p>{e.text}</p>
                <audio controls preload="none" src={e.audio_url} className="h-8 w-full" />
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="space-y-3">
        <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} maxLength={300}
                  placeholder="输入一句中文，测试角色的声音……" aria-label="测试文字"
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void generate(); }} />
        <div className="flex flex-wrap items-center gap-2">
          <Select items={styleItems} value={style} onValueChange={(v) => v && setStyle(v)}>
            <SelectTrigger className="w-28" aria-label="语气风格"><SelectValue /></SelectTrigger>
            <SelectContent>{styleItems.map((s) => <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>)}</SelectContent>
          </Select>
          <Button onClick={() => generate()} disabled={!text.trim() || m.synthesize.isPending}>
            {m.synthesize.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Wand2 aria-hidden />}
            {m.synthesize.isPending ? "正在生成…" : "生成"}
          </Button>
        </div>
        <div className="flex flex-wrap gap-2" aria-label="示例句子">
          {sentences.data?.map((s) => (
            <Button key={s.style} size="sm" variant="secondary" disabled={m.synthesize.isPending}
                    onClick={() => { setText(s.text); void generate(s.text, s.style); }}>
              {s.label}：{s.text.slice(0, 10)}…
            </Button>
          ))}
        </div>
      </div>

      {error && <ErrorAlert title={error.message} hint={error.hint} onRetry={() => generate()} />}

      {results.length > 0 && (
        <ul className="space-y-2" aria-label="生成结果" aria-live="polite">
          {results.map((r, i) => (
            <li key={r.at} className="space-y-2 rounded-xl border p-3">
              <p className="text-sm">{r.text}</p>
              <audio controls autoPlay={i === 0} src={r.audio_url} className="h-9 w-full" />
              <p className="text-xs text-muted-foreground">
                {styleLabel(r.style)} · 生成用时 {(r.generation_ms / 1000).toFixed(1)} 秒 · 音频 {r.duration_s} 秒 · {model.name}
              </p>
            </li>
          ))}
        </ul>
      )}

      {showApprove && (
        <ApproveVoice model={model} />
      )}
    </div>
  );
}

function ApproveVoice({ model }: { model: VoiceModel }) {
  const m = useVoiceModelMutations();
  if (model.approved) {
    return <p className="flex items-center gap-2 text-sm text-ok"><CheckCircle2 className="size-4" aria-hidden />这个声音已通过审核，可以用来创建角色。</p>;
  }
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-muted/30 p-4">
      <p className="flex-1 text-sm">听起来满意吗？通过审核后，就可以用这个声音创建角色。</p>
      <Button onClick={async () => {
        try { await m.update.mutateAsync({ id: model.id, patch: { approved: true } }); toast.success("声音已通过审核"); }
        catch (e) { toast.error(errorText(e).message); }
      }} disabled={m.update.isPending}>
        通过审核
      </Button>
    </div>
  );
}
