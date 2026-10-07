"use client";

import { CheckCircle2, FileAudio, Loader2, UploadCloud, XCircle } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Progress } from "@/components/ui/progress";
import { errorText, uploadFiles } from "@/lib/api";
import { formatBytes, formatDuration } from "@/lib/format";
import type { VoicePack } from "@/lib/types";
import { cn } from "@/lib/utils";

const ACCEPT = ".wav,.mp3,.flac,.m4a,.ogg,.opus,.aac,.zip,.lab,.txt";

type Item = {
  id: string;
  file: File;
  state: "queued" | "uploading" | "done" | "rejected";
  progress: number;
  note?: string;
};

interface UploadResponse {
  accepted: { name: string; kind: string; duration_s?: number }[];
  rejected: { name: string; reason: string }[];
  pack: VoicePack;
}

export function Uploader({ pack, onUploaded }: { pack: VoicePack; onUploaded: () => void }) {
  const [items, setItems] = useState<Item[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const update = (id: string, patch: Partial<Item>) =>
    setItems((list) => list.map((item) => (item.id === id ? { ...item, ...patch } : item)));

  async function add(files: FileList | File[]) {
    const fresh: Item[] = Array.from(files).map((file) => ({
      id: `${file.name}-${file.size}-${Math.random().toString(36).slice(2)}`, file, state: "queued", progress: 0,
    }));
    if (!fresh.length) return;
    setItems((list) => [...fresh, ...list]);
    setBusy(true);
    let accepted = 0;
    let rejected = 0;
    // One file per request: honest per-file progress, and one bad file never sinks a batch.
    for (const item of fresh) {
      update(item.id, { state: "uploading" });
      try {
        const result = await uploadFiles<UploadResponse>(
          `/api/voice-packs/${pack.id}/upload`, [item.file], "files",
          (fraction) => update(item.id, { progress: fraction }));
        accepted += result.accepted.length;
        rejected += result.rejected.length;
        const isZip = item.file.name.toLowerCase().endsWith(".zip");
        if (result.rejected.length && !result.accepted.length) {
          update(item.id, { state: "rejected", progress: 1, note: result.rejected[0].reason });
        } else {
          update(item.id, {
            state: "done", progress: 1,
            note: isZip ? `解压出 ${result.accepted.length} 个文件${result.rejected.length ? `，${result.rejected.length} 个被拒绝` : ""}`
                        : result.accepted[0]?.duration_s ? formatDuration(result.accepted[0].duration_s) : "已上传",
          });
        }
      } catch (e) {
        rejected += 1;
        update(item.id, { state: "rejected", progress: 1, note: errorText(e).message });
      }
      onUploaded();
    }
    setBusy(false);
    if (accepted) toast.success(`上传完成：${accepted} 个文件${rejected ? `，${rejected} 个被拒绝` : ""}`);
    else if (rejected) toast.error("所有文件都被拒绝了");
  }

  return (
    <div className="space-y-4">
      <div
        role="button" tabIndex={0} aria-label="上传语音文件：点击选择，或把文件拖到这里"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.current?.click(); } }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); void add(e.dataTransfer.files); }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none",
          dragging ? "border-primary bg-primary/5" : "hover:border-primary/40 hover:bg-muted/40",
        )}
      >
        <UploadCloud className="size-8 text-muted-foreground" aria-hidden />
        <p className="font-medium">把角色的中文语音拖到这里，或点击选择</p>
        <p className="text-sm text-muted-foreground">支持 WAV、MP3、FLAC、M4A，以及打包好的 ZIP。可附带同名 .lab / .txt 文字稿。</p>
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden"
               onChange={(e) => { if (e.target.files) void add(e.target.files); e.target.value = ""; }} />
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted-foreground" aria-live="polite">
        <span>已收到 <b className="text-foreground tabular-nums">{pack.total_files}</b> 个音频</span>
        <span>总时长 <b className="text-foreground">{formatDuration(pack.total_duration_s)}</b></span>
        <span>大小 <b className="text-foreground">{formatBytes(pack.total_bytes)}</b></span>
        {busy && <span className="flex items-center gap-1.5"><Loader2 className="size-3.5 animate-spin" aria-hidden />正在上传…</span>}
      </div>

      {items.length > 0 && (
        <ul className="max-h-72 divide-y overflow-y-auto rounded-xl border" aria-label="上传列表">
          {items.slice(0, 200).map((item) => (
            <li key={item.id} className="flex items-center gap-3 px-3 py-2 text-sm">
              {item.state === "done" ? <CheckCircle2 className="size-4 text-ok" aria-label="已接收" />
                : item.state === "rejected" ? <XCircle className="size-4 text-bad" aria-label="被拒绝" />
                : item.state === "uploading" ? <Loader2 className="size-4 animate-spin text-info" aria-label="上传中" />
                : <FileAudio className="size-4 text-muted-foreground" aria-label="等待上传" />}
              <span className="min-w-0 flex-1 truncate">{item.file.name}</span>
              {item.state === "uploading"
                ? <Progress value={Math.round(item.progress * 100)} className="w-24" aria-label={`${item.file.name} 上传进度`} />
                : <span className={cn("shrink-0 text-xs", item.state === "rejected" ? "text-bad" : "text-muted-foreground")}>
                    {item.note ?? formatBytes(item.file.size)}
                  </span>}
            </li>
          ))}
        </ul>
      )}

      {pack.rejected_files.length > 0 && items.length === 0 && (
        <p className="text-xs text-muted-foreground">之前有 {pack.rejected_files.length} 个文件被拒绝（格式不支持或无法解码）。</p>
      )}
      {items.length === 0 && pack.total_files === 0 && (
        <p className="text-xs text-muted-foreground">建议至少 10 分钟清晰的单人语音；游戏语音包这类“一句一个文件 + 文字稿”的数据效果最好。</p>
      )}
      <span className="sr-only" aria-live="polite">{busy ? "正在上传" : ""}</span>
    </div>
  );
}
