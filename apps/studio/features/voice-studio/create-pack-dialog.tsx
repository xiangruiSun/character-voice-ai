"use client";

import { Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ErrorAlert } from "@/components/shared/blocks";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { errorText } from "@/lib/api";
import { useVoicePackMutations } from "@/lib/queries";

const LANGUAGES = [{ value: "zh-CN", label: "中文" }];

export function CreatePackDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const router = useRouter();
  const m = useVoicePackMutations();
  const [character, setCharacter] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [source, setSource] = useState("");
  const [error, setError] = useState<{ message: string; hint: string | null } | null>(null);

  async function submit() {
    setError(null);
    try {
      const pack = await m.create.mutateAsync({
        name: name.trim() || `${character.trim()} 声音包`, character_name: character.trim(),
        language: "zh-CN", description, source,
      });
      onOpenChange(false);
      router.push(`/voice-studio/pack?id=${pack.id}`);
    } catch (e) { setError(errorText(e)); }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>创建声音</DialogTitle>
          <DialogDescription>先建一个声音包，接下来上传这个角色的语音。</DialogDescription>
        </DialogHeader>
        <form className="grid gap-4" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
          <div className="grid gap-2">
            <Label htmlFor="vp-character">角色名称</Label>
            <Input id="vp-character" value={character} onChange={(e) => setCharacter(e.target.value)} placeholder="例如：卡提希娅" required maxLength={100} autoFocus />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="vp-name">声音包名称</Label>
            <Input id="vp-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={character ? `${character} 声音包` : "可选"} maxLength={100} />
          </div>
          <div className="grid gap-2">
            <Label>语言</Label>
            <Select items={LANGUAGES} value="zh-CN" disabled>
              <SelectTrigger className="w-full" aria-label="语言"><SelectValue /></SelectTrigger>
              <SelectContent>{LANGUAGES.map((l) => <SelectItem key={l.value} value={l.value}>{l.label}</SelectItem>)}</SelectContent>
            </Select>
            <span className="text-xs text-muted-foreground">目前的训练流程针对中文角色语音。</span>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="vp-source">来源（可选）</Label>
            <Input id="vp-source" value={source} onChange={(e) => setSource(e.target.value)} placeholder="例如：游戏语音、自己录制" maxLength={500} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="vp-desc">备注（可选）</Label>
            <Textarea id="vp-desc" value={description} onChange={(e) => setDescription(e.target.value)} rows={2} maxLength={2000} />
          </div>
          {error && <ErrorAlert title={error.message} hint={error.hint} />}
          <DialogFooter>
            <Button type="submit" disabled={!character.trim() || m.create.isPending}>
              {m.create.isPending && <Loader2 className="animate-spin" aria-hidden />}创建并上传语音
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
