import type { TrainingJob, VoiceModel, VoicePack } from "@/lib/types";

export const STEPS = ["上传", "检查", "准备", "训练", "试听", "创建角色"];
export const ACTIVE = ["queued", "validating", "preparing", "training", "evaluating", "cancel_requested"];

/** Where the user is, from the server's state — so returning later resumes correctly. */
export function stepFor(pack: VoicePack, job?: TrainingJob, model?: VoiceModel): number {
  if (model?.approved) return 5;
  if (job?.status === "completed" && model) return 4;
  if (job && (ACTIVE.includes(job.status) || job.status === "failed")) return 3;
  if (pack.status === "ready") return 3;
  if (pack.status === "needs_review" || pack.status === "processing" || pack.status === "error") return 1;
  return 0;
}
