// Contracts with the FastAPI backend (mirrors cvai_studio/api/schemas.py).

export type ConnectionStatus = "draft" | "testing" | "connected" | "error" | "disabled";
export type ProviderType = "ollama" | "openai_compatible";
export type ConnectionKind = "local" | "public_api" | "custom";

export interface GenerationDefaults {
  temperature: number;
  max_tokens: number;
  stream: boolean;
}

export interface ModelConnection {
  id: string;
  name: string;
  kind: ConnectionKind;
  provider_type: ProviderType;
  connection_mode: string;
  base_url: string;
  model_name: string;
  has_api_key: boolean;
  api_key_hint: string | null;
  status: ConnectionStatus;
  last_error: string | null;
  latency_ms: number | null;
  last_tested_at: string | null;
  is_default: boolean;
  generation_defaults: GenerationDefaults;
  extra: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface ConnectionDraft {
  name: string;
  kind: ConnectionKind;
  provider_type: ProviderType;
  base_url: string;
  model_name: string;
  api_key?: string | null;
  generation_defaults?: GenerationDefaults;
  extra?: Record<string, string>;
}

export interface ProbeResult {
  ok: boolean;
  latency_ms: number | null;
  message: string;
  hint: string | null;
}

export type VoicePackStatus = "empty" | "uploading" | "processing" | "needs_review" | "ready" | "error";

export interface DatasetSummary {
  uploaded_s: number;
  usable_s: number;
  rejected_s: number;
  clips: number;
  approved_clips: number;
  potential_issues: { issue: string; count: number }[];
  ready_to_train: boolean;
  requirements: { min_clips: number; min_seconds: number };
}

export interface VoicePack {
  id: string;
  name: string;
  character_name: string;
  language: string;
  description: string;
  source: string;
  notes: string;
  status: VoicePackStatus;
  rights_confirmed: boolean;
  total_files: number;
  total_duration_s: number;
  usable_duration_s: number;
  total_bytes: number;
  rejected_files: { name: string; reason: string }[];
  progress: { stage?: string; label?: string };
  last_error: string | null;
  created_at: string;
  updated_at: string;
  summary: DatasetSummary | null;
}

export interface AudioSample {
  id: string;
  audio_url: string;
  source_file: string;
  transcript: string;
  transcript_source: string;
  duration_s: number;
  quality_score: number | null;
  issues: string[];
  emotion: string | null;
  style: string;
  approved: boolean;
  rejection_reason: string | null;
  human_edited: boolean;
}

export type TrainingStatus =
  | "queued" | "validating" | "preparing" | "training" | "evaluating"
  | "completed" | "failed" | "cancel_requested" | "cancelled";

export interface TrainingEvent {
  type: string;
  at: string;
  [key: string]: unknown;
}

export interface TrainingJob {
  id: string;
  voice_pack_id: string;
  voice_pack_name: string | null;
  engine: string;
  base_model: string;
  preset: string;
  status: TrainingStatus;
  current_stage: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  config: Record<string, number>;
  events: TrainingEvent[];
  metrics: Record<string, number>;
  error_message: string | null;
  error_hint: string | null;
  error_detail: string | null;
  voice_model_id: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface TrainingPreset {
  key: string;
  label: string;
  description: string;
  params: Record<string, number>;
}

export interface TrainingAdvancedOption {
  key: string;
  label: string;
  kind: "int" | "float";
  default: number;
  minimum: number;
  maximum: number;
  hint: string;
}

export interface TrainingEngineOptions {
  engine: string;
  label: string;
  base_model: string;
  available: boolean;
  unavailable_reason: string | null;
  presets: TrainingPreset[];
  advanced: TrainingAdvancedOption[];
}

export interface VoiceModel {
  id: string;
  name: string;
  voice_pack_id: string | null;
  training_job_id: string | null;
  engine: string;
  base_model: string;
  version: number;
  status: "ready" | "invalid" | "archived";
  approved: boolean;
  styles: string[];
  evaluation: { text: string; audio_url: string }[];
  created_at: string;
}

export interface SynthesisResult {
  audio_url: string;
  duration_s: number;
  generation_ms: number;
  voice_model_id: string;
  style: string;
}

export interface Character {
  id: string;
  name: string;
  description: string;
  avatar_url: string | null;
  system_prompt: string;
  greeting: string;
  model_connection_id: string | null;
  uses_default_model: boolean;
  model_label: string | null;
  voice_model_id: string | null;
  voice_label: string | null;
  default_voice_style: string;
  status: "ready" | "incomplete";
  problems: string[];
  created_at: string;
  updated_at: string;
}

export interface CharacterInput {
  name: string;
  description: string;
  system_prompt: string;
  greeting: string;
  model_connection_id: string | null;
  voice_model_id: string | null;
  default_voice_style: string;
}

export interface HealthCheck {
  ok: boolean;
  checks: Record<string, { ok: boolean; detail?: string }>;
}

export interface Overview {
  model_connections: number;
  voice_packs: number;
  voice_models: number;
  characters: number;
}
