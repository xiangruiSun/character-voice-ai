"use client";

// Server state lives in TanStack Query; components read it through these hooks only.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";
import type {
  AudioSample,
  Character,
  CharacterInput,
  ConnectionDraft,
  HealthCheck,
  ModelConnection,
  Overview,
  ProbeResult,
  SynthesisResult,
  TrainingEngineOptions,
  TrainingJob,
  VoiceModel,
  VoicePack,
} from "./types";

export const keys = {
  overview: ["overview"] as const,
  health: ["health"] as const,
  connections: ["model-connections"] as const,
  packs: ["voice-packs"] as const,
  pack: (id: string) => ["voice-packs", id] as const,
  samples: (id: string) => ["voice-packs", id, "samples"] as const,
  jobs: ["training-jobs"] as const,
  job: (id: string) => ["training-jobs", id] as const,
  trainingOptions: ["training-options"] as const,
  voices: ["voice-models"] as const,
  characters: ["characters"] as const,
  character: (id: string) => ["characters", id] as const,
  testSentences: ["test-sentences"] as const,
};

// -- reads ----------------------------------------------------------------------------

export const useOverview = () =>
  useQuery({ queryKey: keys.overview, queryFn: () => api.get<Overview>("/api/system/overview") });

export const useHealth = (refetchMs = 15000) =>
  useQuery({
    queryKey: keys.health,
    queryFn: () => api.get<HealthCheck>("/api/system/health"),
    refetchInterval: refetchMs,
  });

export const useConnections = () =>
  useQuery({ queryKey: keys.connections, queryFn: () => api.get<ModelConnection[]>("/api/model-connections") });

export const useVoicePacks = () =>
  useQuery({
    queryKey: keys.packs,
    queryFn: () => api.get<VoicePack[]>("/api/voice-packs"),
    refetchInterval: (q) => (q.state.data?.some((p) => p.status === "processing") ? 2000 : false),
  });

export const useVoicePack = (id: string | null) =>
  useQuery({
    queryKey: keys.pack(id ?? ""),
    queryFn: () => api.get<VoicePack>(`/api/voice-packs/${id}`),
    enabled: !!id,
    // Preparation runs in the worker; watch it while it does.
    refetchInterval: (q) => (q.state.data?.status === "processing" ? 1500 : false),
  });

export const useSamples = (id: string | null, enabled = true) =>
  useQuery({
    queryKey: keys.samples(id ?? ""),
    queryFn: () => api.get<AudioSample[]>(`/api/voice-packs/${id}/samples`),
    enabled: !!id && enabled,
  });

export const useTrainingJobs = (packId?: string) =>
  useQuery({
    queryKey: packId ? [...keys.jobs, { packId }] : keys.jobs,
    queryFn: () => api.get<TrainingJob[]>(`/api/training-jobs${packId ? `?voice_pack_id=${packId}` : ""}`),
    refetchInterval: (q) =>
      q.state.data?.some((j) => !["completed", "failed", "cancelled"].includes(j.status)) ? 3000 : false,
  });

export const useTrainingOptions = () =>
  useQuery({ queryKey: keys.trainingOptions, queryFn: () => api.get<TrainingEngineOptions[]>("/api/training-jobs/options") });

export const useVoiceModels = () =>
  useQuery({ queryKey: keys.voices, queryFn: () => api.get<VoiceModel[]>("/api/voice-models") });

export const useCharacters = () =>
  useQuery({ queryKey: keys.characters, queryFn: () => api.get<Character[]>("/api/characters") });

export const useTestSentences = () =>
  useQuery({
    queryKey: keys.testSentences,
    queryFn: () => api.get<{ style: string; label: string; text: string }[]>("/api/voice-models/test-sentences"),
    staleTime: Infinity,
  });

// -- writes ---------------------------------------------------------------------------

function useInvalidate() {
  const client = useQueryClient();
  return (...queryKeys: readonly (readonly unknown[])[]) =>
    Promise.all([...queryKeys, keys.overview].map((k) => client.invalidateQueries({ queryKey: k })));
}

export function useConnectionMutations() {
  const invalidate = useInvalidate();
  const done = () => invalidate(keys.connections, keys.characters);
  return {
    create: useMutation({
      mutationFn: (draft: ConnectionDraft) => api.post<ModelConnection>("/api/model-connections", draft),
      onSuccess: done,
    }),
    update: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: Partial<ConnectionDraft> & { disabled?: boolean } }) =>
        api.patch<ModelConnection>(`/api/model-connections/${id}`, patch),
      onSuccess: done,
    }),
    remove: useMutation({ mutationFn: (id: string) => api.delete(`/api/model-connections/${id}`), onSuccess: done }),
    test: useMutation({
      mutationFn: (id: string) => api.post<ModelConnection>(`/api/model-connections/${id}/test`),
      onSuccess: done,
    }),
    probe: useMutation({ mutationFn: (draft: ConnectionDraft) => api.post<ProbeResult>("/api/model-connections/test", draft) }),
    setDefault: useMutation({
      mutationFn: (id: string) => api.post<ModelConnection>(`/api/model-connections/${id}/default`),
      onSuccess: done,
    }),
  };
}

export function useVoicePackMutations(packId?: string) {
  const invalidate = useInvalidate();
  const done = () => invalidate(keys.packs, ...(packId ? [keys.pack(packId), keys.samples(packId)] : []));
  return {
    create: useMutation({
      mutationFn: (data: { name: string; character_name: string; language: string; description?: string; source?: string; notes?: string }) =>
        api.post<VoicePack>("/api/voice-packs", data),
      onSuccess: done,
    }),
    prepare: useMutation({ mutationFn: (id: string) => api.post<VoicePack>(`/api/voice-packs/${id}/prepare`), onSuccess: done }),
    confirmRights: useMutation({ mutationFn: (id: string) => api.post<VoicePack>(`/api/voice-packs/${id}/rights`), onSuccess: done }),
    approve: useMutation({ mutationFn: (id: string) => api.post<VoicePack>(`/api/voice-packs/${id}/approve`), onSuccess: done }),
    remove: useMutation({ mutationFn: (id: string) => api.delete(`/api/voice-packs/${id}`), onSuccess: done }),
    updateSample: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: Partial<Pick<AudioSample, "transcript" | "style" | "approved">> }) =>
        api.patch<AudioSample>(`/api/audio-samples/${id}`, patch),
      onSuccess: () => invalidate(...(packId ? [keys.pack(packId)] : [])),
    }),
  };
}

export function useTrainingMutations() {
  const invalidate = useInvalidate();
  const done = () => invalidate(keys.jobs, keys.packs, keys.voices);
  return {
    start: useMutation({
      mutationFn: (body: { voice_pack_id: string; preset: string; advanced: Record<string, number> }) =>
        api.post<TrainingJob>("/api/training-jobs", body),
      onSuccess: done,
    }),
    cancel: useMutation({ mutationFn: (id: string) => api.post<TrainingJob>(`/api/training-jobs/${id}/cancel`), onSuccess: done }),
    retry: useMutation({ mutationFn: (id: string) => api.post<TrainingJob>(`/api/training-jobs/${id}/retry`, {}), onSuccess: done }),
  };
}

export function useVoiceModelMutations() {
  const invalidate = useInvalidate();
  const done = () => invalidate(keys.voices, keys.characters);
  return {
    update: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: { name?: string; approved?: boolean; archived?: boolean } }) =>
        api.patch<VoiceModel>(`/api/voice-models/${id}`, patch),
      onSuccess: done,
    }),
    synthesize: useMutation({
      mutationFn: ({ id, text, style }: { id: string; text: string; style: string }) =>
        api.post<SynthesisResult>(`/api/voice-models/${id}/synthesize`, { text, style }),
    }),
  };
}

export function useCharacterMutations() {
  const invalidate = useInvalidate();
  const done = () => invalidate(keys.characters);
  return {
    create: useMutation({ mutationFn: (data: CharacterInput) => api.post<Character>("/api/characters", data), onSuccess: done }),
    update: useMutation({
      mutationFn: ({ id, patch }: { id: string; patch: Partial<CharacterInput> & { use_default_model?: boolean } }) =>
        api.patch<Character>(`/api/characters/${id}`, patch),
      onSuccess: done,
    }),
    remove: useMutation({ mutationFn: (id: string) => api.delete(`/api/characters/${id}`), onSuccess: done }),
    preview: useMutation({
      mutationFn: (body: { text: string; name: string; system_prompt: string; model_connection_id: string | null;
                          voice_model_id: string | null; voice_style: string }) =>
        api.post<{ reply: string; llm_ms: number; audio_url?: string; tts_ms?: number }>("/api/characters/preview", body),
    }),
  };
}
