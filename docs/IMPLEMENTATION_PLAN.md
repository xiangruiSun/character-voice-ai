# Implementation Plan — Character Voice AI (Chinese, V1)

The product question this repo must answer, in order:

1. **Which TTS adaptation approach reproduces one specific Chinese game character most
   faithfully?** (Milestones 1–7 — a research question, answered with a benchmark.)
2. **Can we wrap the winner in a conversation loop that still sounds like the character?**
   (Milestones 8–13 — an engineering question.)
3. Only then: more characters (Milestone 14).

Everything below is sequenced so that no milestone requires rewriting an earlier one.
The interfaces defined in Milestone 1 are the contract that makes that true.

---

## Architectural decisions fixed in Milestone 1

These are the decisions that are expensive to change later, so they are made now and
justified from `docs/TECH_LANDSCAPE.md`.

| # | Decision | Reason |
|---|---|---|
| D1 | **Every TTS engine runs in its own process; adapters speak HTTP/CLI to a sidecar.** Never `import` an engine into the API process. | The six candidate engines have mutually incompatible torch/CUDA pins. Also lets us hot-swap character checkpoints via endpoints like GPT-SoVITS `/set_gpt_weights` instead of reloading from disk per sentence. |
| D2 | **`TTSRequest` carries engine-neutral `StyleControls` + a reference selection; each adapter maps them onto native knobs, and declares what it supports via `ProviderCapabilities`.** | Engines express style completely differently: GPT-SoVITS via reference clip only, IndexTTS-2.5 via an 8-dim emotion vector + `emo_alpha` + `duration_factor`, Qwen3-TTS/CosyVoice3 via natural-language `instruct`. A lowest-common-denominator interface would throw away the strongest character-control features. |
| D3 | **Reference Bank entries store the transcript, not just the audio path.** | Qwen3-TTS (`ref_text`), Fish Speech (`--prompt-text`) and VoxCPM2 require it; GPT-SoVITS (`prompt_text`) benefits strongly. Retrofitting transcripts onto an existing bank means re-listening to every clip. |
| D4 | **Benchmark results are keyed by `(engine, adaptation_mode)` where mode ∈ {zero_shot, finetuned, lora}.** | Comparing a fine-tuned engine to a zero-shot one picks the wrong winner. The spec's thesis is that adaptation beats embedding-only cloning; the benchmark has to be able to *show* that rather than assume it. |
| D5 | **Every generated sample is accompanied by a `GenerationRecord` capturing engine version, checkpoint id, voicepack version, reference clip id, all sampling params and the seed.** | Spec §23. A voice experiment you cannot reproduce is an anecdote. |
| D6 | **`zh-CN` is enforced in schema validation, not by convention.** | Spec §25 non-goals. A hard validation error is the cheapest way to stop multilingual scope creep. |
| D7 | **Optional preprocessing stages (separation, dereverb, denoise) are off by default and each clip records its `processing_chain`.** | Spec §7: over-processing destroys character identity. Recording the chain makes "did cleaning help?" an answerable question instead of a belief. |
| D8 | **Human evaluation decides the winner; objective metrics rank and regression-test.** | Reference-free MOS predictors degrade precisely in the high-quality regime we operate in. Spec §18 already mandates this; D8 records *why*. |

---

## Milestone 1 — Repository, interfaces, config, schemas ✅ (this milestone)

**Components added**

* Monorepo layout (spec §22) with multi-root Python packaging.
* `cvai_types` — every typed schema in the system: style taxonomy, `CharacterProfile`,
  `VoicePackManifest` / `TrainingSample` / `ReferenceSample` / `ReferenceBank`,
  `CharacterSpeechPlan`, `TTSRequest`/`TTSResult`, `StyleControls`,
  `ProviderCapabilities`, conversation state machine, evaluation schemas.
* `cvai_core` — layered YAML config with env-var overrides and `${VAR}` interpolation,
  provider ABCs (`SpeechToTextProvider`, `LLMProvider`, `TTSProvider`,
  `CharacterProvider`, `ReferenceRetriever`, `ConversationSession`), a provider registry
  and factory, and the reproducibility run-manifest writer.
* `cvai_reference_retrieval` — metadata-rule retriever with style fallback and
  deterministic rotation (so the same style doesn't always return clip 01).
* Provider adapters: a runnable `MockTTSProvider`, plus capability-complete stubs for
  GPT-SoVITS / Qwen3-TTS / Fish Speech / IndexTTS / CosyVoice / VoxCPM and OpenAI STT/LLM.
* `cvai_evaluation` — the benchmark pipeline: test-sentence set, runner, blind assignment,
  rating aggregation, Markdown report.
* Voice Pack scaffolder + validator; Denia voice pack skeleton; character profile + JSON schema.
* pytest suite covering schemas, config, registry, retriever, benchmark, scoring.

**How to run:** see `README.md` and `docs/BENCHMARK.md`.

**Definition of done:** `make test` green; `make demo` produces a full benchmark run
(mock engine), a blind test set and an aggregated report without touching a GPU.

---

## Milestone 2 — Voice Pack preprocessing pipeline ✅

**Components:** `cvai_voice_preprocessing` — resumable stage pipeline, pluggable
backends, human review loop, dataset and Reference Bank build. Entry point `cvai-prep`.

1. `ingest` — game assets (AnimeWwise / vgmstream / AssetStudio) or plain audio → `raw/`,
   checksummed, never mutated afterwards.
2. `separate` *(optional)* — `python-audio-separator`, BS-Roformer for vocals, VR-arch for
   de-reverb. Skipped when the source is already dry dialogue.
3. `denoise` *(optional, off by default)* — DeepFilterNet3, conservative settings.
4. `segment` — FSMN-VAD / silero-vad with padding that **keeps leading breaths**.
5. `transcribe` — FunASR `paraformer-zh` + `ct-punc`, char-level timestamps.
6. `speaker_filter` — CAM++/ERes2NetV2 cosine against a hand-curated character centroid;
   everything below threshold goes to `rejected/` with the score recorded.
7. `annotate` — SenseVoice / emotion2vec bootstrap for emotion, plus measured
   speaking-rate, F0 profile and pause statistics; writes `TrainingSample` fields.
8. `quality` — loudness (pyloudnorm), clipping, SNR estimate, DNSMOS/UTMOSv2; produces
   `quality_score` used for **review ordering**, not auto-rejection.
9. `build_dataset` / `build_reference_bank` — emit `datasets/` and `references/`.

**Also:** a local review UI (single static page + JSON patch file) so a human can fix
transcripts, re-label styles and approve/reject — spec §8 requires manual correction.

**Built in Milestone 2, beyond the stage list:**

* **Resumable state** (`processed/state.json`, written atomically after every stage).
  ASR over an hour of audio is slow enough that a pipeline which cannot resume gets run
  less often, and a pipeline that is run less often stops matching the data.
* **Human edits are sticky.** Any segment a reviewer touched is marked `human_edited`,
  and no automatic pass overwrites its transcript or style.
* **Backends degrade honestly.** VAD, pitch, pacing, pauses, SNR, clipping, loudness and
  a coarse speaker embedding are implemented in-repo and need no model. Transcription and
  emotion fall back to backends that *report their own absence*; the ASR stub writes
  visible placeholder text and `cvai-prep build` refuses to ship a dataset containing it.
* **Speaker anchors.** The centroid is built from clips a human confirmed
  (`cvai-prep anchors add`). Without anchors the pipeline says so rather than quietly
  averaging every voice in the pack together.
* **Held-out lines never become reference prompts**, so the benchmark's ground truth
  cannot leak into the systems it measures.

**Definition of done:** 20–60 min of approved `clean/` audio for `denia_cn`, a dataset
manifest that validates, a reference bank with ≥2 clips per core style, and a
`processing_chain` recorded per clip. Plus the null-processing control set.

*Status: the pipeline and its tests are complete and run end to end on synthetic audio.
The remaining work is data work — running it over real Denia recordings and reviewing
the result — which needs the audio and the `preprocess` extra.*

---

## Milestone 3 — GPT-SoVITS experiment

Vendored under `models/gpt_sovits/` (submodule or pinned clone), a `Dockerfile` for the
sidecar, a dataset exporter to its `path|speaker|lang|text` list format, a training
runbook (v4 first, v2Pro fallback for rough audio), and the real `GPTSoVITSProvider`
against `api_v2`. Register checkpoints in the voice pack under `checkpoints/`.

## Milestone 4 — Qwen3-TTS experiment

`pip install qwen-tts` sidecar, exporter to the `train_raw.jsonl` → `train_with_codes.jsonl`
format, LoRA/SFT run using the official `finetuning/` scripts (~16 GB VRAM, 10–100 samples
for a first pass, then the full pack), and the real `QwenTTSProvider` with a cached
`create_voice_clone_prompt` per reference clip.

## Milestone 5 — Fish Speech experiment

`.lab`-per-audio dataset export, `extract_vq.py` → `build_dataset.py` → LoRA
`text2semantic_finetune` → `merge_lora.py`, sidecar via `tools/api_server.py`, and the real
`FishSpeechProvider` with pre-extracted prompt tokens for the whole Reference Bank.

*(Optional, flag-gated:)* **M5b** IndexTTS-2.5 (zero-shot + emotion vectors), **M5c**
CosyVoice3 (zero-shot + instruct + SFT), **M5d** VoxCPM2 (LoRA, 48 kHz). These are cheap
once the adapter interface exists and they cover the "explicit emotion control" axis that
GPT-SoVITS cannot express.

## Milestone 6 — Shared evaluation pipeline + human A/B interface

Objective: TTSDS2, SECS (CAM++), CER (paraformer-zh), F0/rate/pause distribution distance
against real lines. Subjective: the blind rating page from M1 fleshed out — 4 axes × 1–5,
hidden real-recording anchor, randomized order, per-rater key file kept separate, plus a
webMUSHRA config exporter. Inter-rater agreement reported.

## Milestone 7 — Select the primary engine

A written decision record in `docs/decisions/` naming the winner, the runner-up, the
licence check, the measured numbers, and the conditions that would reverse the decision.

## Milestone 8 — Character Brain + Speech Planner + text normalization

`CharacterProfile` loading into a system prompt with retrieved original dialogue examples;
the planner emits the structured `CharacterSpeechPlan` (text + emotion + intensity + rate +
volume + pause + ending + reference_style) via constrained JSON; the Chinese normalizer
wraps WeTextProcessing/`wetext` + `cn2an` + `pypinyin`/`g2pW` with a per-character override
lexicon on top.

## Milestone 9 — Text chat application

FastAPI + streaming LLM, conversation state machine, no audio yet.

## Milestone 10 — Character TTS inference in the app

Reference retrieval → chosen engine → audio out. Engine held warm in its sidecar.

## Milestone 11 — Microphone input, STT, voice conversation

Browser capture → VAD → `SpeechToTextProvider` → the existing loop.

## Milestone 12 — Streaming speech chunks

Sentence-boundary chunker on Chinese punctuation with min/max length and semantic
completeness; audio queue; overlap-aware playback.

## Milestone 13 — Barge-in

Interruption signal drains the queue, cancels in-flight TTS jobs and (optionally) the LLM
stream; state machine moves to `INTERRUPTED` → `LISTENING`.

## Milestone 14 — Additional Voice Packs

Only after 1–13. The proof that the abstraction worked is that a second character needs
**no code changes** — only a new pack and profile.

---

## Risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| Not enough clean character audio (20–60 min target) | M2 measures and reports total approved minutes per style before any training starts; GPT-SoVITS v2Pro is the low-data/rough-audio fallback |
| Winner is licence-incompatible (IndexTTS bilibili licence, Fish weights) | Licence fields are part of the benchmark candidate config and the M7 decision record |
| Over-processing destroys character identity | D7: optional stages off by default, `processing_chain` recorded, null-processing control set in the benchmark |
| Benchmark unfairly favours zero-shot engines | D4: results keyed by adaptation mode; fine-tuned and zero-shot reported separately |
| Objective metrics mislead at high quality | D8: humans decide; metrics rank and regression-test |
| Style collapse — every line sounds the same | Reference Bank + deterministic rotation (M1), prosody-distribution metrics (M6) |
| Engine dependency hell | D1: sidecar per engine, no shared Python environment |
