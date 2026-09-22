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
| D7 | **Optional preprocessing stages (separation, dereverb, denoise) are off by default, each clip records its `processing_chain`, and every reference records which original recording it was cut from — so the benchmark can run a null-processing control.** | Spec §7: over-processing destroys character identity. Recording the chain makes "did cleaning help?" answerable; the provenance makes it *testable*, by running the same lines uncleaned as a candidate. The run refuses rather than falling back to the cleaned clips, because a control that silently becomes a copy of what it controls for returns a confident wrong answer. |
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

## Milestone 8 — Character Brain + Speech Planner + text normalization ✅

**Character Brain / Speech Planner** (`cvai_speech_planner`): the system prompt is built
from the `CharacterProfile` — personality, lore, speaking habits, forbidden behaviour and
retrieved original dialogue examples — in Chinese, because a Chinese character described
in English drifts toward translationese. The planner asks for a `CharacterSpeechPlan`
three ways in order of preference: constrained JSON, parse-and-repair from a code fence,
and finally plain text with the character's default register. A conversation turn never
fails over formatting, and which path ran is recorded, because a planner silently living
on the third path is one whose provider needs changing.

Guards then make the plan performable: a style the voice pack does not have is mapped to
one it does, an over-long reply is cut at a sentence boundary rather than mid-clause,
bracketed narration is stripped, and a reply that breaks character is flagged. Repairs are
counted rather than applied silently — a planner rewriting a quarter of its own output has
a prompt problem.

**Chinese text front-end** (`cvai_text_normalizer`): numbers, dates, times, percentages,
currency, temperatures, fractions, ordinals, phone numbers, Latin acronyms, product
versions and alphanumeric codes, punctuation, emoji and markdown, plus the per-character
pronunciation lexicon. `wetext` and `cn2an` are used when installed; the built-in rules
run either way, as the whole implementation on a bare install and as a safety net after a
library pass. Both paths are covered by tests, so behaviour does not depend on what
happens to be installed.

Also here, early, because it is pure text and testable: the **speech chunker** (spec §14,
nominally Milestone 12). Cuts on Chinese sentence endings, merges pieces too short to be
worth a separate request, uses comma-level boundaries only past the preferred length, and
never strands a conjunction at the start of a chunk.

## Milestones 9, 10, 12, 13 — the conversation loop ✅

`cvai_conversation` (orchestrator), `cvai_api` (HTTP + WebSocket), `apps/web/dev-client.html`.

**The orchestrator** owns pipeline state and emits one `TurnEvent` stream that every
transport consumes — API, CLI and tests alike, so "what happens during a turn" is defined
once and cannot drift between them.

**Planning mode** resolves a real tension between spec §12 (a structured performance plan
for the reply) and spec §14 (start speaking before the reply is finished). The plan
describes the whole utterance, so it cannot exist until the LLM has finished:

* `complete` (default) — full plan, then normalize → chunk → synthesize, streaming each
  chunk's audio as it is ready. TTS is pipelined; the LLM is not. This follows spec §14's
  own instruction that naturalness beats speed.
* `streaming` — chunk the LLM's text as it arrives, speak it in the character's default
  register. Lower latency, no per-line performance direction.

**Barge-in** (Milestone 13) is built in rather than retrofitted, which is what spec §16
asks for. All four of its required behaviours are separately tested: further audio stops,
the queue drains, in-flight synthesis is cancelled, and audio that *finishes* just after
the interrupt is dropped rather than played — otherwise the character gets one more word
in after being cut off. An interrupt carries the turn id it targets, so a late signal
cannot silence the next reply.

**The API** is layered so almost none of it needs a web server to test: session assembly
and event→protocol mapping are pure functions; `app.py` is transport only. The mapping
deliberately drops the performance metadata — spec §12 says the user only sees the text,
and not sending emotion or reference style to the browser means no frontend can start
displaying or deciding it.

## Milestone 11 — Microphone and Speech-to-Text ✅

The upstream half of the conversation. The browser captures 16 kHz mono PCM through an
AudioWorklet and streams it raw up the same WebSocket; **every decision is made on the
server**, per spec §15. A browser-side VAD would be a second state machine, and two state
machines drift.

`UtteranceDetector` (`cvai_conversation.listening`) answers three questions frame by
frame, and the thresholds encode which mistakes cost more:

| Question | Rule | Why that way |
|---|---|---|
| Has the user started? | 3 consecutive voiced frames | one frame is a door closing |
| Have they finished? | 700 ms of trailing silence | shorter and the character talks over a pause for breath |
| Are they talking over her? | 5 voiced frames during playback | a false positive costs a cut-off reply; a miss means the user is talked over, which is the thing that most reliably breaks the illusion |

Three details that are not obvious until something sounds wrong:

* **The pre-roll ring.** Speech is declared several frames after it began, so the frames
  that triggered the decision are held and flushed into the utterance. Losing 60 ms of a
  Mandarin initial is enough to turn 四 into 是. It is a *ring*, not a growing buffer, so
  a cough heard at the start of a 20 s reply is not glued onto the front of whatever the
  user eventually says.
* **The noise floor does not track speech.** An earlier version adapted on every frame,
  and a long uninterrupted sentence pulled the floor to within the margin of the
  speaker's own voice: after about a second and a half, speech read as silence and the
  character began answering halfway through. Found by a test, not by listening.
* **Intake is separable from response.** `observe()` is ordered and microsecond-cheap;
  `respond()` runs ASR, an LLM and a TTS engine. A transport that ran them together would
  stop reading the microphone for exactly as long as the reply takes — which is precisely
  the window barge-in exists for.

STT has two adapters. OpenAI is the default; **FunASR (`paraformer-zh`) is the better one
once the weights are local** — trained for Mandarin rather than for everything, takes real
weighted hotwords so the character's name is recognised rather than guessed at, and keeps
the user's microphone off someone else's server. It borrows the loader from the Voice Pack
preprocessing backends so the ASR that transcribed the training data is the ASR that hears
the user.

Echo cancellation is the browser's `echoCancellation` constraint plus headphones. Without
it, open speakers make the character barge in on her own voice. Anything more belongs in
WebRTC, not in this loop.

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
