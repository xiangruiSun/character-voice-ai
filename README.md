# Character Voice AI (Chinese, V1)

**Author:** andersonmork817 · MIT licensed (code only — see `LICENSE` for what that does
and does not cover)

Build a **character-specific** Chinese voice — not generic TTS, and not ten-second
zero-shot cloning. The target is that someone who knows the character could plausibly
mistake a generated line for a new recording of her.

The project is organised around one question, asked before any product is built:

> Given a high-quality Chinese Voice Pack from one specific game character, which
> available TTS adaptation approach reproduces that character most naturally and
> accurately?

Milestones 1–7 answer it with a benchmark. Milestones 8–13 build the conversation
product around the winner. `docs/IMPLEMENTATION_PLAN.md` has the full sequence.

---

## Current status

**Everything that can be built without model weights is built.** You can hold a spoken
Chinese conversation with a character today, end to end — microphone in, her voice out —
using the mock engine.

| Milestone | State |
|---|---|
| M1 — architecture, interfaces, config, schemas, benchmark pipeline | ✅ |
| M2 — Voice Pack preprocessing, from raw audio to an approved dataset | ✅ |
| M3–M5 — per-engine dataset exporters, sidecars, training runbooks | scaffolded; needs a GPU |
| M6 — objective metrics: prosody compared against the character herself | ✅ |
| M7 — **choose the engine** | blocked on real character audio and a GPU |
| M8 — Character Brain, Speech Planner, Chinese text front-end | ✅ |
| M9, M10, M12, M13 — conversation orchestrator, API, streaming, barge-in | ✅ |
| M11 — microphone capture, server-side endpointing, speech-to-text | ✅ |
| M14 — a second character | after 1–13, and it should need no code |

The whole apparatus runs with **no GPU, no model weights and no network** — only
`pydantic` and `PyYAML`. That is deliberate: the machinery that will answer the research
question has to be trustworthy before a real engine is plugged into it, and every
model-backed stage reports its own absence rather than silently producing nothing.

```bash
make install     # pip install -e ".[dev]"
make test        # 452 tests
make demo        # full benchmark pipeline on a synthetic pack with the mock engine
```

`make demo` generates audio for four candidates — including the null-processing control — over 25 Chinese test sentences, builds a
blind listening test with real-recording anchors, aggregates (simulated) ratings and
writes both reports — in about a minute, on a laptop.

---

## What the pieces are

```
character-voice-ai/
├── packages/
│   ├── shared_types/      cvai_types          — every typed schema in the system
│   └── audio_protocol/    cvai_audio_protocol — browser ↔ server messages (M9-13)
├── services/
│   ├── core/              cvai_core           — config, interfaces, registry, run logs
│   ├── reference_retrieval/                   — which reference clip to condition on
│   ├── evaluation/        cvai_evaluation     — the benchmark and listening test
│   ├── voice_preprocessing/                   — Voice Pack pipeline (M2)
│   ├── text_normalizer/                       — Chinese TTS front-end (M8)
│   ├── speech_planner/                        — performance director (M8)
│   └── conversation/                          — orchestrator + barge-in (M9-13)
├── providers/{tts,stt,llm}/                   — engine adapters, swappable by config
├── characters/profiles/                       — CharacterProfile YAML
├── voicepacks/<id>/                           — raw → clean → references → checkpoints
├── configs/                                   — app, providers, benchmarks
├── scripts/, tests/, docs/, models/, apps/
```

Two rules hold the shape together:

1. **Nothing outside `providers/` imports a concrete engine class.** Configuration names
   an adapter by key; the registry builds it. This is what keeps model-specific code out
   of conversation and evaluation logic.
2. **A character's *words* and a character's *voice* never mix.** A `CharacterProfile`
   references a voice pack by id. Swapping the TTS engine cannot touch a profile, and
   rewriting a personality cannot invalidate a trained voice.

---

## The two halves

**Character Brain** — personality, lore, dialogue examples, memory. Prompting and
retrieval only in V1; the LLM is never fine-tuned and never produces audio.

**Character Voice** — a Voice Pack: a versioned directory holding raw audio, approved
training clips, a Reference Bank organised by speaking style, engine-specific datasets,
fine-tuned checkpoints and evaluation results. Full layout and rules in
`docs/VOICEPACK.md`.

Between them sits the **Character Speech Planner** (spec §12): the LLM returns not just a
line but how to perform it — emotion, intensity, rate, volume, pauses, ending shape and
which reference style to draw from. The user sees only the text.

---

## Engines

Six candidates, all behind one `TTSProvider` interface, each in its own sidecar process:

| key | engine | adaptation | style control | licence |
|---|---|---|---|---|
| `gpt_sovits` | GPT-SoVITS v4 / v2Pro | fine-tune | reference clip only | MIT |
| `qwen3_tts` | Qwen3-TTS 0.6B/1.7B | fine-tune / LoRA | `instruct` | Apache-2.0 |
| `fish_speech` | Fish Speech / OpenAudio S1 | LoRA | emotion markers | verify weights |
| `index_tts` | IndexTTS-2.5 | zero-shot only | 8-dim emotion vector | bilibili licence |
| `cosyvoice` | Fun-CosyVoice3 0.5B | SFT | `instruct` | Apache-2.0 |
| `voxcpm` | VoxCPM2 | SFT / LoRA | style prompt | Apache-2.0 |
| `mock` | built-in | — | — | MIT |

The spec named the first three; the survey in `docs/TECH_LANDSCAPE.md` added the rest —
IndexTTS-2.5 because it is the only one that decouples emotion from speaker identity,
CosyVoice3 for native streaming, VoxCPM2 for 48 kHz and LoRA from ten minutes of audio.

No engine has been chosen. That is Milestone 7's job, and it is decided by human
listening, not by a leaderboard.

---

## Running things

```bash
# Voice packs
cvai-voicepack init denia_cn --character denia_cn --display-name "迪尼娅"
cvai-voicepack validate denia_cn --strict --check-profile denia_cn
cvai-voicepack list

# Characters (the other half)
cvai-character init denia_cn --name "迪尼娅" --voicepack denia_cn
cvai-character lint denia_cn --pack denia_cn        # styles with no examples, and more
cvai-character list

# Preprocessing (Milestone 2)
cvai-prep backends                                  # what is installed
cvai-prep run denia_cn --source ~/denia_voice_lines --hotword 迪尼娅
cvai-prep review denia_cn                           # writes processed/review.html
cvai-prep apply denia_cn review-patch.json
cvai-prep build denia_cn                            # clean/ + dataset + reference bank

# Benchmark
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
cvai-bench blind runs/<run_id> --webmushra
cvai-bench aggregate runs/<run_id> --ratings ratings/

# Talk to her (Milestones 9-13)
pip install -e '.[runtime]'
uvicorn cvai_api.app:app --port 8000
open apps/web/dev-client.html          # type, or press 开麦 and speak
```

Configuration layers: `configs/app.yaml`, then `CVAI__section__key` environment
variables, then command-line overrides. `${VAR}` and `${VAR:-default}` are expanded at
load time, so no secret is ever written into a config file.

---

## Documentation

| Document | What it covers |
|---|---|
| `docs/GETTING_STARTED.md` | **Start here** — the path from a folder of voice lines to a character who talks back, with the decisions you have to make on the way |
| `docs/IMPLEMENTATION_PLAN.md` | All 14 milestones, and the eight architectural decisions fixed in Milestone 1 |
| `docs/TECH_LANDSCAPE.md` | The open-source survey behind those decisions, with sources |
| `docs/VOICEPACK.md` | Voice Pack format, the preprocessing pipeline, and the cleaning philosophy |
| `docs/BENCHMARK.md` | How to run the benchmark and read its output honestly |
| `docs/SIDECAR_PROTOCOL.md` | The HTTP contract each engine container implements |
| `docs/decisions/` | Decision records, starting with the Milestone 7 engine choice |

---

## The failure this project is built to avoid

A voice that is clean, pleasant, natural — and not the character. High naturalness with
low character similarity is the specific outcome the whole apparatus exists to detect,
which is why the listening test asks about speaker similarity, naturalness, character
similarity and AI artefacts **separately**, always against real recordings of the
character mixed into the same blind test.
