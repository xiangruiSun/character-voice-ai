# Technology Landscape — Chinese Character Voice AI

Survey date: **2026-09-21**. This document records the open-source components this
project builds on, so we write integration code instead of re-implementing solved
problems. Every architectural decision in `docs/IMPLEMENTATION_PLAN.md` traces back to
something here.

**Rule of thumb for this repo:** we own the *character abstraction* (Voice Pack, Reference
Bank, Speech Planner, Conversation Orchestrator). We do **not** own TTS model code, ASR,
VAD, source separation, Chinese text normalization, or MOS prediction. Those are vendored
or pip-installed.

---

## 1. TTS / voice-cloning candidates

The original spec named three candidates (GPT-SoVITS, Qwen3-TTS, Fish Speech). Research
found three more that are directly relevant to *character* reproduction, in particular
IndexTTS-2.5 (explicit emotion control decoupled from timbre — exactly the "performance,
not just timbre" requirement) and VoxCPM2 (48 kHz native, LoRA fine-tune from 5–10 min).
The benchmark harness therefore ships with **six** candidate slots; the spec's three are
required, the other three are optional and flag-gated.

| Engine | License | Cloning mode | Fine-tuning | Style / emotion control | Streaming | Native SR |
|---|---|---|---|---|---|---|
| **GPT-SoVITS** v4 / v2Pro | MIT | few-shot (5 s) **+ fine-tune** | ✅ first-class, 2 stages (GPT + SoVITS) | reference audio only | `api_v2` modes 1–3 | 48 kHz (v4) |
| **Qwen3-TTS** 0.6B/1.7B Base | Apache-2.0 | `ref_audio` + `ref_text` | ✅ official `finetuning/` | `instruct` (VoiceDesign / CustomVoice) | ✅ ~97 ms | — |
| **Fish Speech / OpenAudio S1** | code open, **weights: verify** | ref audio + prompt tokens | ✅ LoRA only | inline emotion markers | HTTP `api_server.py` | — |
| **IndexTTS-2.5** | *bilibili Model Use License* (not OSI) | zero-shot, strong | ❌ none documented | ✅ **8-dim emotion vector, `emo_alpha`, `use_emo_text`, `duration_factor`** | — | — |
| **Fun-CosyVoice3-0.5B** | Apache-2.0 | zero-shot, cross-lingual | ✅ SFT + GRPO in repo; LoRA via community fork | `instruct`: emotion / speed / volume / dialect | ✅ 150 ms bidirectional | — |
| **VoxCPM2** (2B) | Apache-2.0 | "ultimate cloning" (ref + transcript) | ✅ SFT **and LoRA**, 5–10 min data | style guidance prompt | ✅ | **48 kHz** |

### Notes that change the design

* **Fine-tuning is the differentiator.** The spec's core thesis — "do not treat the
  character as only a speaker embedding" — means engines that support character
  adaptation (GPT-SoVITS, Qwen3-TTS, VoxCPM2, CosyVoice3) are strategically favoured over
  pure zero-shot engines, even when the zero-shot engine scores better on a 10-second clip.
  The benchmark must report **zero-shot and fine-tuned rows separately**; comparing a
  fine-tuned engine against a zero-shot one is not a fair test and would pick the wrong winner.
* **Emotion control is not uniform.** GPT-SoVITS expresses emotion *only* through which
  reference clip you pick. IndexTTS-2.5 takes an explicit 8-dimensional emotion vector.
  Qwen3-TTS and CosyVoice3 take natural-language `instruct` strings. This is why
  `TTSRequest` in this repo carries a **neutral `StyleControls` payload** plus a
  per-provider `ProviderCapabilities` declaration, and each adapter maps the neutral
  payload onto its native knobs. Without that, the provider interface would leak
  GPT-SoVITS's assumptions into everything.
* **Reference text is required by some engines.** Qwen3-TTS (`ref_text`), Fish Speech
  (`--prompt-text`) and VoxCPM2 need the transcript of the reference clip; GPT-SoVITS
  strongly benefits from it (`prompt_text`). Therefore the **Reference Bank stores the
  transcript with every reference clip** — not just the audio path. This is a hard schema
  requirement, easy to get wrong and expensive to retrofit.
* **Licensing gate.** IndexTTS ships under a bilibili model-use licence, not an OSI
  licence, and Fish Speech's weight licence must be confirmed before any commercial use.
  Every candidate in the benchmark config carries a `license` and `commercial_use` field
  so a licence-incompatible winner can't be selected by accident.
* GPT-SoVITS v4 measured timbre similarity 0.735 against a 0.750 ground-truth ceiling, up
  from 0.549 at v2 — and the maintainers note v3/v4 need *less* data to approach the target
  speaker. For poor-quality source audio, v2/v2Pro is the recommended fallback.

### Integration mode per engine

Every engine gets its own process (they have mutually incompatible torch/CUDA pins).
Adapters talk **HTTP to a per-engine sidecar**, never `import` the model into the API
process. Concretely:

* GPT-SoVITS → `api_v2.py`. Exact request schema captured in
  `configs/providers/tts.gpt_sovits.yaml` (`text`, `text_lang`, `ref_audio_path`,
  `aux_ref_audio_paths`, `prompt_text`, `prompt_lang`, `top_k=15`, `top_p=1`,
  `temperature=1`, `text_split_method="cut5"`, `speed_factor`, `seed=-1`,
  `media_type="wav"`, `streaming_mode`, `repetition_penalty=1.35`, `sample_steps=32`,
  `parallel_infer`, plus `/set_gpt_weights`, `/set_sovits_weights`, `/set_refer_audio`).
  The `/set_*_weights` endpoints are how we hot-swap character checkpoints without a
  process restart — this directly answers the spec's "do not load the model from disk for
  every sentence" failure mode.
* Qwen3-TTS → `pip install qwen-tts`, `Qwen3TTSModel.generate_voice_clone(text, language,
  ref_audio, ref_text)` with a reusable `create_voice_clone_prompt(...)` object — cache that
  prompt object per reference clip, it is the "loaded model" equivalent.
* Fish Speech → `tools/api_server.py`; cloning needs VQ tokens extracted from the
  reference (`fake.npy`) + `--prompt-text`. Pre-extract tokens for the whole Reference
  Bank at pack build time.
* CosyVoice3 → `AutoModel(...).inference_zero_shot(text, prompt_text, prompt_wav, stream=)`.
* IndexTTS-2.5 → `IndexTTS2(...).infer(spk_audio_prompt=..., text=..., emo_vector=...)`.
* VoxCPM2 → `pip install voxcpm`, CPU/MPS/CUDA selectable.

---

## 2. Voice Pack data pipeline

| Stage | Chosen tool | Why |
|---|---|---|
| Game asset extraction | AnimeWwise / `ue-wwise-extractor` / AssetStudio / `vgmstream` | Wwise `.bnk`/`.wem` → wav with original filenames; filenames often encode speaker + line ID, which is free metadata |
| Source separation (only if BGM present) | **`python-audio-separator`** (MIT) | Programmatic access to UVR model zoo; `model_bs_roformer_ep_317_sdr_12.9755` is top-SDR for vocals; Demucs available for stems; CPU/CUDA/MPS |
| De-reverb / de-echo | UVR VR-arch dereverb models, same package | Game dialogue is often bussed through reverb |
| Denoise (**off by default**) | DeepFilterNet3 | Small, fast, conservative. Deliberately *not* `resemble-enhance`: generative enhancers resynthesize the voice and change timbre — fatal for a character-identity dataset |
| VAD / segmentation | FSMN-VAD (FunASR) or **silero-vad v5** | Both mature; FSMN keeps us inside one toolkit |
| Chinese ASR + timestamps | **FunASR `paraformer-zh`** + `ct-punc` | Char-level timestamps, punctuation restoration, MIT toolkit; `AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++")` does segmentation, ASR, punctuation and speaker turns in one call |
| Emotion / event tags | **SenseVoiceSmall** (ASR + emotion + audio events) and/or **emotion2vec+large** | Bootstraps the style annotation in §8 of the spec; human review still required |
| Speaker filtering | **CAM++ / ERes2NetV2** from `3D-Speaker` | Cosine distance to a character centroid built from hand-picked clips; this is the defence against the "training on other characters" failure mode |
| Phoneme alignment (optional) | Montreal Forced Aligner (Mandarin acoustic + dict) | Only needed for diagnosing pronunciation defects; char timestamps from paraformer cover QC |
| Loudness | `pyloudnorm` (ITU-R BS.1770-4) / `ffmpeg-normalize` | Normalize to a fixed LUFS target with true-peak guard; **loudness-normalize, don't compress** |
| Slicing / preprocess | `fishaudio/audio-preprocess` (`fish-audio-preprocess`), GPT-SoVITS slicer | Loudness norm + slicing already solved |
| Quality scoring | DNSMOS / UTMOSv2 / NISQA | Use as a **relative filter within one pack**, never as an absolute quality claim |

### Preprocessing principles that come out of the research

1. Separation and denoising are **conditional stages, not defaults**. Every processed clip
   records which stages ran (`processing_chain`) so an over-processed dataset can be
   diagnosed and rebuilt from `raw/` without re-extracting.
2. Keep a **null-processing control set**: a handful of clips that skip every optional
   stage, so the benchmark can measure whether cleaning helped or hurt.
3. MOS predictors are trained mostly on read/telephony speech. A whispered or screamed
   character line will score badly while being perfectly good training data. Quality
   scores gate *review order*, not automatic rejection.

---

## 3. Evaluation

* **TTSDS2** (`github.com/ttsds/pipeline`) — factored objective score (generic / speaker /
  prosody / intelligibility), multilingual incl. Chinese via mHuBERT-147 + XLSR-53. It was
  the only metric of 16 tested to hold Spearman > 0.50 with human MOS in *every* domain
  (avg 0.67). Best available automatic proxy; still a proxy.
* **Speaker similarity (SECS)** — cosine between CAM++/ERes2NetV2 embeddings of the
  generated line and a held-out set of real character lines. Report against a
  **real-vs-real ceiling**, not against 1.0.
* **Intelligibility** — CER via `paraformer-zh` on generated audio vs the input text.
* **Prosody** — F0 mean/σ, speaking rate (chars/s), pause count and duration distribution,
  compared against the character's real distribution. This is how "identical intonation
  across all sentences" gets caught numerically.
* **Human evaluation is mandatory and decisive** (spec §18). Recent work specifically
  warns that reference-free quality metrics degrade as evaluators on modern high-quality
  TTS, which is exactly our operating regime. Objective metrics rank candidates and catch
  regressions; humans choose the winner.
* Listening-test tooling: **webMUSHRA** (audiolabs) and **BeaqleJS** are the established
  browser frameworks. We ship a self-contained local rating page (4 axes, 1–5, blind,
  randomized, incl. a real recording as a hidden anchor) and **also** export a
  webMUSHRA-compatible config for anyone who wants the standard tool.

---

## 4. Chinese text front-end

Do **not** hand-roll this (spec §13, and hand-rolled TN is a classic source of "poor
Chinese text normalization" failures).

* **WeTextProcessing** (`wenet-e2e`) — FST-based TN/ITN, the de-facto Chinese standard;
  or **`wetext`** (`pengzhendong/wetext`), a pynini-free runtime of the same rules, which
  installs cleanly without the pynini/OpenFst build.
* **`cn2an`** — numbers ↔ Chinese numerals, both directions, handles 分数/小数/百分比.
* **`pypinyin`** + **`g2pW`** (via `pypinyin-g2pW`) — polyphone disambiguation (多音字).
  Needed for character-specific pronunciation habits and for a per-character pronunciation
  override lexicon.
* Our own layer stays thin and is only: a per-character override lexicon, Latin/abbreviation
  spoken-form rules (e.g. `GPT-5.6`), emoji/symbol stripping, and TTS-safe punctuation
  mapping. Everything else delegates.

---

## 5. Runtime (later milestones)

* **VAD:** `ten-vad` (low-latency, designed for agents) or `silero-vad` v5.
* **Orchestration:** the spec mandates our own `ConversationOrchestrator` state machine, so
  we keep it — but the barge-in design borrows the proven pattern from **Pipecat** /
  **LiveKit Agents** / **TEN**: frame-based pipeline, cancellable tasks, a single
  interruption signal that drains the audio queue and cancels in-flight TTS jobs.
* **Transport:** WebSocket + AudioWorklet for the MVP. If we later need echo cancellation,
  packet loss handling and mobile networks, LiveKit's WebRTC transport is the drop-in.
* **STT:** OpenAI transcription first (spec), `faster-whisper` or FunASR streaming
  paraformer as the local substitute — the `SpeechToTextProvider` interface keeps both
  viable.

---

## 6. What we deliberately do not use

* **`resemble-enhance` / generative restoration on training data** — resynthesizes and
  changes timbre.
* **Aggressive spectral denoising / noise gates** — removes breaths and vocal fry, which
  the spec explicitly lists as character identity.
* **A single fixed reference clip** — the Reference Bank exists precisely to avoid this.
* **An end-to-end voice LLM (e.g. OpenAI voice) as the character voice** — spec §3.
* **Cross-lingual cloning, multilingual packs** — V1 is `zh-CN` only, enforced in schema
  validation.

---

## Sources

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) · [README](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/README.md?plain=1) · [v3/v4 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)) · [api_v2.py](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py)
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) · [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) · [finetuning](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) · [community finetune pipeline](https://github.com/sruckh/Qwen3-TTS-finetune)
- [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) · [finetune docs](https://github.com/fishaudio/fish-speech/blob/main/docs/en/finetune.md) · [inference docs](https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md) · [fishaudio/audio-preprocess](https://github.com/fishaudio/audio-preprocess)
- [index-tts/index-tts](https://github.com/index-tts/index-tts) · [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) · [IndexTTS2 project page](https://index-tts.github.io/index-tts2.github.io/)
- [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice) · [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) · [2026 Chinese voice-cloning comparison](https://liudon.com/posts/voice-cloning-solution-comparison/)
- [modelscope/FunASR](https://github.com/modelscope/FunASR) · [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) · [ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec) · [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) · [facebookresearch/demucs](https://github.com/facebookresearch/demucs) · [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) · [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- [wenet-e2e/WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) · [pengzhendong/wetext](https://github.com/pengzhendong/wetext) · [GitYCC/g2pW](https://github.com/GitYCC/g2pW) · [mozillazg/pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [TTSDS2 paper](https://arxiv.org/html/2506.19441v2) · [audiolabs/webMUSHRA](https://github.com/audiolabs/webMUSHRA) · [HSU-ANT/beaqlejs](https://github.com/HSU-ANT/beaqlejs) · [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) · [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA)
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) · [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) · [Escartem/AnimeWwise](https://github.com/Escartem/AnimeWwise)

*Not retrieved:* arXiv 2609.13150 ("The Limits of Reference-Free Speech Quality Metrics as
Evaluators and Rewards on Modern Text-to-Speech") — rate-limited during this survey. Its
abstract supports the "human evaluation is decisive" position above; re-check before
relying on automatic metrics for model selection.
