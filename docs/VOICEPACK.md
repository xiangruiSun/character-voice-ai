# Voice Pack format and data pipeline

A Voice Pack is everything derived from one character's recordings: a versioned,
relocatable directory that a benchmark, a training run or a live conversation can all be
pointed at.

```
voicepacks/<pack_id>/
├── raw/            untouched originals — never written to after ingest
├── processed/      intermediate stage outputs, freely regenerable
├── clean/          human-approved training audio
├── rejected/       filtered-out clips, kept with the reason
├── transcripts/    ASR output and human corrections
├── metadata/       voicepack.yaml · dataset.json · references.json
├── references/     Reference Bank, one subdirectory per style
├── datasets/       engine-specific exports
├── checkpoints/    character-specific fine-tuned models
└── evaluation/     benchmark runs and listening-test results
```

Create one with `cvai-voicepack init <id>`; check it at any time with
`cvai-voicepack validate <id>`.

---

## The four rules

**1. `raw/` is immutable.** Drop the originals in once. Everything else is derived and
can be deleted and rebuilt. The schema enforces the consequence: a `TrainingSample` may
not point into `raw/`, because anything approved has at minimum been segmented.

**2. Optional cleaning stays off unless the audio needs it.** Source separation,
de-reverb and denoising are each off by default, and every clip records the
`processing_chain` that produced it. This is decision D7, and it exists because spec §7
is right: the objective is not maximum spectral cleanliness, and over-processing removes
the breaths, aspiration, soft consonants and vocal fry that *are* the character.

> Remove interference that hurts training, but preserve the voice actor's performance.

Keep a **null-processing control set** — a handful of clips that skipped every optional
stage — so "did cleaning help?" is a question with an answer.

**3. Reference clips carry their transcript.** Qwen3-TTS (`ref_text`), Fish Speech
(`--prompt-text`) and VoxCPM2 require it; GPT-SoVITS (`prompt_text`) improves sharply
with it. A bank without transcripts has to be rebuilt by listening to every clip. The
schema makes `transcript` required on `ReferenceSample`.

**4. At least two reference clips per style.** One clip per style means every line in
that style is conditioned on the same audio — spec §27's "using one reference clip for
all emotions", arrived at by accident. The validator warns at one and errors at zero.

---

## The pipeline (Milestone 2)

```
raw audio
  → ingest            checksum, never mutate again
  → separate*         python-audio-separator (BS-Roformer) — only if BGM is present
  → dereverb*         UVR VR-arch models — only if the dialogue is wet
  → denoise*          DeepFilterNet3, conservative — usually skip
  → segment           FSMN-VAD / silero-vad, padded to keep leading breaths
  → speaker_filter    CAM++ / ERes2NetV2 cosine vs a hand-curated character centroid
  → transcribe        FunASR paraformer-zh + ct-punc, char-level timestamps
  → annotate          SenseVoice / emotion2vec bootstrap + measured rate, F0, pauses
  → quality           pyloudnorm, clipping, SNR, DNSMOS/UTMOSv2
  → review            human corrects transcripts and labels, approves or rejects
  → build             dataset.json + references.json + engine exports
```

`*` optional, off by default.

Tool choices and why, with sources: `docs/TECH_LANDSCAPE.md`. Install the stack with
`make install-preprocess`.

### Notes that matter more than they look

* **Segment padding.** Trim to speech and you cut the intake of breath before a line.
  That breath is a large part of why a character sounds alive. Pad generously and let the
  human reviewer tighten.
* **Speaker filtering is the defence against training on the wrong character.** Game
  dialogue files routinely contain more than one voice. Build the centroid from clips a
  human has confirmed, then score everything against it and keep the score on the sample.
* **Quality scores order the review queue; they do not reject.** MOS predictors are
  trained mostly on read and telephony speech. A whispered or shouted character line
  scores badly while being excellent training data.
* **Emotion labels start automatic and end human.** SenseVoice and emotion2vec get you a
  first pass; the character-specific styles (`soft_teasing`, `embarrassed`, `cold`) are
  the ones that carry identity, and no off-the-shelf model knows them.

---

## Manifests

### `metadata/voicepack.yaml`

Identity, version, source and licence note, target format, which optional stages are
enabled, the style list, and registered checkpoints. Styles declare a `core_style` so the
retriever can fall back: `soft_teasing` → `teasing` → `neutral`.

### `metadata/references.json` — the Reference Bank

Per clip: id, path, **transcript**, style, core style, duration, quality score, tags, and
`precomputed` — engine-specific artifacts extracted once at build time (Fish Speech VQ
prompt tokens, for instance) rather than per request.

### `metadata/dataset.json`

Every `TrainingSample` with its transcript, performance annotation (emotion, intensity,
rate, pitch profile, pause style, voice style), measured signals (chars/second, F0 mean
and spread, pause count), quality and speaker-similarity scores, `processing_chain`,
review status, and the split it belongs to.

The `heldout` split matters disproportionately: those are real lines no engine trained
on, and they become the **hidden anchor** in every blind listening test. Without them a
listening test has no scale.

---

## Validation gates

`cvai-voicepack validate <id> --strict --check-profile <character_id>` before any
training run. It reports:

| Code | Severity | Meaning |
|---|---|---|
| `manifest.id_mismatch` | error | directory and manifest disagree |
| `references.style_empty` | error | a declared style has no clips — every line in it falls back |
| `references.style_thin` | warning | one clip for a style |
| `references.no_transcript` | error | a clip several engines cannot use |
| `references.missing_audio` | error | manifest points at a file that is not there |
| `dataset.too_small` | warning | under the 20-minute target from spec §19 |
| `dataset.no_heldout` | warning | the listening test would have no real anchor |
| `dataset.unknown_styles` | error | approved samples use undeclared styles |
| `dataset.unreviewed` | warning | samples no human has looked at |
| `profile.unsupported_styles` | error | the character asks for styles her voice cannot perform |

`dataset_minutes_by_style` is the number to read before training. Forty minutes that are
95% neutral cannot perform a character, and the total alone hides that.
