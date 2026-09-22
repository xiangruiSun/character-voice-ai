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

The **null-processing control** makes that testable rather than merely recorded. Every
`ReferenceSample` carries `source_clip` and `source_offset_s` — which original recording
it was cut from, and where — so the benchmark can rebuild the untouched version of the
same clip and run it as a candidate (`reference_source: unprocessed`). Same lines, same
styles, same seeds, no cleaning: whatever the listening test hears between the two is the
processing chain.

This is why rule 1 matters in practice. Delete `raw/` and the control becomes
impossible — the run will refuse rather than substitute the cleaned clips, which is the
correct behaviour and also a permanent loss of the only evidence about your own
preprocessing.

**3. Reference clips carry their transcript.** Qwen3-TTS (`ref_text`), Fish Speech
(`--prompt-text`) and VoxCPM2 require it; GPT-SoVITS (`prompt_text`) improves sharply
with it. A bank without transcripts has to be rebuilt by listening to every clip. The
schema makes `transcript` required on `ReferenceSample`.

**4. At least two reference clips per style.** One clip per style means every line in
that style is conditioned on the same audio — spec §27's "using one reference clip for
all emotions", arrived at by accident. The validator warns at one and errors at zero.

---

## Running it

```bash
cvai-prep backends                                  # what is actually installed
cvai-voicepack init denia_cn --display-name "迪尼娅"
cvai-prep run denia_cn --source ~/denia_voice_lines --hotword 迪尼娅
cvai-prep status denia_cn
cvai-prep review denia_cn                           # writes processed/review.html
# … listen, correct transcripts, set styles, approve/reject, export the patch …
cvai-prep apply denia_cn review-patch.json
cvai-prep anchors denia_cn add <confirmed_segment_ids…>
cvai-prep run denia_cn --stages speaker_filter      # rescore against the real centroid
cvai-prep build denia_cn                            # clean/ + dataset + reference bank
```

Stages are **resumable** — state lives in `processed/state.json` and a re-run picks up
where the last one stopped. Human corrections are marked `human_edited` and are never
overwritten by a later automatic pass, so `cvai-prep run` after a review round is safe.

Useful flags: `--limit N` for a trial pass over a big dump before committing hours of
ASR; `--no-segment` when the source is already one line per file (common for game voice
exports); `--separate` / `--denoise` to turn on the optional cleaning stages;
`--auto-approve` to skip human review, which you should read rule 4 below before using.

`cvai-prep backends` reports which implementation each stage resolved to. Without the
`preprocess` extra, segmentation and all the acoustic measurements still work (they are
built in), but transcription falls back to a **stub** that writes visible placeholder
text — and `cvai-prep build` refuses to ship a dataset containing it.

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

### What runs without any of that installed

Segmentation and every acoustic measurement are implemented in the repository and need
no model, no GPU and no download:

| Measurement | How |
|---|---|
| Voice activity | adaptive energy gate — threshold from the clip's own noise floor, so one file being studio-clean and the next being reverberant does not need two configs |
| Pitch (F0 mean / spread) | normalized autocorrelation on a decimated signal |
| Speaking rate | CJK characters per second |
| Internal pauses | VAD run at a tighter silence threshold |
| SNR | speech-frame vs non-speech-frame energy |
| Clipping | fraction of samples at full scale |
| Loudness | `pyloudnorm` (BS.1770-4) when installed, RMS approximation otherwise — the record says which |
| Speaker similarity | spectral band energies + pitch statistics, cosine against the character centroid; **advisory only**, and never used to auto-reject |

Transcription and emotion classification genuinely need models. Rather than fake them,
the pipeline falls back to a stub that labels itself, and the dataset builder blocks on
it.

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
