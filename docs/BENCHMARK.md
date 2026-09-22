# The Character Voice Benchmark

The first technical question the project must answer, and the thing spec §19 says to
build before any website:

> Which TTS adaptation approach reproduces *this* character most faithfully?

---

## Running it

```bash
# 1. generate — every candidate, same sentences, same seeds, same reference bank
cvai-bench run configs/benchmarks/denia_cn_v1.yaml

# 2. assemble a blind listening test with real recordings mixed in
cvai-bench blind runs/<run_id>            # add --webmushra to also emit that config

# 3. hand blind/rate.html to listeners. Do not hand them blind/key.json.

# 4. aggregate what comes back
cvai-bench aggregate runs/<run_id> --ratings ratings/
```

To see the whole thing work with no GPU, no weights and no listeners:

```bash
make demo
```

---

## What a run produces

```
runs/<run_id>/
├── run.json          every GenerationRecord — the reproducibility record
├── config.json       the exact merged configuration used
├── env.json          python, platform, git commit, whether the tree was dirty
├── events.jsonl      append-only log, written as the run proceeds
├── audio/<candidate>/<sentence>.wav
├── ground_truth/     real held-out recordings, copied in as anchors
├── blind/
│   ├── audio/item-000.wav …   copied under opaque names
│   ├── items.json             what the rater sees
│   ├── key.json               what the rater must not see
│   └── rate.html              self-contained rating page
├── report.md         run health
└── evaluation.md     results, after ratings
```

Every `GenerationRecord` carries engine version, checkpoint id, voice pack id **and
version**, reference clip id, whether that reference was a fallback, the seed, and the
exact resolved parameters sent to the engine. That is spec §23's list, and it is why a
result six weeks old can still be re-created.

---

## Design choices that keep the comparison honest

**Candidates are `(engine, adaptation_mode)`, never bare engines.** A fine-tuned engine
compared against a zero-shot one is not a comparison of engines. Keeping zero-shot rows
for the fine-tunable engines is also the cleanest evidence for the project's core claim:
the zero-shot → fine-tuned delta *for the same engine* is what shows whether character
adaptation actually beats speaker embedding.

**Same seed and same reference clip per sentence, across candidates.** Otherwise the
difference you hear is the random draw. Reference rotation varies clips *between* sentences,
never *between* candidates on one sentence.

**An unavailable engine does not abort the run.** Its sentences are recorded as failures
with the reason and the report shows the gap. A four-hour run should not die at minute
ten because one sidecar was not started.

**Dropped controls are reported.** GPT-SoVITS cannot honour an emotion vector; if one was
sent, the record says so. A silently ignored control is indistinguishable from a bad
model at listening-test time.

**A null-processing control runs alongside the real candidates** (decision D7). Set
`reference_source: unprocessed` on a candidate and it is conditioned on reference clips
cut straight from the original recordings — same lines, same styles, same seeds, no
cleaning. It is possible because every reference records which recording it came from
and at what offset, so the control is the *same clip*, not a similar one.

This is the only instrument that answers "did our preprocessing preserve her, or sand her
down?", which spec §27 lists as a way to lose a character voice while every number
improves. The run **refuses** rather than falling back to the cleaned clips when the
provenance or the originals are missing: a control that silently becomes a copy of what
it controls for does not fail loudly, it produces a confident wrong answer. The run report
labels it `unprocessed (control)` so nobody reads it as a competitor.

**Fallback rate is a first-class number.** If the Reference Bank could not serve the
requested style for a quarter of the lines, every candidate was partly judged on fallback
styles and the differences the benchmark exists to measure have been flattened. The run
report says so in bold before anyone starts listening.

---

## The test sentence set

`configs/benchmarks/sentences_zh_v1.yaml` — 25 unseen Mandarin sentences, none of which
may appear in any training split.

Twenty neutral statements would let every candidate score well and tell you nothing, so
the set is built around what actually separates a character voice from a competent
generic one:

| Category | Why it is there |
|---|---|
| `everyday` | the baseline everything else is read against |
| `emotional` | eight styles, because one emotion is not a range |
| `short_interjection` | `哼。` — one syllable, nothing to hide behind |
| `punctuation_heavy` | `我……我不是那个意思` — pause style, exposed |
| `long_sentence` | breath placement, and chunking artefacts |
| `numbers_and_dates` | also an end-to-end check on the text normalizer |
| `latin_and_abbreviations` | `GPT-5.6`, `CUDA 12.8`, `A7-3B` |
| `pronunciation_trap` | 重 chóng/zhòng, 行 háng/xíng, 调 tiáo/diào |
| `character_signature` | **replace these** with lines this character is known for |

The last category is the most diagnostic and the only one that must be edited per
character: a line someone knows by heart is where a wrong reading is unmissable.

---

## Human evaluation

Mandatory and decisive (spec §18). Four axes, 1–5, blind:

| Axis | Question | Direction |
|---|---|---|
| Speaker similarity | 这听起来像是目标角色的声音吗？ | higher is better |
| Naturalness | 这听起来像真人录音吗？ | higher is better |
| Character similarity | 这听起来像这个角色平时说话的方式吗？ | higher is better |
| AI artefact level | 能多明显地听出这是AI生成的？ | **lower is better** |

Plus one direct question — "I think this is a real recording" — because spec §18's
success condition is about fooling someone who knows the character, and it is better
asked than inferred.

**Blinding is structural, not a promise.** Item ids are opaque and assigned after
shuffling; audio is copied to `blind/audio/item-000.wav` so the path cannot leak the
candidate; sentence ids stay in the key; the key is a separate file the page never loads.
A test asserts all of this, because it is the kind of thing that decays quietly.

**Real recordings are in the test.** Held-out character lines are mixed in as hidden
anchors. If the candidates score 3.4 and the real recordings score 3.5, the raters were
not discriminating and the run establishes nothing — which the report will say.

The report's headline is not the raw mean but the **gap to real recordings** on each
axis. Spec §18 asks whether a generated line could pass as a new recording; the number
that answers that is the distance to the anchor, approaching zero.

---

## Objective metrics (Milestone 6)

```bash
cvai-bench report runs/<run_id> --objective
```

They rank candidates and catch regressions. They do not choose the winner.

**Prosody comparison — always available, no model needed.** The character's real speech
has a measurable distribution: pitch centre and spread, speaking rate, pause density,
silence ratio. Generated audio is measured with *the same code* (`cvai_core.dsp`, shared
with the Voice Pack pipeline precisely so the two agree) and compared against it. The
reference distribution is built from the **held-out** split — real lines no engine was
trained on.

This is what turns two of spec §27's failure modes into numbers:

| Symptom | What it looks like | Flag |
|---|---|---|
| identical intonation across all sentences | F0 σ ratio below ~0.6 | "flat delivery" |
| wrong voice, however natural | pitch centre more than 2σ from the character's | "pitch centre is ±N Hz from the character's" |
| robotic pacing | speaking rate outside 75–133% of hers | "speaking rate is N% of the character's" |
| padded clips | silence ratio well above hers | "more silence than the character's real lines" |

The per-candidate distance combines the axes by **RMS, not mean**: a candidate that
matches on three axes and is badly wrong on the fourth is not three-quarters right, and
averaging would dilute exactly the signal worth acting on.

**Model-backed metrics — when installed.** Each reports its own absence rather than
quietly returning nothing.

* **TTSDS2** — factored (generic / speaker / prosody / intelligibility), the only metric
  of sixteen tested that held Spearman > 0.50 against human MOS in every domain.
* **SECS** — CAM++/ERes2NetV2 cosine against real character audio, always reported
  against the **real-vs-real ceiling**, never against 1.0.
* **CER** — paraformer-zh on the generated audio versus the input text.
* **UTMOSv2 / DNSMOS** — recorded, weakest evidence of the set; reference-free predictors
  degrade precisely in the high-quality regime this project operates in.

---

## Choosing a winner (Milestone 7)

Write a decision record in `docs/decisions/`. It must state:

1. the winner and the runner-up, with the per-axis numbers and the gap to the anchor;
2. **the licence check** — IndexTTS-2.5 is under a bilibili model licence, not OSI, and
   Fish Speech's weight licence needs confirming. The candidate config carries
   `license` and `commercial_use` fields so this cannot be skipped by accident;
3. what would reverse the decision.

A composite score is printed for convenience. Quote the individual axes in the decision,
because a composite hides the case that matters most: natural but not this character.
