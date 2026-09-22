# Getting started — from a folder of voice lines to a character who talks back

Everything else in `docs/` is reference. This is the path, in order, with the decisions
you have to make along the way and the places people usually go wrong.

It assumes you have voice lines for one Chinese-speaking character and want a voice
someone who knows her would accept as hers. It does **not** assume you have a GPU yet —
the first three steps run on a laptop, and they are the steps that decide whether the
rest is worth doing.

---

## Before any of it: see what you are building

```bash
make install && make talk        # or: make audition
```

No API key, no GPU, no network, no character audio. A scripted LLM stands in for her
brain and the mock engine for her voice; everything between them — performance planning,
Chinese text normalization, chunking, reference retrieval, synthesis, playback — is the
real pipeline. It sounds like tones, because the "voice pack" is generated tones. What it
shows is the shape of the thing you are about to spend a weekend collecting audio for.

---

## 0. What you need before you start

| | Minimum | Comfortable | Why it matters |
|---|---|---|---|
| Clean speech | 20 min | 40–60 min | Below ~20 min, fine-tuning learns timbre but not her *range*, and every line comes out in the same register |
| Distinct styles | 2 | 4–6 | The Reference Bank conditions each line on a clip in the right style; with one style everything sounds the same (spec §27's "style collapse") |
| Sample rate | 22.05 kHz | 44.1/48 kHz | You can downsample later; you cannot invent high frequencies that were never recorded |
| Background | speech only | speech only | Music or effects under dialogue are the single most common cause of a ruined pack |

Game rips are usually ideal: one line per file, already segmented, no background. Anime
rips usually are not — dialogue sits over music and effects, and separation is lossy.

**A word on rights.** Extracting and training on a character's voice is not automatically
permissible just because the files are on your disk. `docs/VOICEPACK.md` has the rules
this project holds itself to; the `LICENSE` covers the code and nothing else.

---

## 1. Make the pack and the profile

```bash
make install                                   # pydantic + PyYAML + dev tools
cvai-voicepack init denia_cn --character denia_cn --display-name "迪尼娅"
cvai-character init denia_cn --name "迪尼娅" --voicepack denia_cn
```

The first writes `voicepacks/denia_cn/` — the directory skeleton and a manifest. The
second writes `characters/profiles/denia_cn.yaml` — who she is, how she speaks, which
styles exist. The profile is *words*; the pack is *voice*. They meet only through
`voice.voicepack_id`, and keeping them apart is what lets you rewrite her personality
without invalidating a trained voice.

Fill in the **dialogue examples** before anything else. V1 does not fine-tune the LLM
(spec §11), so her original lines are the only thing in the prompt carrying her voice:
personality adjectives say what to aim for, real lines show it. Record where each line
came from — a quest name, a voice file — so the set can be audited later. Paraphrased or
LLM-invented "examples" are worse than none, because they teach the model to imitate an
imitation.

```bash
cvai-character lint denia_cn --pack denia_cn
```

The linter is the cheap version of the listening test. Its central check is style
coverage, from both sides: a style in `available_styles` with no examples means the
planner will ask for it and the model will guess; a style with no reference clips means
retrieval falls back to neutral. Either way you get spec §27's "every line sounds the
same", and neither shows up as an error at runtime.

So start narrow. Two styles you can fill properly beat ten you declared ambitiously.

---

## 2. Preprocess, then *look at what it did*

```bash
pip install -e '.[preprocess]'                 # torch, FunASR, separation, loudness
cvai-prep backends                             # confirm what is actually installed
cvai-prep run denia_cn --source ~/denia_voice_lines --hotword 迪尼娅 --no-segment
```

Flags worth understanding rather than copying:

* `--no-segment` — game dialogue is usually already one line per file. Segmenting it
  again splits sentences at breaths.
* `--separate` — **only** if music or effects are mixed under the dialogue. Separation
  always costs something; paying that cost on clean speech is pure loss.
* `--denoise` — usually leave off. It attenuates breaths and mouth texture, and those are
  a large part of why a voice sounds like a person rather than a model.
* `--hotword` — repeatable. Her name and your world's proper nouns, so the transcripts do
  not quietly disagree with the audio.

Then the step people skip:

```bash
cvai-prep review denia_cn        # writes processed/review.html — open it and listen
cvai-prep apply denia_cn review-patch.json
cvai-prep build denia_cn         # clean/ + dataset.json + references.json
cvai-voicepack validate denia_cn --strict --check-profile denia_cn
```

The review page is where you catch the two failures that survive every automatic check:
**another character's line** attributed to yours, and a transcript that says something the
audio does not. Both teach the model something false, and neither shows up as a number.

`--auto-approve` exists for pipeline testing. Using it on a real pack means shipping
whatever the ASR believed.

> **Checkpoint.** `cvai-prep status denia_cn` reports approved minutes per style. If you
> have under ~20 minutes, or one style holds 90% of it, stop here and collect more audio.
> No engine choice downstream will fix a thin pack, and you will spend GPU hours
> discovering that.

---

## 3. Decide what you are comparing

`configs/benchmarks/` holds the candidate list. A candidate is an `(engine,
adaptation_mode)` pair — GPT-SoVITS fine-tuned and IndexTTS zero-shot are not competing
on equal terms, and the report keeps them apart for that reason.

Include, always:

* At least one **fine-tuned** and one **zero-shot** candidate, so you learn whether
  training was worth it for *your* pack rather than in general.
* The **null-processing control** — the same engine on unprocessed audio. This is how you
  find out if your cleaning helped or quietly sanded her voice down.
* **Real recordings** of the character, which the blind test mixes in as hidden anchors.

That last one is the whole method. Without it, listeners rate candidates against each
other and 3.8/5 looks like success; with it, you see how far the best candidate still is
from the real thing.

---

## 4. Train (this is where the GPU starts)

Each engine has a runbook in `models/<engine>/RUNBOOK.md`, and each has a sidecar
container in `infra/`:

```bash
cvai-export denia_cn gpt_sovits                # engine-specific dataset
docker compose -f infra/docker-compose.sidecars.yml up gpt_sovits
```

Held-out clips are excluded from every export automatically. **Do not add them back.**
They are the benchmark's ground truth and the listening test's hidden anchor; training on
them turns the whole evaluation into a lie you cannot detect afterwards.

Keep every checkpoint you might want to compare — two epochs of one run are a legitimate
pair of candidates, and "which epoch" is a real question the benchmark can answer.

---

## 5. Benchmark, then listen

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
cvai-bench report runs/<run_id> --objective     # prosody vs. the character herself
cvai-bench blind runs/<run_id> --webmushra      # the part that decides
cvai-bench aggregate runs/<run_id> --ratings ratings/
```

The objective metrics rank and regression-test. They do not decide. Spec §18's bar is a
human one — *could someone who knows her mistake this for a new recording?* — and the
blind test with real anchors is the only instrument that measures it.

Read the report for the failure it was built to catch: **high naturalness, low character
similarity.** A clean, pleasant, natural voice that is not her is the specific outcome
this whole apparatus exists to detect, which is why those two axes are rated separately.

Record the decision in `docs/decisions/`, including the licence position of the engine
you picked. IndexTTS is under bilibili's licence and Fish Speech's weights need
verifying; discovering that after building a product on one of them is expensive.

---

## 6. Talk to her

```bash
cvai-talk denia_cn                  # from the terminal, no browser needed
cvai-talk denia_cn --audition       # one line per style, as a page to listen through

pip install -e '.[runtime]'
uvicorn cvai_api.app:app --port 8000
open apps/web/dev-client.html
```

Run `--audition` first and listen for one thing: whether the styles are actually
different from each other. If they are not, the fault is upstream of the engine — the
Reference Bank, or styles declared in the profile that the pack cannot perform — and no
amount of retraining will fix it.

Type, or press 开麦 and speak — endpointing, transcription and barge-in all happen
server-side. Set her engine in `characters/profiles/<id>.yaml` under
`voice.preferred_engine` so the Milestone 7 decision takes effect everywhere without
touching deployment config.

For a fully local conversation, point `providers.stt.active` at `funasr` (Mandarin ASR
that runs offline and takes her name as a real hotword) and your chosen engine's sidecar
at `providers.tts`.

---

## When it sounds wrong

| Symptom | Where it usually comes from |
|---|---|
| Every line is the same register | One style dominates the pack, or the Reference Bank has no clips for the style being requested |
| Natural but not her | The engine is generalising; check character-similarity ratings separately from naturalness, and compare against the null-processing control |
| Beautiful on training lines, flat on new ones | Overfitting. The unseen-sentence benchmark is what surfaces it — do not judge by ear on training text |
| Numbers, dates or Latin read wrongly | The Chinese front-end, not the engine. `cvai_text_normalizer` — and add a test for the case |
| She interrupts herself | Echo. Headphones, or check `echoCancellation` in the client |
| She answers half a sentence | Endpointing. Raise `silence_end_ms`; 700 ms suits most Mandarin speakers, slower speakers need more |
| The ASR mangles her name | Add it to the character profile's `frequent_expressions` — hotwords are built from the profile |
