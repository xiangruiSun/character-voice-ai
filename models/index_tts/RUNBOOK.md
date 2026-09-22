# IndexTTS-2.5 — Milestone 5b (zero-shot only)

The one candidate whose **emotion is decoupled from speaker identity**: an 8-float vector
sets the performance while the speaker prompt sets the voice. Every other engine in this
benchmark carries emotion inside the reference clip, which means a sad line needs a sad
reference clip — and a Voice Pack whose "sad" style has three clips can only ever be sad
three ways. That is worth measuring even though the engine cannot be fine-tuned.

Two constraints that decide how it is used here:

* **No fine-tuning path.** It competes zero-shot only, and the report keeps it in the
  zero-shot column where that is a fair comparison (decision D4).
* **The weights are under the bilibili Model Use License, not an OSI licence.** The
  benchmark config records `commercial_use: null` so it cannot silently win Milestone 7.
  Read the licence before selecting it; some redistributions of these weights are marked
  non-commercial.

---

## 1. Get the engine and the weights

```bash
git clone https://github.com/index-tts/index-tts models/index_tts/src
cd models/index_tts/src
pip install -U uv && uv sync --all-extras

uv tool install "huggingface-hub"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

Python 3.10–3.11, and roughly 6 GB of VRAM — the least demanding candidate, and the one
most likely to run on a laptop GPU.

## 2. Start the sidecar

```bash
docker compose -f infra/docker-compose.sidecars.yml up index_tts
```

The sidecar wraps `indextts.infer_v2_5.IndexTTS2` and exposes the contract in
`docs/SIDECAR_PROTOCOL.md`. Nothing else in this project imports it (decision D1): the
engine's dependencies stay inside its own container.

## 3. Run the zero-shot row

No export, no training. The adapter takes the reference clip the retriever chose as
`spk_audio_prompt`, sets `lang="ZH"`, and maps `StyleControls` onto `emo_vector`:

```
happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
```

`cvai_types.style.EmotionVector` is written in exactly that order, which is why it exists
as a typed thing rather than a bare list — a silently permuted vector produces a
confidently wrong performance, and nothing downstream would notice.

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
```

## 4. What to listen for

The question this candidate is in the benchmark to answer is **not** "is it the best
voice?" — a zero-shot engine competing against fine-tunes usually is not. It is:

> Does explicit emotion control produce a *wider and more accurate* range than
> reference-clip conditioning does, for this character?

If yes, that matters even if IndexTTS loses overall, because it says the winning engine's
limitation is style coverage rather than timbre — and style coverage is fixable with more
reference clips, while timbre is not.

Watch for the opposite too: emotion vectors applied to a voice the model has never been
trained on can drift the timbre away from the speaker prompt, which reads in the
listening test as high naturalness and low speaker similarity.

## 5. If it wins

It cannot be selected without resolving the licence, and that resolution belongs in
`docs/decisions/` with the date, the licence version read, and the intended use. A
Milestone 7 decision record that says "IndexTTS won" and nothing about the licence is not
a decision, it is a deferral.

---

Sources: [index-tts/index-tts](https://github.com/index-tts/index-tts) ·
[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) ·
[IndexTTS 2.5 technical report](https://index-tts.github.io/index-tts2-5.github.io/)
