# GPT-SoVITS — Milestone 3

The community standard for anime and game character voices, and the candidate with the
strongest fine-tuning story. MIT licensed, commercial use permitted.

Version choice: **v4** for the best reported timbre similarity (0.735 against a 0.750
ground-truth ceiling, up from 0.549 at v2) and 48 kHz native output. If the Denia source
audio turns out rough, add **v2Pro** as a second candidate rather than switching — the
maintainers note v2 handles poor-quality datasets better, and the benchmark exists
precisely to settle that kind of question with evidence.

---

## 1. Export the dataset

```bash
cvai-export denia_cn gpt_sovits
# → voicepacks/denia_cn/datasets/gpt_sovits/
#     denia_cn.list          audio_path|speaker|ZH|text
#     audio/                 copied from clean/
#     export.json            pack version + split counts
```

Held-out samples are excluded automatically. Do not add them back: they are the
benchmark's ground truth and the listening test's hidden anchor.

## 2. Get the engine

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS models/gpt_sovits/src
cd models/gpt_sovits/src
# follow upstream install; pretrained weights land in GPT_SoVITS/pretrained_models/
```

Skip its bundled UVR5 / slicer / ASR steps. Our pack is already segmented, transcribed
and human-reviewed, and re-deriving transcripts here would produce a second set that
disagrees with the dataset.

## 3. Train

Two stages, in this order:

1. **SoVITS** — the acoustic side. Most of the timbre comes from here.
2. **GPT** — the semantic/prosody side.

Point the WebUI (or `s2_train.py` / `s1_train.py`) at the exported `.list` file. Start
with upstream defaults; the parameters worth touching first are epoch count and batch
size, in that order.

Practical notes:

* Watch for overfitting on a small pack. The symptom is that the model reproduces the
  training lines beautifully and everything else sounds flat — which is exactly what the
  unseen-sentence benchmark is designed to catch, so check it there rather than by ear
  on training text.
* Keep every checkpoint you might compare. Two epochs of the same run are a legitimate
  pair of benchmark candidates.

## 4. Register the checkpoint

GPT-SoVITS has two weight files and both must be switched together, so this project
treats them as one id:

```yaml
# voicepacks/denia_cn/metadata/voicepack.yaml
checkpoints:
  - checkpoint_id: "SoVITS_weights_v4/denia_e8.pth|GPT_weights_v4/denia-e15.ckpt"
    engine: gpt_sovits
    engine_version: v4
    adaptation_mode: finetuned
    paths:
      gpt: checkpoints/gpt_sovits/denia-e15.ckpt
      sovits: checkpoints/gpt_sovits/denia_e8.pth
    trained_on_dataset_version: 0.1.0
    training_config: { epochs_gpt: 15, epochs_sovits: 8, batch_size: 4 }
```

The `checkpoint_id` is what the adapter passes to `/set_gpt_weights` and
`/set_sovits_weights`, split on `|`. Getting these two out of step produces a voice that
is subtly wrong in a way that is very hard to diagnose from the audio.

## 5. Serve

```bash
docker compose -f infra/docker-compose.sidecars.yml up gpt_sovits
curl -s localhost:9880/docs >/dev/null && echo ok
```

## 6. Benchmark

Enable the fine-tuned candidate in `configs/benchmarks/denia_cn_v1.yaml` and fill in its
`checkpoint_id`:

```yaml
  - candidate_id: gpt_sovits_v4_ft
    engine: gpt_sovits
    adaptation_mode: finetuned
    checkpoint_id: "SoVITS_weights_v4/denia_e8.pth|GPT_weights_v4/denia-e15.ckpt"
    enabled: true
```

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
```

Keep `gpt_sovits_v4_zs` (zero-shot) enabled alongside it. The zero-shot → fine-tuned
delta for the *same engine* is the cleanest evidence the project has for its central
claim that character adaptation beats speaker embedding, and it costs one extra row.

---

## What this engine cannot do

Style reaches GPT-SoVITS **only** through the choice of reference clip — no instruction
string, no emotion vector. That is why the Reference Bank carries several clips per
style, and why `ProviderCapabilities` reports `supports_instruct=False` and
`supports_emotion_vector=False`: anything the planner sends on those channels is dropped,
and the run report lists it rather than letting the listening test attribute the
difference to the model.
