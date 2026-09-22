# VoxCPM2 — Milestone 5d

Apache-2.0 for both code and weights, **48 kHz output**, and a LoRA path that its own
tooling claims works from five to ten minutes of audio. For a character voice pack that
is the interesting property: most of this project's risk is "not enough clean audio", and
an engine that adapts from ten minutes changes what a thin pack is worth.

The 48 kHz output matters more than it sounds. Breath, sibilance and mouth texture live
above 12 kHz, and they are a large part of why a voice sounds like a person rather than a
model. A 24 kHz engine cannot reproduce them at all; whether that is audible for *this*
character is exactly what the listening test is for.

---

## 1. Get the engine and the weights

```bash
pip install voxcpm            # Python ≥3.10 <3.13, PyTorch ≥2.5, CUDA ≥12
```

```python
from voxcpm import VoxCPM
tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
```

`load_denoiser=False` is deliberate and matters here. Our clips have already been through
a human-reviewed cleaning chain whose `processing_chain` is recorded per clip (decision
D7); a second, unrecorded denoiser on top of that is exactly the over-processing spec §7
warns about, and it would silently invalidate the null-processing control.

## 2. Start the sidecar

```bash
docker compose -f infra/docker-compose.sidecars.yml up voxcpm
```

Three cloning tiers, in increasing fidelity:

| Inputs | What it does |
|---|---|
| `reference_wav_path` | timbre only |
| `prompt_wav_path` + `prompt_text` | prompt-based cloning, with the transcript |
| both together | its highest tier |

This is why `ReferenceSample.transcript` is required rather than optional (decision D3).
A Reference Bank without transcripts cannot reach the top tier, and rebuilding one means
listening to every clip again.

## 3. Zero-shot row first

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml     # voxcpm2_zs is enabled
```

Its zero-shot row is also the fairest test of the 48 kHz claim, because no training has
had a chance to narrow the band.

## 4. LoRA or full SFT

```bash
cvai-export denia_cn voxcpm
# → voicepacks/denia_cn/datasets/voxcpm/
#     train.jsonl   {audio, text, speaker, style, prompt_audio, prompt_text}
#     val.jsonl
#     audio/, reference.wav
```

```bash
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml    # LoRA
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml     # full SFT
```

There is also a `lora_ft_webui.py` for a click-through version of the same thing.

**Run LoRA first, even if you have enough data for full SFT.** It is faster, it cannot
catastrophically forget the base model's prosody, and if LoRA already matches the
character then full SFT is spending GPU hours to risk a regression. Both are legitimate
benchmark candidates; run them as two rows rather than choosing by argument.

Point the recipe's dataset config at the exported `train.jsonl` / `val.jsonl`. Held-out
samples are excluded from the export automatically, and must not be added back.

## 5. Register the checkpoint

```yaml
- candidate_id: voxcpm2_lora
  engine: voxcpm
  adaptation_mode: lora
  checkpoint_id: /models/voxcpm/denia_lora_v1
  display_name: VoxCPM2 (LoRA, 10 min)
  license: Apache-2.0
  commercial_use: true
  notes: >
    Trained on a deliberately small subset to test the low-data claim. Compare against
    voxcpm2_sft on the full pack — if they are indistinguishable, the pack is larger
    than it needs to be and the extra audio is better spent on style coverage.
```

That note describes a genuinely useful experiment: a LoRA on ten minutes versus one on
the whole pack answers "do we need more audio, or more *varied* audio?" — and those have
very different costs.

## 6. What to listen for

* **The output really being 48 kHz.** `cvai-bench` records the sample rate per file; if
  the benchmark's `output_sample_rate` resamples everything to a common rate for fairness,
  listen to the raw files too, because that resample is where the advantage disappears.
* **Over-smooth breath.** Tokeniser-free models can render breath beautifully or erase it
  entirely. The null-processing control row is the comparison that shows which, since it
  is conditioned on clips that never went through our cleaning chain.

---

Sources: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) ·
[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
