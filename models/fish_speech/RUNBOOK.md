# Fish Speech / OpenAudio S1 — Milestone 5

LoRA fine-tuning on the LLaMA component, with a two-step cloning flow: VQ tokens are
extracted from the reference audio, then generation is conditioned on those tokens plus
the reference text.

**Licence gate.** The code is open; the *weight* licence must be confirmed before any
commercial use. `commercial_use` is `null` for this candidate in the benchmark config
rather than guessed, and Milestone 7's decision record has to resolve it if this engine
wins. Check before spending GPU hours, not after.

---

## 1. Export

```bash
cvai-export denia_cn fish_speech
# → voicepacks/denia_cn/datasets/fish_speech/data/denia_cn/
#     <sample_id>.wav  +  <sample_id>.lab   (transcript, one per clip)
```

## 2. Prepare and train

```bash
pip install fish-audio-preprocess      # loudness normalization, if needed

python tools/vqgan/extract_vq.py data --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"

python tools/llama/build_dataset.py --input "data" --output "data/protos" \
    --text-extension .lab --num-workers 16

python fish_speech/train.py --config-name text2semantic_finetune \
    project=denia_cn +lora@model.model.lora_config=r_8_alpha_16

python tools/llama/merge_lora.py --lora-config r_8_alpha_16 \
    --base-weight checkpoints/openaudio-s1-mini \
    --lora-weight results/denia_cn/checkpoints/step_000000010.ckpt \
    --output checkpoints/openaudio-s1-mini-denia/
```

Two upstream constraints worth restating: **LoRA only**, and **do not fine-tune an
RL-trained checkpoint**. Both are easy to violate by picking the wrong base weights.

## 3. Pre-extract Reference Bank tokens

This engine needs VQ tokens for every reference clip. Extracting them per request is
10–30× slower, and — worse — the cost is invisible in a benchmark's timings because it
looks like the model is simply slow.

```bash
python tools/vqgan/extract_vq.py voicepacks/denia_cn/references \
    --config-name modded_dac_vq \
    --checkpoint-path checkpoints/openaudio-s1-mini/codec.pth
```

Then record each `.npy` in the reference bank so the adapter can pass it through:

```json
{
  "reference_id": "soft_teasing_01",
  "audio_path": "references/soft_teasing/soft_teasing_01.wav",
  "transcript": "哦？这么快就回来了。",
  "precomputed": {
    "fish_speech_prompt_tokens": "references/soft_teasing/soft_teasing_01.npy"
  }
}
```

The adapter **fails loudly** when tokens are missing rather than falling back to
on-the-fly extraction, for exactly the reason above. `require_precomputed_tokens: false`
disables that check if you knowingly want the slow path.

## 4. Register and serve

```yaml
checkpoints:
  - checkpoint_id: denia_fish_lora_v1
    engine: fish_speech
    engine_version: openaudio-s1-mini
    adaptation_mode: lora
    paths: { merged: checkpoints/fish_speech/openaudio-s1-mini-denia }
```

```bash
docker compose -f infra/docker-compose.sidecars.yml up fish_speech
```

## 5. Benchmark

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
```
