# Qwen3-TTS — Milestone 4

Apache-2.0 weights, official fine-tuning scripts, ~97 ms streaming latency, and an
`instruct` channel for style. Operationally the most comfortable of the strong
candidates: nothing about its licence needs a lawyer before Milestone 7.

Sizes: 0.6B and 1.7B, each in Base / CustomVoice / VoiceDesign variants. **Use Base** —
CustomVoice offers nine preset timbres, which is the opposite of what this project
wants.

---

## 1. Export

```bash
cvai-export denia_cn qwen3_tts
# → voicepacks/denia_cn/datasets/qwen3_tts/
#     train_raw.jsonl   {audio, text, speaker, language, style, duration}
#     val_raw.jsonl
#     audio/
#     reference.wav + reference.txt    best neutral clip, for inference
```

The official flow transcribes with WhisperX to build `train_raw.jsonl`. Ours is already
transcribed and human-reviewed, so that step is skipped — and skipping it avoids a second
transcription that disagrees with the dataset.

## 2. Extract codes and train

```bash
pip install -U qwen-tts
git clone https://github.com/QwenLM/Qwen3-TTS && cd Qwen3-TTS/finetuning
# train_raw.jsonl → train_with_codes.jsonl (16-layer codec tokens), then train
```

Starting point from the community pipeline: batch size 2, learning rate 2e-5, 3 epochs,
~16 GB VRAM. 10–100 samples already work for a single speaker, so run a short pass on a
subset *before* committing the whole pack — it surfaces format problems in minutes
rather than hours.

Reduce batch size to 1 before anything else if you hit OOM.

## 3. Register

```yaml
checkpoints:
  - checkpoint_id: denia_qwen3_lora_v1
    engine: qwen3_tts
    engine_version: Qwen3-TTS-12Hz-1.7B-Base
    adaptation_mode: lora
    paths: { lora: checkpoints/qwen3_tts/denia_lora_v1 }
    trained_on_dataset_version: 0.1.0
    training_config: { lr: 2e-5, epochs: 3, batch_size: 2 }
```

## 4. Serve

```bash
docker compose -f infra/docker-compose.sidecars.yml up qwen3_tts
curl -s localhost:9881/health | jq
```

## 5. Benchmark

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml
```

Run **three** rows for this engine, not one:

| candidate | `auto_instruct` | what it isolates |
|---|---|---|
| `qwen3_1b7_zs` | on | zero-shot baseline |
| `qwen3_1b7_zs_noinstruct` | off | how much the instruction channel is actually contributing |
| `qwen3_1b7_ft` | on | the fine-tune |

The middle row is the like-for-like comparison against GPT-SoVITS, which has no
instruction channel at all. Without it, a win for Qwen3-TTS cannot be attributed to
either the model or the extra control surface.

---

## Implementation note worth remembering

`create_voice_clone_prompt(ref_audio, ref_text)` builds a reusable prompt object, and
rebuilding it per sentence is this engine's version of loading weights from disk
(spec §27). The client adapter sends `voice_clone_prompt_key` — the Reference Bank clip
id — and the sidecar caches on it, so rotating through six reference clips costs six
builds rather than one per line.

The cache is cleared on every checkpoint switch. A prompt object built against the old
weights silently produces the old voice, which is a genuinely nasty bug to find from the
audio alone.
