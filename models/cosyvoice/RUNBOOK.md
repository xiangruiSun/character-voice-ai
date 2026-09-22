# Fun-CosyVoice3-0.5B — Milestone 5c

Apache-2.0 code *and* weights, an `instruct` channel for style, and the strongest
streaming story of the candidates — around 150 ms, which is the number that matters for
Milestone 12 when the character has to start speaking before the reply is finished.

It earns its place in the benchmark for a specific reason: if it comes close to the
fine-tuned winner while streaming at 150 ms, the product trade-off is real rather than
theoretical, and a slightly-less-similar voice that answers immediately may be the better
character. That is a judgement for the listening test, not for a latency table.

---

## 1. Get the engine and the weights

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/cosyvoice/src
cd models/cosyvoice/src && git submodule update --init --recursive
conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice
pip install -r requirements.txt
```

```python
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                  local_dir="pretrained_models/Fun-CosyVoice3-0.5B")
snapshot_download("FunAudioLLM/CosyVoice-ttsfrd",
                  local_dir="pretrained_models/CosyVoice-ttsfrd")
```

`CosyVoice-ttsfrd` is its own Chinese text front-end. **We do not use it.** Our normalizer
(`cvai_text_normalizer`) runs before the engine, and two front-ends in series normalise
the same string twice — numbers already read as words get re-read, and the second pass is
the one nobody can see. The sidecar sends text that is already TTS-ready.

## 2. Start the sidecar

```bash
docker compose -f infra/docker-compose.sidecars.yml up cosyvoice
```

The sidecar wraps `cosyvoice.cli.cosyvoice.AutoModel` and routes by what the request
carries:

| Request | Method |
|---|---|
| reference clip + its transcript | `inference_zero_shot` |
| reference clip + a style instruction | `inference_instruct2` |
| reference clip, no transcript | `inference_cross_lingual` |

Its inference methods are generators yielding `{"tts_speech": tensor}` chunks; the
sidecar concatenates them for a non-streaming request and passes them through for a
streaming one. Prompt audio is loaded once and cached per path — re-reading the same
reference WAV for every line of a conversation is pure waste.

## 3. Zero-shot row first

```bash
cvai-bench run configs/benchmarks/denia_cn_v1.yaml     # cosyvoice3_zs is enabled
```

It costs nothing and sets the bar the SFT row has to clear. An SFT run that does not beat
its own engine's zero-shot row is telling you something about the dataset, not the engine.

## 4. SFT

```bash
cvai-export denia_cn cosyvoice
# → voicepacks/denia_cn/datasets/cosyvoice/
#     train/{wav.scp,text,utt2spk,spk2utt}
#     dev/{wav.scp,text,utt2spk,spk2utt}
#     audio/
```

Kaldi-style directories, which is what the training recipes in the upstream
`examples/` tree consume. Held-out samples are excluded automatically — they are the
benchmark's ground truth and the listening test's hidden anchor, and training on them
invalidates every number that follows, invisibly.

The upstream recipes are stage-based (`examples/libritts/...`): extract embeddings and
speech tokens, build parquet shards, then train the LLM and flow-matching stages. Follow
the recipe in the version of the repo you actually checked out rather than a command
copied from here — the stage numbering has changed between releases, and a stale
instruction that half-works is worse than a pointer.

Practical notes:

* Train the **flow/acoustic** stage before deciding the model cannot do the voice. Most
  of the timbre lives there.
* Keep every checkpoint worth comparing. Two epoch counts of one run are a legitimate
  pair of benchmark candidates.

## 5. Register the checkpoint

```yaml
- candidate_id: cosyvoice3_sft
  engine: cosyvoice
  adaptation_mode: finetuned
  checkpoint_id: /models/cosyvoice/denia_sft_v1    # model_dir the sidecar loads
  display_name: Fun-CosyVoice3 0.5B (SFT)
  license: Apache-2.0
  commercial_use: true
```

`checkpoint_id` is passed to the sidecar's `load_checkpoint`, which is a `model_dir` swap
followed by a reload, so the value is a path inside the container.

## 6. What to listen for

* **Streaming artefacts at chunk boundaries.** The benchmark generates whole lines, so it
  will not show them. Check them with `cvai-talk denia_cn --verbose`, which speaks the
  chunked text the way the conversation actually will.
* **Instruction leakage.** With `inference_instruct2`, a style instruction occasionally
  gets *spoken* rather than performed. Rare, obvious, and fatal to the illusion — worth
  one deliberate pass through the generated files before a listening test goes out.

---

Sources: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) ·
[Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
