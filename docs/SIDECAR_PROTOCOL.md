# Engine sidecar protocol

Every TTS engine runs in its own process with its own Python environment
(decision D1). The six candidates pin mutually incompatible torch and CUDA versions, so
importing any of them into the application process would make it impossible to run the
others — and the benchmark's entire purpose is running all of them.

Two integration shapes, depending on what the engine already offers.

---

## A. Engines with their own HTTP API

**GPT-SoVITS** (`api_v2.py`) and **Fish Speech** (`tools/api_server.py`) ship servers.
The adapter speaks their native API directly; nothing is wrapped. Endpoints and the full
parameter schema for GPT-SoVITS are pinned in `configs/providers` / the adapter, and
recorded in `docs/TECH_LANDSCAPE.md`.

The important one is checkpoint hot-swap: `/set_gpt_weights` and `/set_sovits_weights`
let a character checkpoint be switched without restarting the process. That is what makes
"do not load the model from disk for every sentence" (spec §27) achievable rather than
aspirational, and it is why `ProviderCapabilities.supports_hot_checkpoint_swap` exists.

---

## B. Engines used through their Python API — the CVAI sidecar

**Qwen3-TTS, IndexTTS-2.5, CosyVoice3 and VoxCPM2** are called as Python libraries. Each
gets a thin HTTP wrapper in its own container.

The split of responsibility is the point:

* **The adapter, in this repository**, translates an engine-neutral `TTSRequest` into
  *that engine's own keyword arguments*. That mapping is the interesting part — it is
  where `speaking_rate: slow` becomes `duration_factor: 1.176` for IndexTTS and
  `speed: 0.85` for CosyVoice — and it stays in version control, next to the capability
  declaration it has to agree with, covered by unit tests.
* **The sidecar** loads the model once and calls it with the arguments it is handed. It
  contains no style logic, no defaults worth arguing about, and no knowledge of
  characters or voice packs.

### Endpoints

```
GET  /health
     → 200 {"engine": "qwen3_tts", "version": "...", "ready": true}

POST /v1/checkpoint
     ← {"checkpoint_id": "denia_v3_lora"}
     → 200 {"loaded": "denia_v3_lora"}

POST /v1/synthesize
     ← {
         "call": "generate_voice_clone",
         "kwargs": { ... engine-native arguments ... },
         "seed": 1234,
         "sample_rate": null
       }
     → 200 audio/wav
       X-CVAI-Sample-Rate: 24000
       X-CVAI-Engine-Version: Qwen3-TTS-12Hz-1.7B-Base
```

`call` names the engine method, so one image can expose every entry point an engine has
(`generate_voice_clone`, `inference_zero_shot`, `infer`, `generate`) without the protocol
growing a field each time a new one appears.

### What a sidecar must do

1. **Load the model once, at start-up.** Never per request.
2. **Cache per-reference artifacts.** For Qwen3-TTS that is the object from
   `create_voice_clone_prompt(ref_audio, ref_text)`, keyed by the
   `voice_clone_prompt_key` the adapter sends — re-deriving it per sentence is this
   engine's version of reloading from disk. For Fish Speech the equivalent (VQ prompt
   tokens) is precomputed at pack build time and passed by path.
3. **Honour the seed** so a run is reproducible.
4. **Return WAV** with an accurate `X-CVAI-Sample-Rate`. Do not resample unless asked:
   the 48 kHz output of VoxCPM2 and GPT-SoVITS v4 is part of what is being judged.
5. **Fail loudly.** A 4xx/5xx with a readable body is recorded against the candidate and
   the run continues. Silently returning a near-empty or default-voice clip is far worse
   than an error, because it reaches the listening test looking like a result.

### Error semantics the adapter relies on

| Condition | Adapter raises | Benchmark behaviour |
|---|---|---|
| connection refused / timeout | `ProviderUnavailableError` | skip the candidate, record the reason |
| HTTP 4xx / 5xx | `SynthesisError` | record this generation as failed, continue |
| empty body | `SynthesisError` | same |

---

## Ports

Defaults in `configs/app.yaml`, each overridable by environment variable so a container
compose file does not need to edit config:

| Engine | Port | Variable |
|---|---|---|
| GPT-SoVITS | 9880 | `GPT_SOVITS_URL` |
| Qwen3-TTS | 9881 | `QWEN3_TTS_URL` |
| IndexTTS | 9882 | `INDEX_TTS_URL` |
| CosyVoice | 9883 | `COSYVOICE_URL` |
| VoxCPM | 9884 | `VOXCPM_URL` |
| Fish Speech | 8888 | `FISH_SPEECH_URL` |

---

## Adding an engine

1. Write the adapter in `providers/tts/`, decorated with `@TTS_PROVIDERS.register("key")`.
2. Declare `ProviderCapabilities` honestly — especially `requires_reference_text`,
   `supports_instruct`, `supports_emotion_vector` and the licence fields. The benchmark
   and the router read this instead of special-casing engine names, and an over-claimed
   capability produces audio that quietly ignores half the request.
3. Implement `native_kwargs` (sidecar) or `synthesize` (native API).
4. Add an instance to `configs/app.yaml` and a candidate to the benchmark config.
5. Add a payload-mapping test. It needs no GPU, and it catches the failure mode that is
   otherwise invisible: audio that renders fine but is not what was asked for.
