# Engine checkouts

One directory per candidate engine. Each holds a pinned checkout (submodule or clone),
a `Dockerfile` for its sidecar, and the training runbook for this project's voice packs.
Engine source is not vendored into git — see `.gitignore`.

| Directory | Engine | Milestone | Integration |
|---|---|---|---|
| `gpt_sovits/` | GPT-SoVITS v4 / v2Pro | 3 | native `api_v2.py` |
| `qwen3_tts/` | Qwen3-TTS 0.6B / 1.7B | 4 | CVAI sidecar |
| `fish_speech/` | Fish Speech / OpenAudio S1 | 5 | native `api_server.py` |
| `index_tts/` | IndexTTS-2.5 | 5b | CVAI sidecar |
| `cosyvoice/` | Fun-CosyVoice3 0.5B | 5c | CVAI sidecar |
| `voxcpm/` | VoxCPM2 | 5d | CVAI sidecar |

The sidecar contract is in `docs/SIDECAR_PROTOCOL.md`.
