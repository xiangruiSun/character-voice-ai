# Engine checkouts

One directory per candidate engine: a pinned checkout (not vendored into git — see
`.gitignore`), the training runbook for this project's voice packs, and any weights.
Container definitions live in `infra/sidecars/`, and the HTTP contract in
`docs/SIDECAR_PROTOCOL.md`.

| Directory | Engine | Milestone | Integration | Adaptation | Licence |
|---|---|---|---|---|---|
| `gpt_sovits/` | GPT-SoVITS v4 / v2Pro | 3 | native `api_v2.py` | fine-tune | MIT |
| `qwen3_tts/` | Qwen3-TTS 0.6B / 1.7B | 4 | CVAI sidecar | fine-tune / LoRA | Apache-2.0 |
| `fish_speech/` | Fish Speech / OpenAudio S1 | 5 | native `api_server.py` | LoRA | **verify weights** |
| `index_tts/` | IndexTTS-2.5 | 5b | CVAI sidecar | none (zero-shot) | **bilibili, not OSI** |
| `cosyvoice/` | Fun-CosyVoice3 0.5B | 5c | CVAI sidecar | SFT | Apache-2.0 |
| `voxcpm/` | VoxCPM2 | 5d | CVAI sidecar | SFT / LoRA | Apache-2.0 |

Runbooks exist for the three engines the brief named (`gpt_sovits`, `qwen3_tts`,
`fish_speech`). The optional three are zero-shot rows first; their setup is one
`docker compose up` each and needs no training pass.

---

## Order of work

1. **Zero-shot rows for every engine first.** They need no training, they cost an hour
   in total, and they set the bar each fine-tune has to clear. A fine-tune that does not
   beat its own engine's zero-shot row is a signal about the data, not the engine.
2. **Then fine-tune, in brief order:** GPT-SoVITS, Qwen3-TTS, Fish Speech.
3. **Keep every checkpoint worth comparing.** Two epoch counts of one run are a
   legitimate pair of benchmark candidates, and re-training to recover a deleted one is
   far more expensive than the disk.

## Two things that are easy to get wrong

**Never train on the held-out split.** The exporters enforce this and a test checks each
one, but if you build a dataset by hand, remember that those lines are the benchmark's
ground truth *and* the listening test's hidden anchor. Training on them invalidates every
number that follows, invisibly.

**Licences decide eligibility, not just conscience.** IndexTTS-2.5 ships under a bilibili
model-use licence and Fish Speech's weight licence is unconfirmed. Both carry
`commercial_use: null` in the benchmark config so a licence-incompatible engine cannot
win by accident — Milestone 7's decision record has to resolve it before selection.
