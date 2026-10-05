# Training a character voice (GPT-SoVITS v2ProPlus)

卡提希娅's voice is fine-tuned on her own lines from the AI Hobbyist dataset — not
zero-shot cloning. The voice server loads the trained weights at startup.

| | Zero-shot (before) | Fine-tuned (now) |
|---|---|---|
| Syllables dropped, 20 unseen sentences | 1 of 456 | **0 of 456** |
| Pinyin error rate (ASR on her output) | 2.9 % | **2.0 %** |
| Character error rate | 3.9 % | **3.1 %** |
| Speaker similarity to 15 held-out real lines | 0.781 | **0.820** |

For scale, her real recordings score 0.797 against each other on the same measure.

## What was trained

- Data: `C:\Users\gaming\datasets\cartethyia` — 370 `.wav` + `.lab` pairs (2.3–2.4 main
  story, 44.1 kHz, ~28 min). 355 kept (0.8–15 s), **15 held out** and never trained on
  (`models/gpt_sovits/src/logs/cartethyia/heldout.list`).
- SoVITS (timbre): 8 epochs, batch 8 → `SoVITS_weights_v2ProPlus/cartethyia_e8_s360.pth`
- GPT (prosody): 15 epochs, batch 8 → `GPT_weights_v2ProPlus/cartethyia-e15.ckpt`
  (e5 and e10 kept for comparison)
- About 5 minutes in total on an RTX 4090.

## Retrain, or train another character

The source folder only needs `name.wav` + `name.lab` pairs (the dataset's format).

```powershell
cd C:\Users\gaming\character-voice-ai
.\models\gpt_sovits\src\.venv\Scripts\python.exe scripts\train_gpt_sovits.py C:\Users\gaming\datasets\cartethyia cartethyia
```

Options: `--sovits-epochs 8 --gpt-epochs 15 --batch-size 8 --holdout 15`. Each stage is
skipped when its output exists; delete `models\gpt_sovits\src\logs\<name>` to start over.

## Use the trained voice

Set the weights in `models\gpt_sovits\src\GPT_SoVITS\configs\tts_infer.yaml` under
`custom:` (paths relative to `models\gpt_sovits\src`), then restart the voice server:

```yaml
  t2s_weights_path: GPT_weights_v2ProPlus/cartethyia-e15.ckpt
  vits_weights_path: SoVITS_weights_v2ProPlus/cartethyia_e8_s360.pth
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_voice_server.ps1
```

To go back to zero-shot, point them at `GPT_SoVITS/pretrained_models/s1v3.ckpt` and
`GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth`.

## Check that every word is spoken

```powershell
.\.venv\Scripts\python.exe scripts\eval_voice_coverage.py --label my-test
# compare weights without editing the config:
.\.venv\Scripts\python.exe scripts\eval_voice_coverage.py --label e10 `
    --gpt GPT_weights_v2ProPlus/cartethyia-e10.ckpt --sovits SoVITS_weights_v2ProPlus/cartethyia_e8_s360.pth
```

It runs 20 reply-style sentences through the app's own normalizer and chunker,
synthesizes them, transcribes them locally, and lists any dropped syllables. Audio and a
JSON report land in `runs\voice_eval\<label>\`. Note `--gpt/--sovits` switch the running
server's weights until it is restarted.

In the app, `truncate_long_replies: false` in her profile means a reply longer than the
120 characters the LLM is asked for is spoken whole rather than cut.

## Windows notes

- The GPT-SoVITS environment uses **PyTorch 2.7.1** (cu128). PyTorch 2.11's `gloo`
  backend crashes with an access violation on the first training step on Windows
  (SoVITS training uses single-process DDP), and 2.7.1 does not.
- `jieba_fast` is a shim over `jieba` (no C compiler needed); identical segmentation.
