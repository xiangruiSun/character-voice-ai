"""Fine-tune GPT-SoVITS (v2ProPlus) on one character's wav + lab voice lines.

Reproduces the WebUI's one-click pipeline without the WebUI: build the training list,
run the three dataset-preparation stages (text/BERT, HuBERT + 32 kHz audio + speaker
embedding, semantic tokens), then train SoVITS (timbre) and GPT (prosody).

Run with the GPT-SoVITS environment's Python, from anywhere:

    models\\gpt_sovits\\src\\.venv\\Scripts\\python.exe scripts\\train_gpt_sovits.py ^
        C:\\Users\\gaming\\datasets\\cartethyia cartethyia

Each stage is skipped when its output already exists, so a failed run resumes. Weights
land in models/gpt_sovits/src/SoVITS_weights_v2ProPlus and GPT_weights_v2ProPlus, and
are copied to ``--output-dir`` when given.

With ``--events`` the script also prints one ``@@CVAI {json}`` line per progress event
(stage changes, completed epochs, losses, saved checkpoints) — the structured feed the
Studio's training worker consumes, so nothing downstream parses console text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "models" / "gpt_sovits" / "src"
VERSION = "v2ProPlus"
PRETRAINED = SRC / "GPT_SoVITS" / "pretrained_models"
EVENTS = False
TAIL: list[str] = []                     # last lines of child output, for failure reports

_SOVITS_EPOCH = re.compile(r"====> Epoch: (\d+)")
# Lightning prints "Epoch 3: 40%|…" (tqdm) or "Epoch 3/7 ━━━ …" (rich), 0-based.
_GPT_EPOCH = re.compile(r"Epoch (\d+)[:/]")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_LOSS = re.compile(r"(total_loss\w*|loss_gen\w*|loss_mel\w*)[=:]\s*([0-9.]+(?:e[+-]?\d+)?)")


def emit(event: dict) -> None:
    if EVENTS:
        print("@@CVAI " + json.dumps(event, ensure_ascii=False), flush=True)


def ffmpeg_shared_dir() -> str | None:
    # Newer torchaudio decodes through torchcodec, which needs FFmpeg's shared DLLs.
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for dll in root.glob("Gyan.FFmpeg.Shared*/**/avcodec-*.dll"):
        return str(dll.parent)
    return None


def run(stage: str, script: str, env: dict[str, str], *args: str,
        phase: str | None = None, total_epochs: int = 0) -> None:
    print(f"\n==== {stage} ====", flush=True)
    full_env = {**os.environ, **env}
    process = subprocess.Popen(
        [sys.executable, "-s", script, *args], cwd=SRC, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    last_epoch = 0
    buffer = b""
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk
        # Progress bars redraw with \r; treat it as a line break too.
        *lines, buffer = re.split(rb"[\r\n]", buffer)
        for raw in lines:
            line = _ANSI.sub("", raw.decode("utf-8", "replace")).strip()
            if not line:
                continue
            print(line, flush=True)
            TAIL.append(line)
            del TAIL[:-60]
            if phase == "sovits" and (m := _SOVITS_EPOCH.search(line)):
                epoch = int(m.group(1))
                if epoch > last_epoch:
                    last_epoch = epoch
                    emit({"type": "epoch_completed", "phase": phase, "epoch": epoch,
                          "total_epochs": total_epochs})
            elif phase == "gpt" and (m := _GPT_EPOCH.search(line)):
                # Lightning shows the 0-based epoch in progress: seeing "Epoch N" means
                # N epochs are done.
                epoch = int(m.group(1))
                if epoch > last_epoch:
                    last_epoch = epoch
                    emit({"type": "epoch_completed", "phase": phase, "epoch": epoch,
                          "total_epochs": total_epochs})
            if phase and (losses := _LOSS.findall(line)):
                emit({"type": "metric", "phase": phase,
                      "data": {k: float(v) for k, v in losses[:3]}})
    code = process.wait()
    if code != 0:
        emit({"type": "failed", "stage": stage, "message": f"exit {code}",
              "data": {"tail": TAIL[-40:]}})
        sys.exit(f"{stage} failed (exit {code})")
    if phase:
        emit({"type": "epoch_completed", "phase": phase, "epoch": total_epochs,
              "total_epochs": total_epochs})


def build_list(source: Path, exp: str, out: Path, holdout: int) -> tuple[Path, list[str]]:
    rows = []
    for wav_path in sorted(source.glob("*.wav")):
        lab = wav_path.with_suffix(".lab")
        if not lab.is_file():
            continue
        text = lab.read_text(encoding="utf-8").strip()
        with wave.open(str(wav_path)) as w:
            seconds = w.getnframes() / w.getframerate()
        # Very short barks carry little to learn from; very long lines strain memory.
        if text and 0.8 <= seconds <= 15.0:
            rows.append((wav_path, text))
    random.Random(0).shuffle(rows)
    held, train = rows[:holdout], rows[holdout:]
    list_path = out / f"{exp}.list"
    list_path.write_text(
        "".join(f"{p}|{exp}|zh|{t}\n" for p, t in train), encoding="utf-8")
    (out / "heldout.list").write_text(
        "".join(f"{p}|{exp}|zh|{t}\n" for p, t in held), encoding="utf-8")
    print(f"training lines: {len(train)}, held out: {len(held)}")
    return list_path, [str(p) for p, _ in held]


def merge_part(opt_dir: Path, stem: str, ext: str, header: str | None = None) -> None:
    part = opt_dir / f"{stem}-0.{ext}"
    if part.exists():
        lines = part.read_text(encoding="utf-8").strip("\n").split("\n")
        if header:
            lines = [header, *lines]
        (opt_dir / f"{stem}.{ext}").write_text("\n".join(lines) + "\n", encoding="utf-8")
        part.unlink()


def main() -> int:
    global EVENTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="folder of .wav + .lab pairs")
    parser.add_argument("exp_name")
    parser.add_argument("--holdout", type=int, default=15)
    parser.add_argument("--sovits-epochs", type=int, default=8)
    parser.add_argument("--gpt-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--events", action="store_true", help="print @@CVAI progress events")
    parser.add_argument("--output-dir", type=Path, help="copy the final weights here")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete intermediate features/checkpoints after success")
    args = parser.parse_args()
    EVENTS = args.events

    opt_dir = SRC / "logs" / args.exp_name
    opt_dir.mkdir(parents=True, exist_ok=True)
    emit({"type": "stage", "stage": "preparing", "message": "building the training list"})
    list_path, _ = build_list(args.source.resolve(), args.exp_name, opt_dir, args.holdout)

    ffmpeg = ffmpeg_shared_dir()
    base = {
        "inp_text": str(list_path), "inp_wav_dir": "", "exp_name": args.exp_name,
        "opt_dir": str(opt_dir), "i_part": "0", "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0", "is_half": "True", "version": VERSION,
        "PYTHONIOENCODING": "utf-8",
        # The stage scripts import `text`, `module`, `tools` as top-level packages.
        "PYTHONPATH": os.pathsep.join([str(SRC), str(SRC / "GPT_SoVITS")]),
    }
    if ffmpeg:
        base["PATH"] = ffmpeg + os.pathsep + os.environ["PATH"]

    if not (opt_dir / "2-name2text.txt").exists():
        emit({"type": "stage", "stage": "preparing", "message": "text features"})
        run("1a text + BERT features", "GPT_SoVITS/prepare_datasets/1-get-text.py",
            {**base, "bert_pretrained_dir": str(PRETRAINED / "chinese-roberta-wwm-ext-large")})
        merge_part(opt_dir, "2-name2text", "txt")
    if not any((opt_dir / "5-wav32k").glob("*.wav")):
        emit({"type": "stage", "stage": "preparing", "message": "audio features"})
        run("1b HuBERT features + 32 kHz audio", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            {**base, "cnhubert_base_dir": str(PRETRAINED / "chinese-hubert-base")})
    if not (opt_dir / "7-sv_cn").exists() or not any((opt_dir / "7-sv_cn").iterdir()):
        emit({"type": "stage", "stage": "preparing", "message": "speaker embeddings"})
        run("1b speaker embeddings", "GPT_SoVITS/prepare_datasets/2-get-sv.py",
            {**base, "sv_path": str(PRETRAINED / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt")})
    if not (opt_dir / "6-name2semantic.tsv").exists():
        emit({"type": "stage", "stage": "preparing", "message": "semantic tokens"})
        run("1c semantic tokens", "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            {**base, "pretrained_s2G": str(PRETRAINED / "v2Pro" / "s2Gv2ProPlus.pth"),
             "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json"})
        merge_part(opt_dir, "6-name2semantic", "tsv", header="item_name\tsemantic_audio")

    # ---- SoVITS ----------------------------------------------------------------------
    sovits_dir = SRC / f"SoVITS_weights_{VERSION}"
    gpt_dir = SRC / f"GPT_weights_{VERSION}"
    # The WebUI creates these at startup; the trainers assume they exist and otherwise
    # train every epoch and then fail to write the weights.
    sovits_dir.mkdir(exist_ok=True)
    gpt_dir.mkdir(exist_ok=True)
    emit({"type": "stage", "stage": "training", "phase": "sovits",
          "total_epochs": args.sovits_epochs + args.gpt_epochs})
    if not list(sovits_dir.glob(f"{args.exp_name}_e{args.sovits_epochs}_*.pth")):
        config = json.loads((SRC / f"GPT_SoVITS/configs/s2{VERSION}.json").read_text())
        config["train"].update({
            "batch_size": args.batch_size, "epochs": args.sovits_epochs,
            "text_low_lr_rate": 0.4, "pretrained_s2G": str(PRETRAINED / "v2Pro" / "s2Gv2ProPlus.pth"),
            "pretrained_s2D": str(PRETRAINED / "v2Pro" / "s2Dv2ProPlus.pth"),
            "if_save_latest": True, "if_save_every_weights": True,
            "save_every_epoch": max(1, args.sovits_epochs // 2), "gpu_numbers": "0",
            "grad_ckpt": False, "lora_rank": 32,
        })
        config["model"]["version"] = VERSION
        config["data"]["exp_dir"] = config["s2_ckpt_dir"] = str(opt_dir)
        config["save_weight_dir"] = sovits_dir.name
        config["name"], config["version"] = args.exp_name, VERSION
        (opt_dir / f"logs_s2_{VERSION}").mkdir(exist_ok=True)
        path = opt_dir / "tmp_s2.json"
        path.write_text(json.dumps(config))
        run("2 train SoVITS", "GPT_SoVITS/s2_train.py", base, "--config", str(path),
            phase="sovits", total_epochs=args.sovits_epochs)
    sovits_final = sorted(sovits_dir.glob(f"{args.exp_name}_e{args.sovits_epochs}_*.pth"))
    if not sovits_final:
        emit({"type": "failed", "stage": "2 train SoVITS", "message": "no SoVITS weights were written",
              "data": {"tail": TAIL[-40:]}})
        sys.exit("SoVITS training finished without writing weights")
    emit({"type": "checkpoint_saved", "phase": "sovits", "data": {"file": sovits_final[-1].name}})

    # ---- GPT -------------------------------------------------------------------------
    emit({"type": "stage", "stage": "training", "phase": "gpt"})
    if not list(gpt_dir.glob(f"{args.exp_name}-e{args.gpt_epochs}.ckpt")):
        config = yaml.safe_load((SRC / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text())
        config["train"].update({
            "batch_size": args.batch_size, "epochs": args.gpt_epochs,
            "save_every_n_epoch": max(1, args.gpt_epochs // 3), "if_save_every_weights": True,
            "if_save_latest": True, "if_dpo": False,
            "half_weights_save_dir": gpt_dir.name, "exp_name": args.exp_name,
        })
        config["pretrained_s1"] = str(PRETRAINED / "s1v3.ckpt")
        config["train_semantic_path"] = str(opt_dir / "6-name2semantic.tsv")
        config["train_phoneme_path"] = str(opt_dir / "2-name2text.txt")
        config["output_dir"] = str(opt_dir / f"logs_s1_{VERSION}")
        path = opt_dir / "tmp_s1.yaml"
        path.write_text(yaml.dump(config, default_flow_style=False))
        # FORCE_COLOR keeps Lightning's rich progress bar redrawing per batch; without a
        # terminal it otherwise prints only the final state, and progress jumps to the end.
        run("3 train GPT", "GPT_SoVITS/s1_train.py", {**base, "hz": "25hz", "FORCE_COLOR": "1"},
            "--config_file", str(path), phase="gpt", total_epochs=args.gpt_epochs)
    gpt_final = gpt_dir / f"{args.exp_name}-e{args.gpt_epochs}.ckpt"
    if not gpt_final.is_file():
        emit({"type": "failed", "stage": "3 train GPT", "message": "no GPT weights were written",
              "data": {"tail": TAIL[-40:]}})
        sys.exit("GPT training finished without writing weights")
    emit({"type": "checkpoint_saved", "phase": "gpt", "data": {"file": gpt_final.name}})

    outputs = {"sovits": sovits_final[-1], "gpt": gpt_final}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "sovits": Path(shutil.copy2(sovits_final[-1], args.output_dir / "sovits.pth")),
            "gpt": Path(shutil.copy2(gpt_final, args.output_dir / "gpt.ckpt")),
        }
    if args.cleanup:
        # Features and full G/D checkpoints run to several GB per job.
        shutil.rmtree(opt_dir, ignore_errors=True)
        if args.output_dir:
            for p in list(sovits_dir.glob(f"{args.exp_name}_*.pth")) + list(gpt_dir.glob(f"{args.exp_name}-*.ckpt")):
                p.unlink(missing_ok=True)
    emit({"type": "done", "data": {k: str(v) for k, v in outputs.items()}})
    print("\nweights:")
    for role, path in outputs.items():
        print(f"   {role}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
