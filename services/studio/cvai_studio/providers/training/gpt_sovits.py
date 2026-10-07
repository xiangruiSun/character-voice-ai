"""GPT-SoVITS v2ProPlus fine-tuning — the first (and most proven) training engine.

Training runs ``scripts/train_gpt_sovits.py`` with the GPT-SoVITS environment's own
Python (it needs its own torch build), reading the script's ``@@CVAI`` event lines.
Synthesis goes through the running GPT-SoVITS API server (``api_v2.py``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from cvai_core.paths import repo_root

from .base import (
    AdvancedOption,
    CancelledFn,
    EmitFn,
    PresetOption,
    TrainingCancelled,
    TrainingEvent,
    TrainingFailure,
    TrainingResult,
    TrainingSample,
    VoiceTrainingProvider,
)

_FAILURES = [
    # (substring in the log tail, message, hint)
    ("out of memory", "训练时显存不足", "请关闭其他占用显卡的程序，或在高级设置中减小批大小"),
    ("cuda error", "显卡运行出错", "请确认显卡驱动正常，并关闭其他占用显卡的程序后重试"),
    ("no module named", "训练环境缺少依赖", "请检查 GPT-SoVITS 环境是否完整安装"),
    ("filenotfounderror", "训练所需的文件缺失", "请确认 GPT-SoVITS 预训练模型已下载"),
    ("access violation", "训练进程意外崩溃", "这通常与 PyTorch 版本有关；GPT-SoVITS 环境应使用 PyTorch 2.7.1"),
]


def _explain(tail: list[str]) -> tuple[str, str]:
    text = "\n".join(tail).lower()
    for needle, message, hint in _FAILURES:
        if needle in text:
            return message, hint
    return "训练意外中止", "请展开高级详情查看日志；调整设置后可以重试"


class GPTSoVITSTrainingProvider(VoiceTrainingProvider):
    engine = "gpt_sovits"
    label = "GPT-SoVITS"
    base_model = "v2ProPlus"

    def __init__(self, engine_dir: Path, api_url: str) -> None:
        self.engine_dir = Path(engine_dir)
        self.api_url = api_url.rstrip("/")
        self.python = self.engine_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.script = repo_root() / "scripts" / "train_gpt_sovits.py"

    # -- configuration ----------------------------------------------------------------

    def presets(self) -> list[PresetOption]:
        return [
            PresetOption("quick", "快速试听", "几分钟出结果，适合先确认数据可用",
                         {"sovits_epochs": 4, "gpt_epochs": 8, "batch_size": 8}),
            PresetOption("balanced", "均衡", "推荐：音色和稳定性的平衡",
                         {"sovits_epochs": 8, "gpt_epochs": 15, "batch_size": 8}),
            PresetOption("high", "高保真", "训练更久，数据充足时效果更好",
                         {"sovits_epochs": 12, "gpt_epochs": 25, "batch_size": 8}),
        ]

    def advanced_options(self) -> list[AdvancedOption]:
        return [
            AdvancedOption("sovits_epochs", "音色训练轮数（SoVITS）", "int", 8, 1, 30,
                           "决定音色还原度；过多容易过拟合"),
            AdvancedOption("gpt_epochs", "语调训练轮数（GPT）", "int", 15, 1, 50,
                           "决定语气和节奏；过多会让语调僵硬"),
            AdvancedOption("batch_size", "批大小", "int", 8, 1, 32, "显存不足时调小"),
        ]

    def available(self) -> tuple[bool, str]:
        if not self.python.is_file():
            return False, f"找不到 GPT-SoVITS 环境：{self.python}"
        if not self.script.is_file():
            return False, "找不到训练脚本 scripts/train_gpt_sovits.py"
        return True, ""

    # -- dataset ------------------------------------------------------------------------

    def prepare_dataset(self, samples: list[TrainingSample], workdir: Path) -> Path:
        """Write ``NNNN.wav`` + ``NNNN.lab`` pairs at 32 kHz mono — the script's input."""
        from cvai_voice_preprocessing.audio_io import decode_to_wav

        dataset = workdir / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            target = dataset / f"{index:05d}.wav"
            decode_to_wav(sample.audio_path, target, sample_rate=32000, mono=True)
            (dataset / f"{index:05d}.lab").write_text(sample.transcript.strip(), encoding="utf-8")
        return dataset

    # -- training -----------------------------------------------------------------------

    def train(self, dataset: Path, workdir: Path, config: dict[str, Any], *,
              job_id: str, emit: EmitFn, cancelled: CancelledFn) -> TrainingResult:
        ok, reason = self.available()
        if not ok:
            raise TrainingFailure("训练引擎不可用", hint=reason)
        output = workdir / "checkpoints"
        log_path = workdir / "train.log"
        sovits_epochs = int(config["sovits_epochs"])
        gpt_epochs = int(config["gpt_epochs"])
        total = sovits_epochs + gpt_epochs
        command = [
            str(self.python), "-u", str(self.script), str(dataset), job_id,
            "--sovits-epochs", str(sovits_epochs), "--gpt-epochs", str(gpt_epochs),
            "--batch-size", str(int(config["batch_size"])), "--holdout", "0",
            "--events", "--output-dir", str(output), "--cleanup",
        ]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, creationflags=flags, start_new_session=os.name != "nt")
        tail: list[str] = []
        state: dict[str, Any] = {"done": None, "failed": None}

        def pump() -> None:
            assert process.stdout is not None
            with open(log_path, "a", encoding="utf-8") as log:
                for raw in process.stdout:
                    line = raw.decode("utf-8", "replace").rstrip()
                    if not line.startswith("@@CVAI "):
                        log.write(line + "\n")
                        tail.append(line)
                        del tail[:-80]
                        continue
                    event = json.loads(line[7:])
                    kind = event.get("type")
                    if kind == "epoch_completed":
                        done = event["epoch"] + (sovits_epochs if event.get("phase") == "gpt" else 0)
                        emit(TrainingEvent(type="epoch_completed", stage="training",
                                           phase=event.get("phase"), epoch=done, total_epochs=total,
                                           progress=min(1.0, done / total)))
                    elif kind == "stage":
                        emit(TrainingEvent(type="stage", stage=event.get("stage"), phase=event.get("phase"),
                                           total_epochs=total, message=event.get("message")))
                    elif kind == "metric":
                        emit(TrainingEvent(type="metric", phase=event.get("phase"), data=event.get("data", {})))
                    elif kind == "checkpoint_saved":
                        emit(TrainingEvent(type="checkpoint_saved", phase=event.get("phase"), data=event.get("data", {})))
                    elif kind == "failed":
                        state["failed"] = event
                    elif kind == "done":
                        state["done"] = event

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while process.poll() is None:
            if cancelled():
                self._kill(process)
                reader.join(timeout=10)
                raise TrainingCancelled()
            time.sleep(1.0)
        reader.join(timeout=30)

        if process.returncode != 0 or not state["done"]:
            failed_tail = (state["failed"] or {}).get("data", {}).get("tail") or tail[-40:]
            message, hint = _explain(failed_tail)
            raise TrainingFailure(message, hint=hint, detail="\n".join(failed_tail[-60:]))
        files = {role: Path(path) for role, path in state["done"]["data"].items()}
        return TrainingResult(checkpoint_files=files, base_model=self.base_model,
                              metrics={"sovits_epochs": sovits_epochs, "gpt_epochs": gpt_epochs})

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        if os.name == "nt":
            # The trainer spawns its own children (data loaders, DDP); end the tree.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:  # pragma: no cover - POSIX
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()

    # -- synthesis ----------------------------------------------------------------------

    def synthesize(self, text: str, *, checkpoint: dict[str, Path], reference_audio: Path,
                   reference_text: str, out_path: Path) -> Path:
        from cvai_tts_providers.gpt_sovits import _SERVER_CHECKPOINT

        checkpoint_id = f"{checkpoint['gpt']}|{checkpoint['sovits']}"
        try:
            with httpx.Client(base_url=self.api_url, timeout=300) as client:
                if _SERVER_CHECKPOINT.get(self.api_url) != checkpoint_id:
                    client.get("/set_gpt_weights", params={"weights_path": str(checkpoint["gpt"])}).raise_for_status()
                    client.get("/set_sovits_weights", params={"weights_path": str(checkpoint["sovits"])}).raise_for_status()
                    _SERVER_CHECKPOINT[self.api_url] = checkpoint_id
                response = client.post("/tts", json={
                    "text": text, "text_lang": "zh", "ref_audio_path": str(reference_audio),
                    "prompt_text": reference_text, "prompt_lang": "zh", "text_split_method": "cut5",
                    "top_k": 15, "top_p": 1.0, "temperature": 1.0, "repetition_penalty": 1.35,
                    "media_type": "wav",
                })
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise TrainingFailure("语音引擎未启动", hint="请先运行 scripts/start_voice_server.ps1") from exc
        except httpx.HTTPStatusError as exc:
            raise TrainingFailure("语音生成失败", detail=exc.response.text[:500]) from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)
        return out_path


def default_provider(settings) -> GPTSoVITSTrainingProvider:
    return GPTSoVITSTrainingProvider(settings.gpt_sovits_dir, settings.gpt_sovits_url)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("import this module; it is not a script")
