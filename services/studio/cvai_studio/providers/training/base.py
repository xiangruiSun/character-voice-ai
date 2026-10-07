"""The boundary every voice training engine implements.

The Studio's training service only ever talks to ``VoiceTrainingProvider``. An engine
reports progress as ``TrainingEvent``s — never as console text for someone else to
parse — and may stream them while ``train`` runs.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingEvent:
    type: str                      # stage | epoch_completed | metric | checkpoint_saved | log
    stage: str | None = None       # validating | preparing | training | evaluating
    phase: str | None = None       # engine-specific sub-phase, e.g. "sovits" / "gpt"
    epoch: int | None = None
    total_epochs: int | None = None
    progress: float | None = None  # 0..1 within the whole training stage
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PresetOption:
    key: str
    label: str
    description: str
    params: dict[str, Any]


@dataclass
class AdvancedOption:
    """One setting the engine really supports, with its bounds."""
    key: str
    label: str
    kind: str                      # "int" | "float"
    default: float
    minimum: float
    maximum: float
    hint: str = ""


@dataclass
class TrainingSample:
    audio_path: Path
    transcript: str
    style: str = "neutral"


@dataclass
class TrainingResult:
    #: Engine-specific checkpoint files, by role (e.g. {"gpt": path, "sovits": path}).
    checkpoint_files: dict[str, Path]
    base_model: str
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingCancelled(Exception):
    pass


class TrainingFailure(Exception):
    """``message`` and ``hint`` are for the user; ``detail`` for Advanced Details."""

    def __init__(self, message: str, *, hint: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail


EmitFn = Callable[[TrainingEvent], None]
CancelledFn = Callable[[], bool]


class VoiceTrainingProvider(abc.ABC):
    engine: str = "unknown"
    label: str = "unknown"
    base_model: str = "unknown"

    @abc.abstractmethod
    def presets(self) -> list[PresetOption]: ...

    @abc.abstractmethod
    def advanced_options(self) -> list[AdvancedOption]: ...

    def resolve_config(self, preset: str, overrides: dict[str, Any]) -> dict[str, Any]:
        presets = {p.key: p for p in self.presets()}
        if preset not in presets:
            raise ValueError(f"unknown preset {preset!r}")
        config = dict(presets[preset].params)
        allowed = {o.key: o for o in self.advanced_options()}
        for key, value in (overrides or {}).items():
            option = allowed.get(key)
            if option is None or value is None:
                continue
            number = int(value) if option.kind == "int" else float(value)
            config[key] = min(max(number, option.minimum), option.maximum)
        return config

    def validate_dataset(self, samples: list[TrainingSample]) -> list[str]:
        """Problems that make training pointless. Empty means go."""
        problems = []
        if len(samples) < 20:
            problems.append(f"只有 {len(samples)} 个可用片段，至少需要 20 个")
        if sum(1 for s in samples if not s.transcript.strip()):
            problems.append("有片段没有文字")
        return problems

    @abc.abstractmethod
    def prepare_dataset(self, samples: list[TrainingSample], workdir: Path) -> Path: ...

    @abc.abstractmethod
    def train(self, dataset: Path, workdir: Path, config: dict[str, Any], *,
              job_id: str, emit: EmitFn, cancelled: CancelledFn) -> TrainingResult: ...

    @abc.abstractmethod
    def synthesize(self, text: str, *, checkpoint: dict[str, Path], reference_audio: Path,
                   reference_text: str, out_path: Path) -> Path: ...

    def evaluate_checkpoint(self, checkpoint: dict[str, Path], reference_audio: Path,
                            reference_text: str, out_dir: Path) -> dict[str, Any]:
        """Render a few fixed sentences so the result can be heard, not just trusted."""
        sentences = ["你好，很高兴见到你。", "今天的风有点大，记得多穿一件外套。", "嗯……让我想一想再回答你，好吗？"]
        outputs = []
        for index, sentence in enumerate(sentences):
            path = self.synthesize(sentence, checkpoint=checkpoint, reference_audio=reference_audio,
                                   reference_text=reference_text, out_path=out_dir / f"eval_{index}.wav")
            outputs.append({"text": sentence, "path": path})
        return {"samples": outputs}
