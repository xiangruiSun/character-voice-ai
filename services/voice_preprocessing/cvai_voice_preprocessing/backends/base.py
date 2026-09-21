"""Backend interfaces for the model-backed preprocessing stages.

Each stage that needs a model talks to a small interface here, and the concrete
implementation is chosen at run time from what is installed. Three things fall out of
that, all of them things this pipeline needs:

* the pipeline runs with nothing installed (built-in DSP backends do the measurable
  work; model stages report themselves unavailable and are skipped, loudly);
* swapping FunASR for faster-whisper, or UVR for Demucs, is a new class rather than an
  edit to the pipeline;
* every backend reports ``describe()``, which goes into the segment's
  ``processing_chain`` — so a dataset always says which tool and version produced it.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Sequence

from cvai_types import CVAIModel, ProcessingStage, ProcessingStep
from pydantic import Field

from ..dsp import SpeechSegment


class BackendInfo(CVAIModel):
    name: str
    version: str = "unknown"
    available: bool = True
    detail: str = ""
    #: Whether this backend alters the waveform. Recorded so the validator and the
    #: benchmark can tell a cleaned pack from an untouched one (decision D7).
    destructive: bool = False


class Backend(abc.ABC):
    """Common base: a name, an availability probe, and a provenance record."""

    stage: ProcessingStage = ProcessingStage.INGEST
    name: str = "unknown"
    version: str = "unknown"
    destructive: bool = False

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def info(self) -> BackendInfo:
        ok = self.available()
        return BackendInfo(
            name=self.name,
            version=self.version,
            available=ok,
            detail="" if ok else self.unavailable_reason(),
            destructive=self.destructive,
        )

    def step(self, **params: object) -> ProcessingStep:
        return ProcessingStep(
            stage=self.stage,
            tool=self.name,
            tool_version=self.version,
            params=params,
        )


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


class CharTiming(CVAIModel):
    char: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class ASRResult(CVAIModel):
    text: str = ""
    confidence: float | None = None
    #: Character-level timings where the model provides them (paraformer-zh does).
    #: Used for alignment QC and for locating a mispronounced syllable.
    char_timings: list[CharTiming] = Field(default_factory=list)
    #: Sub-utterances with their own timings, when the model segmented internally.
    sentences: list[dict] = Field(default_factory=list)
    model: str = "unknown"


class EmotionResult(CVAIModel):
    label: str = "unknown"
    scores: dict[str, float] = Field(default_factory=dict)
    model: str = "unknown"


# --------------------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------------------


class VADBackend(Backend):
    stage = ProcessingStage.VAD_SEGMENT

    @abc.abstractmethod
    def detect(self, samples: Sequence[float], sample_rate: int) -> list[SpeechSegment]:
        """Speech regions, padded to keep breaths."""


class ASRBackend(Backend):
    stage = ProcessingStage.TRANSCRIBE

    @abc.abstractmethod
    def transcribe(self, path: Path, *, hotwords: Sequence[str] = ()) -> ASRResult:
        """Transcribe one short Mandarin utterance.

        ``hotwords`` carries character and world proper nouns; a backend that cannot use
        them ignores the argument rather than failing.
        """


class SpeakerBackend(Backend):
    stage = ProcessingStage.SPEAKER_FILTER

    @abc.abstractmethod
    def embed(self, path: Path) -> list[float]:
        """Speaker embedding for one clip."""


class EmotionBackend(Backend):
    stage = ProcessingStage.ANNOTATE

    @abc.abstractmethod
    def classify(self, path: Path) -> EmotionResult:
        """Emotion scores for one clip. A first pass; a human decides the final label."""


class SeparationBackend(Backend):
    stage = ProcessingStage.SOURCE_SEPARATION
    destructive = True

    @abc.abstractmethod
    def isolate_vocals(self, source: Path, output_dir: Path) -> Path:
        """Write a vocals-only copy and return its path."""


class DenoiseBackend(Backend):
    stage = ProcessingStage.DENOISE
    destructive = True

    @abc.abstractmethod
    def denoise(self, source: Path, destination: Path) -> Path:
        """Write a denoised copy and return its path."""


class UnavailableBackend(Backend):
    """Stands in for a backend whose dependency is not installed.

    A placeholder that *reports* its absence, rather than a silent no-op: a stage that
    quietly did nothing is indistinguishable in the output from a stage that ran and
    found nothing to do, and those need very different responses.
    """

    def __init__(self, name: str, reason: str, stage: ProcessingStage) -> None:
        self.name = name
        self._reason = reason
        self.stage = stage

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return self._reason
