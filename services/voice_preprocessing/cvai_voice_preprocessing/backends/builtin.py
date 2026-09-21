"""Dependency-free backends built on the DSP module.

These are not stand-ins. Energy-gated VAD works well on game dialogue, which is usually
either clean or obviously not; the spectral-feature speaker embedding is weak next to
CAM++ but is enough to separate two clearly different voices in one file, and it
degrades honestly (it reports a low-confidence flag rather than pretending).

The genuinely model-shaped problems — transcription and emotion classification — have no
credible dependency-free implementation. Rather than inventing one, this module supplies
backends that say so, plus a clearly-labelled deterministic ASR stub that exists only so
the pipeline can be exercised offline and in tests. Anything it produces is marked
``transcript_source="stub"``, and the dataset builder refuses to ship it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from cvai_types import ProcessingStage

from .. import dsp
from ..audio_io import read_samples
from .base import ASRBackend, ASRResult, EmotionBackend, EmotionResult, SpeakerBackend, VADBackend

STUB_TRANSCRIPT_SOURCE = "stub"


class EnergyVADBackend(VADBackend):
    """Adaptive energy gate. See :func:`cvai_voice_preprocessing.dsp.energy_vad`."""

    name = "builtin-energy-vad"
    version = "1"

    def __init__(
        self,
        *,
        min_speech_ms: float = 200.0,
        min_silence_ms: float = 300.0,
        speech_pad_ms: float = 180.0,
        margin_db: float = 12.0,
    ) -> None:
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.margin_db = margin_db

    def detect(self, samples: Sequence[float], sample_rate: int) -> list[dsp.SpeechSegment]:
        return dsp.energy_vad(
            samples,
            sample_rate,
            min_speech_ms=self.min_speech_ms,
            min_silence_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            margin_db=self.margin_db,
        )


class SpectralSpeakerBackend(SpeakerBackend):
    """A cheap spectral-shape embedding, for when no speaker model is installed.

    Log-energy in a bank of bands plus pitch statistics, L2-normalized. It will separate
    a low male voice from a high female one inside the same file, which is the common
    game-dialogue case. It will *not* reliably separate two similar voices, so the
    pipeline records ``speaker_backend`` alongside the score and the validator treats a
    built-in score as advisory rather than as a filter.
    """

    name = "builtin-spectral-speaker"
    version = "1"

    #: Band edges in Hz. Coarse on purpose: finer bands would mostly encode phonetic
    #: content, which is exactly what a speaker embedding should be invariant to.
    BANDS = (80, 160, 260, 400, 600, 900, 1400, 2200, 3400, 5200, 8000)

    def embed(self, path: Path) -> list[float]:
        samples, rate = read_samples(Path(path))
        if not samples:
            return [0.0] * (len(self.BANDS) + 2)

        energies = self._band_energies(samples, rate)
        track = dsp.estimate_f0_track(samples, rate)
        mean, spread = dsp.f0_statistics(track)
        vector = energies + [
            math.log1p(mean or 0.0) / 7.0,
            math.log1p(spread or 0.0) / 7.0,
        ]
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def _band_energies(self, samples: Sequence[float], rate: int) -> list[float]:
        """Goertzel-style band energies — no FFT dependency needed."""
        length = min(len(samples), rate * 3)  # first 3 s is plenty for timbre
        window = samples[:length]
        if not window:
            return [0.0] * len(self.BANDS)

        energies: list[float] = []
        for index in range(len(self.BANDS)):
            low = self.BANDS[index]
            high = self.BANDS[index + 1] if index + 1 < len(self.BANDS) else rate // 2
            centre = math.sqrt(max(1.0, low) * max(low + 1.0, high))
            if centre >= rate / 2:
                energies.append(0.0)
                continue
            energies.append(self._goertzel_db(window, rate, centre))
        peak = max(energies) if energies else 0.0
        return [e - peak for e in energies]

    @staticmethod
    def _goertzel_db(samples: Sequence[float], rate: int, frequency: float) -> float:
        coefficient = 2.0 * math.cos(2.0 * math.pi * frequency / rate)
        s_prev = s_prev2 = 0.0
        for value in samples:
            s = value + coefficient * s_prev - s_prev2
            s_prev2, s_prev = s_prev, s
        power = s_prev2 * s_prev2 + s_prev * s_prev - coefficient * s_prev * s_prev2
        return dsp.to_dbfs(math.sqrt(max(0.0, power)) / max(1, len(samples)))


class StubASRBackend(ASRBackend):
    """Deterministic placeholder transcription. **Never ships a dataset.**

    Exists so the whole pipeline — segmentation, annotation, review page, dataset build
    — can be exercised without downloading an ASR model. Output is marked
    ``transcript_source="stub"``; :func:`build_dataset` refuses to emit a dataset
    containing stub transcripts unless explicitly forced, and the review page shows them
    highlighted as needing a human.
    """

    name = STUB_TRANSCRIPT_SOURCE
    version = "1"

    #: Short neutral Mandarin phrases. Picked deterministically from the audio hash, so
    #: a test run is reproducible and a human reviewing the page sees obviously wrong
    #: text rather than something plausible they might accept by accident.
    _PHRASES = (
        "（待人工转写）",
        "（占位文本，需要人工校对）",
        "（ASR 未运行）",
    )

    def transcribe(self, path: Path, *, hotwords: Sequence[str] = ()) -> ASRResult:
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        phrase = self._PHRASES[digest[0] % len(self._PHRASES)]
        return ASRResult(text=phrase, confidence=0.0, model=self.name)


class NullEmotionBackend(EmotionBackend):
    """Reports no emotion rather than guessing one."""

    name = "none"
    version = "1"

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return (
            "no emotion model installed; install the 'preprocess' extra for "
            "emotion2vec, or label styles by hand in the review page"
        )

    def classify(self, path: Path) -> EmotionResult:
        return EmotionResult(label="unknown", model=self.name)


def unavailable(name: str, reason: str, stage: ProcessingStage):
    from .base import UnavailableBackend

    return UnavailableBackend(name, reason, stage)
