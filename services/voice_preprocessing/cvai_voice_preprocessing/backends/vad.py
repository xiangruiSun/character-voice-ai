"""Silero VAD backend.

Preferred over the built-in energy gate whenever it is installed: it is a trained model,
it handles noisy and reverberant dialogue far better, and it is MIT-licensed with no
telemetry or registration. The energy gate remains the fallback so the pipeline still
segments on a bare install.

One project-specific choice: ``speech_pad_ms`` defaults to 180 rather than Silero's own
30. Tight padding clips the breath before a line, and breaths are listed in spec §7 as
something to preserve. Padding is easy to reduce in review; a lost breath is not
recoverable without re-segmenting.
"""

from __future__ import annotations

from typing import Any, Sequence

from cvai_core.dsp import SpeechSegment
from .base import VADBackend

_HINT = "silero-vad is not installed; `pip install silero-vad` (MIT) for better segmentation"

#: Silero accepts 8 kHz and 16 kHz only.
SILERO_RATES = (8000, 16000)


class SileroVADBackend(VADBackend):
    name = "silero-vad"
    version = "5"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 200,
        max_speech_duration_s: float = 20.0,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 180,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model: Any = None

    def available(self) -> bool:
        try:
            import silero_vad  # noqa: F401,PLC0415
            import torch  # noqa: F401,PLC0415

            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return _HINT

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from silero_vad import load_silero_vad  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(_HINT) from exc
            self._model = load_silero_vad()
        return self._model

    def detect(self, samples: Sequence[float], sample_rate: int) -> list[SpeechSegment]:
        import torch  # noqa: PLC0415
        from silero_vad import get_speech_timestamps  # noqa: PLC0415

        model = self._ensure()
        target_rate = 16000
        work, rate = _resample_for_vad(samples, sample_rate, target_rate)
        tensor = torch.tensor(work, dtype=torch.float32)

        stamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )
        duration = len(samples) / sample_rate
        segments: list[SpeechSegment] = []
        for stamp in stamps:
            start = max(0.0, float(stamp["start"]))
            end = min(duration, float(stamp["end"]))
            if end > start:
                segments.append(SpeechSegment(round(start, 4), round(end, 4)))
        return segments


def _resample_for_vad(
    samples: Sequence[float], sample_rate: int, target_rate: int
) -> tuple[list[float], int]:
    """Cheap linear resample to a rate Silero accepts.

    Quality is irrelevant here — the output is used to find boundaries, and the
    boundaries are then applied to the original full-rate audio.
    """
    if sample_rate in SILERO_RATES:
        return list(samples), sample_rate
    if not samples:
        return [], target_rate

    ratio = target_rate / sample_rate
    count = max(1, int(len(samples) * ratio))
    out: list[float] = []
    for index in range(count):
        position = index / ratio
        left = int(position)
        right = min(len(samples) - 1, left + 1)
        fraction = position - left
        out.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return out, target_rate
