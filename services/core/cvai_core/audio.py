"""Minimal audio I/O built on the standard library.

Deliberately dependency-free. Milestone 1 must run with only pydantic and PyYAML
installed so that the architecture, the benchmark harness and the tests can be exercised
on any machine — including a CPU-only box with no torch and no model weights. The heavy
audio stack (soundfile, librosa, torchaudio) arrives with the Milestone 2 ``preprocess``
extra and is used only by the preprocessing pipeline.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Final, Iterable, Sequence

from cvai_types import AudioProperties

#: Average Mandarin speaking rate in characters per second, used to estimate how long a
#: line *should* take. The benchmark compares generated duration against this and against
#: the character's measured rate; a model that is 40% too fast is one of the easier
#: character-fidelity failures to catch automatically.
DEFAULT_CHARS_PER_SECOND: Final = 4.8

_INT16_MAX: Final = 32767


def write_wav(
    path: Path,
    samples: Sequence[float] | Iterable[float],
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Write float samples in [-1, 1] as 16-bit PCM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for value in samples:
        clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
        frames += struct.pack("<h", int(round(clamped * _INT16_MAX)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def read_wav_properties(path: Path) -> AudioProperties:
    """Probe a WAV file without loading the whole thing into memory."""
    with wave.open(str(Path(path)), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
    if rate <= 0:
        raise ValueError(f"{path} reports a sample rate of {rate}")
    duration = frames / float(rate)
    if duration <= 0:
        raise ValueError(f"{path} contains no audio frames")
    return AudioProperties(
        sample_rate=rate,
        channels=channels,
        duration_s=round(duration, 6),
    )


def read_wav_samples(path: Path) -> tuple[list[float], int]:
    """Read a 16-bit mono/stereo WAV into floats. Only for small files (references)."""
    with wave.open(str(Path(path)), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported by this reader")
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    count = len(raw) // 2
    values = struct.unpack(f"<{count}h", raw)
    floats = [v / _INT16_MAX for v in values]
    if channels > 1:
        # Downmix by averaging, which is right for dialogue and wrong for nothing we care
        # about at this layer.
        floats = [
            sum(floats[i : i + channels]) / channels
            for i in range(0, len(floats) - channels + 1, channels)
        ]
    return floats, rate


def estimate_speech_duration(
    text: str,
    *,
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
    speed_factor: float = 1.0,
    pause_bonus_s: float = 0.18,
) -> float:
    """Estimate how long a Chinese line should take to speak.

    Counts CJK characters and Latin words; adds a small pause for each piece of
    sentence-internal punctuation, because pause structure is a large part of why a
    character's pacing reads as natural (spec §5).
    """
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin_words = len([w for w in _latin_runs(text) if w.strip()])
    pauses = sum(1 for ch in text if ch in "，。！？；：、…,.!?;:")
    units = cjk + latin_words * 2.2
    if units <= 0:
        units = max(1.0, len(text) * 0.6)
    base = units / max(0.5, chars_per_second)
    return max(0.25, (base + pauses * pause_bonus_s) / max(0.1, speed_factor))


def _latin_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            current.append(ch)
        else:
            if current:
                runs.append("".join(current))
                current = []
    if current:
        runs.append("".join(current))
    return runs


def rms_dbfs(samples: Sequence[float]) -> float:
    """RMS level in dBFS. Used by the mock engine and by simple sanity checks."""
    if not samples:
        return -120.0
    mean_square = sum(s * s for s in samples) / len(samples)
    if mean_square <= 0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(math.sqrt(mean_square)))
