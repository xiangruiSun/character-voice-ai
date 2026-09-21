"""Signal measurements the pipeline needs, implemented without a model.

These are real analyses, not placeholders: an energy-gated VAD, autocorrelation pitch
tracking, SNR estimation, clipping detection and loudness. They are here rather than
behind an optional dependency because they are small, and because being able to measure
a voice pack on any machine — no torch, no CUDA, no downloads — is what makes the
pipeline's numbers reviewable by whoever is looking at the data.

The model-backed stages (ASR, source separation, speaker identity, emotion) do live
behind optional dependencies; see ``backends/``.

``numpy`` is used when present and the pure-Python path is kept correct, because the
fallback is what runs in CI and in a fresh checkout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:  # numpy is in the `preprocess` extra, not the base install
    import numpy as _np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - depends on install extras
    _np = None  # type: ignore[assignment]
    HAVE_NUMPY = False

#: Mandarin speaking F0 search range. Generous at both ends: a whispered or shouted
#: character line sits well outside a textbook range, and clamping too tightly turns a
#: real measurement into a wrong one.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0

_SILENCE_DB = -120.0


# --------------------------------------------------------------------------------------
# Level measurements
# --------------------------------------------------------------------------------------


def rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    if HAVE_NUMPY:
        array = _np.asarray(samples, dtype=_np.float64)
        return float(_np.sqrt(_np.mean(array * array)))
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def to_dbfs(value: float) -> float:
    return _SILENCE_DB if value <= 1e-10 else max(_SILENCE_DB, 20.0 * math.log10(value))


def peak_dbfs(samples: Sequence[float]) -> float:
    if not samples:
        return _SILENCE_DB
    peak = max(abs(float(s)) for s in samples)
    return to_dbfs(peak)


def clipping_ratio(samples: Sequence[float], threshold: float = 0.997) -> float:
    """Fraction of samples at or beyond full scale.

    Clipped source audio teaches a model to reproduce distortion as part of the voice,
    so this gates approval rather than merely informing it.
    """
    if not samples:
        return 0.0
    if HAVE_NUMPY:
        array = _np.abs(_np.asarray(samples, dtype=_np.float32))
        return float((array >= threshold).mean())
    hits = sum(1 for s in samples if abs(s) >= threshold)
    return hits / len(samples)


def approximate_lufs(samples: Sequence[float], sample_rate: int) -> float:
    """Loudness estimate.

    Uses ``pyloudnorm`` (ITU-R BS.1770-4) when installed. Otherwise returns an RMS-based
    approximation offset to sit roughly where BS.1770 lands for speech — good enough to
    order clips and spot outliers, **not** good enough to publish as LUFS. Callers record
    which one produced the number.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import pyloudnorm  # noqa: PLC0415
    except ImportError:
        # RMS dBFS for speech runs a few dB above BS.1770 integrated loudness; the
        # constant is a convention for comparability within one pack, nothing more.
        return round(to_dbfs(rms(samples)) - 3.0, 3)

    if len(samples) < sample_rate // 2:
        # BS.1770 needs at least 400 ms for one gating block.
        return round(to_dbfs(rms(samples)) - 3.0, 3)
    meter = pyloudnorm.Meter(sample_rate)
    data = np.asarray(samples, dtype=np.float64)
    return round(float(meter.integrated_loudness(data)), 3)


def loudness_backend_name() -> str:
    try:
        import pyloudnorm  # noqa: F401,PLC0415

        return "pyloudnorm(BS.1770-4)"
    except ImportError:
        return "rms-approximation"


def apply_gain_db(samples: Sequence[float], gain_db: float) -> list[float]:
    factor = 10.0 ** (gain_db / 20.0)
    if HAVE_NUMPY:
        array = _np.asarray(samples, dtype=_np.float32) * factor
        return [float(v) for v in array]
    return [s * factor for s in samples]


def normalize_loudness(
    samples: Sequence[float],
    sample_rate: int,
    target_lufs: float,
    *,
    true_peak_ceiling_dbfs: float = -1.0,
    max_gain_db: float = 18.0,
) -> tuple[list[float], float, float]:
    """Gain-only loudness normalization with a peak guard.

    Gain only, deliberately: no compression, no limiting, no EQ. Spec §5 counts volume
    dynamics as part of the character, and a compressor is the fastest way to flatten
    them into something that sounds like every other AI voice.

    Returns ``(samples, applied_gain_db, measured_lufs_before)``.
    """
    measured = approximate_lufs(samples, sample_rate)
    gain = target_lufs - measured
    gain = max(-max_gain_db, min(max_gain_db, gain))

    current_peak = peak_dbfs(samples)
    headroom = true_peak_ceiling_dbfs - current_peak
    if gain > headroom:
        # Prefer staying quieter than the target over clipping the performance.
        gain = headroom
    if abs(gain) < 0.05:
        return list(samples), 0.0, measured
    return apply_gain_db(samples, gain), round(gain, 3), measured


# --------------------------------------------------------------------------------------
# Framing and voice activity
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def frame_energies_db(
    samples: Sequence[float],
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[list[float], float]:
    """Per-frame RMS in dBFS, plus the hop in seconds."""
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if not samples:
        return [], hop / sample_rate

    if HAVE_NUMPY:
        array = _np.asarray(samples, dtype=_np.float32)
        count = max(1, 1 + (len(array) - frame) // hop) if len(array) >= frame else 1
        energies: list[float] = []
        for index in range(count):
            window = array[index * hop : index * hop + frame]
            if window.size == 0:
                energies.append(_SILENCE_DB)
                continue
            value = float(_np.sqrt(_np.mean(window.astype(_np.float64) ** 2)))
            energies.append(to_dbfs(value))
        return energies, hop / sample_rate

    energies = []
    index = 0
    while index < len(samples):
        window = samples[index : index + frame]
        energies.append(to_dbfs(rms(window)))
        index += hop
    return energies, hop / sample_rate


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return _SILENCE_DB
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
    return ordered[position]


def energy_vad(
    samples: Sequence[float],
    sample_rate: int,
    *,
    min_speech_ms: float = 200.0,
    min_silence_ms: float = 300.0,
    speech_pad_ms: float = 180.0,
    margin_db: float = 12.0,
    floor_below_peak_db: float = 38.0,
) -> list[SpeechSegment]:
    """Adaptive energy-gated voice activity detection.

    The threshold is set from the clip's own noise floor (10th-percentile frame energy)
    plus a margin, bounded so it can never sit more than ``floor_below_peak_db`` under
    the peak. That adaptation matters for game dialogue, where one file is studio-clean
    and the next was bussed through a room reverb.

    ``speech_pad_ms`` defaults high on purpose. Trimming tight to detected speech cuts
    the intake of breath before a line and the decay after it, and those are part of
    what makes a character sound alive (spec §7). It is much cheaper to trim further in
    review than to discover the breaths are gone after training.
    """
    energies, hop_s = frame_energies_db(samples, sample_rate)
    if not energies:
        return []

    noise_floor = _percentile(energies, 0.10)
    peak = max(energies)
    threshold = max(noise_floor + margin_db, peak - floor_below_peak_db)

    voiced = [energy >= threshold for energy in energies]
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(voiced)))

    if not segments:
        return []

    # Merge across short silences, then drop anything too brief to be a line.
    min_silence_frames = max(1, int(min_silence_ms / 1000.0 / hop_s))
    merged: list[list[int]] = [list(segments[0])]
    for begin, end in segments[1:]:
        if begin - merged[-1][1] <= min_silence_frames:
            merged[-1][1] = end
        else:
            merged.append([begin, end])

    duration_s = len(samples) / sample_rate
    pad_s = speech_pad_ms / 1000.0
    min_speech_s = min_speech_ms / 1000.0

    result: list[SpeechSegment] = []
    for begin, end in merged:
        start_s = max(0.0, begin * hop_s - pad_s)
        end_s = min(duration_s, end * hop_s + pad_s)
        if end_s - start_s >= min_speech_s:
            result.append(SpeechSegment(round(start_s, 4), round(end_s, 4)))
    return result


def estimate_snr_db(samples: Sequence[float], sample_rate: int) -> float | None:
    """Speech-to-background ratio from the VAD's own frame partition.

    Crude — it compares speech-frame energy with non-speech-frame energy — but it is the
    number that separates "recorded in a booth" from "recorded over battle music", which
    is the distinction that matters when deciding whether source separation is worth its
    damage.
    """
    energies, hop_s = frame_energies_db(samples, sample_rate)
    if len(energies) < 6:
        return None
    noise_floor = _percentile(energies, 0.10)
    peak = max(energies)
    threshold = max(noise_floor + 12.0, peak - 38.0)

    speech = [e for e in energies if e >= threshold]
    silence = [e for e in energies if e < threshold]
    if not speech or not silence:
        return None
    return round(sum(speech) / len(speech) - sum(silence) / len(silence), 2)


# --------------------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------------------


def _downsample(samples: Sequence[float], sample_rate: int, target_rate: int) -> tuple[list[float], int]:
    """Averaging decimation. Adequate for pitch, which only needs the low band."""
    if sample_rate <= target_rate:
        return list(samples), sample_rate
    factor = int(sample_rate / target_rate)
    if factor < 2:
        return list(samples), sample_rate
    if HAVE_NUMPY:
        array = _np.asarray(samples, dtype=_np.float32)
        usable = (len(array) // factor) * factor
        if usable == 0:
            return list(samples), sample_rate
        reduced = array[:usable].reshape(-1, factor).mean(axis=1)
        return [float(v) for v in reduced], sample_rate // factor
    reduced = [
        sum(samples[i : i + factor]) / factor
        for i in range(0, len(samples) - factor + 1, factor)
    ]
    return reduced, sample_rate // factor


def estimate_f0_track(
    samples: Sequence[float],
    sample_rate: int,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    voicing_threshold: float = 0.35,
) -> list[float]:
    """Per-frame F0 in Hz via normalized autocorrelation; unvoiced frames are dropped.

    Used for the character's pitch distribution, which the benchmark compares generated
    speech against. A model that nails timbre but sits a third above the character's
    real register is caught here, numerically, before anyone has to describe it.
    """
    work, rate = _downsample(samples, sample_rate, 8000)
    if len(work) < int(0.05 * rate):
        return []

    frame = int(rate * frame_ms / 1000.0)
    hop = max(1, int(rate * hop_ms / 1000.0))
    min_lag = max(2, int(rate / F0_MAX_HZ))
    max_lag = min(frame - 1, int(rate / F0_MIN_HZ))
    if max_lag <= min_lag:
        return []

    track: list[float] = []
    if HAVE_NUMPY:
        array = _np.asarray(work, dtype=_np.float64)
        for start in range(0, max(1, len(array) - frame), hop):
            window = array[start : start + frame]
            if window.size < frame:
                break
            window = window - window.mean()
            energy = float(_np.dot(window, window))
            if energy <= 1e-8:
                continue
            correlation = _np.correlate(window, window, mode="full")[frame - 1 :]
            candidates = correlation[min_lag : max_lag + 1]
            if candidates.size == 0:
                continue
            best = int(_np.argmax(candidates)) + min_lag
            score = float(correlation[best] / energy)
            if score >= voicing_threshold:
                track.append(rate / best)
        return track

    # Pure-python fallback: analyse every fourth frame to keep this tractable.
    index = 0
    stride = hop * 4
    while index + frame < len(work):
        window = work[index : index + frame]
        mean = sum(window) / frame
        centred = [v - mean for v in window]
        energy = sum(v * v for v in centred)
        if energy > 1e-8:
            best_lag, best_score = 0, 0.0
            for lag in range(min_lag, max_lag + 1):
                total = 0.0
                for i in range(frame - lag):
                    total += centred[i] * centred[i + lag]
                score = total / energy
                if score > best_score:
                    best_score, best_lag = score, lag
            if best_lag and best_score >= voicing_threshold:
                track.append(rate / best_lag)
        index += stride
    return track


def f0_statistics(track: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and standard deviation of a pitch track, outliers trimmed."""
    values = [v for v in track if F0_MIN_HZ <= v <= F0_MAX_HZ]
    if len(values) < 3:
        return None, None
    ordered = sorted(values)
    cut = max(1, len(ordered) // 10)
    trimmed = ordered[cut:-cut] or ordered
    mean = sum(trimmed) / len(trimmed)
    variance = sum((v - mean) ** 2 for v in trimmed) / len(trimmed)
    return round(mean, 2), round(math.sqrt(variance), 2)


# --------------------------------------------------------------------------------------
# Pauses and pacing
# --------------------------------------------------------------------------------------


def count_internal_pauses(
    samples: Sequence[float], sample_rate: int, *, min_pause_ms: float = 180.0
) -> int:
    """Silences *inside* a line, ignoring lead-in and tail.

    Pause structure is one of the strongest character-identity cues and one of the
    easiest things for a TTS model to flatten, so it is measured on the real data and
    compared against generated speech later.
    """
    segments = energy_vad(
        samples,
        sample_rate,
        min_speech_ms=80.0,
        min_silence_ms=min_pause_ms,
        speech_pad_ms=0.0,
    )
    return max(0, len(segments) - 1)


def chars_per_second(text: str, duration_s: float) -> float | None:
    """Speaking rate in CJK characters per second."""
    if duration_s <= 0:
        return None
    characters = sum(1 for ch in text if "一" <= ch <= "鿿")
    if characters == 0:
        characters = sum(1 for ch in text if not ch.isspace())
    if characters == 0:
        return None
    return round(characters / duration_s, 3)


def trim_silence(
    samples: Sequence[float],
    sample_rate: int,
    *,
    pad_ms: float = 120.0,
) -> tuple[list[float], float, float]:
    """Trim leading and trailing silence, keeping ``pad_ms`` of it deliberately.

    Returns ``(samples, start_s, end_s)`` relative to the input.
    """
    segments = energy_vad(samples, sample_rate, speech_pad_ms=pad_ms)
    if not segments:
        return list(samples), 0.0, len(samples) / sample_rate
    start_s = segments[0].start_s
    end_s = segments[-1].end_s
    start = int(start_s * sample_rate)
    end = min(len(samples), int(end_s * sample_rate))
    return list(samples[start:end]), start_s, end_s


# --------------------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------------------


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    if HAVE_NUMPY:
        a = _np.asarray(left, dtype=_np.float64)
        b = _np.asarray(right, dtype=_np.float64)
        denominator = float(_np.linalg.norm(a) * _np.linalg.norm(b))
        return 0.0 if denominator == 0 else float(_np.dot(a, b) / denominator)
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Centroid of a set of embeddings, L2-normalized."""
    if not vectors:
        return []
    size = len(vectors[0])
    totals = [0.0] * size
    for vector in vectors:
        for index in range(size):
            totals[index] += vector[index]
    centroid = [value / len(vectors) for value in totals]
    norm = math.sqrt(sum(v * v for v in centroid))
    return [v / norm for v in centroid] if norm else centroid
