"""Objective metrics for a benchmark run (Milestone 6).

Objective metrics **rank candidates and catch regressions**. They do not choose the
winner — decision D8, and the reason is in `docs/TECH_LANDSCAPE.md`: reference-free
quality predictors degrade precisely in the high-quality regime this project operates
in. A model that scores 4.3 where another scores 4.2 has not been shown to be better;
a model whose pitch distribution sits a third above the character's has been shown to
be wrong, and that is the kind of thing measured here.

Two tiers:

**Prosody comparison — always available.** The character's real speech has a measurable
distribution: pitch centre and spread, speaking rate, pause density, seconds per
character. Generated speech can be measured the same way with the same code
(`cvai_core.dsp`, shared with the Voice Pack pipeline precisely so the two agree) and
compared. This is how spec §27's "identical intonation across all sentences" and
"robotic pacing" become numbers instead of impressions — a candidate whose F0 spread is
40% of the character's is monotone, whatever it scores on naturalness.

**Model-backed metrics — when installed.** Speaker similarity (SECS) against real
character audio, character error rate from ASR, and TTSDS2. Each reports its own absence
rather than silently returning nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

from cvai_core import dsp
from cvai_core.audio import read_wav_properties, read_wav_samples
from cvai_core.logging_setup import get_logger
from cvai_core.paths import VoicePackPaths
from cvai_core.runlog import RunPaths
from cvai_types import (
    BenchmarkRun,
    CVAIModel,
    DatasetManifest,
    GenerationRecord,
    ObjectiveMetrics,
)
from pydantic import Field

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Measuring one clip
# --------------------------------------------------------------------------------------


class ClipMeasurement(CVAIModel):
    """What can be measured from one audio file plus its text, with no model."""

    duration_s: float
    sample_rate: int
    f0_mean_hz: float | None = None
    f0_std_hz: float | None = None
    chars_per_second: float | None = None
    pause_count: int | None = None
    pauses_per_second: float | None = None
    seconds_per_char: float | None = None
    peak_dbfs: float | None = None
    clipping_ratio: float | None = None
    snr_db: float | None = None
    #: Fraction of the clip that is silence. A model that pads every line with half a
    #: second of nothing sounds sluggish in conversation without scoring badly anywhere.
    silence_ratio: float | None = None


def measure_clip(path: Path, text: str = "") -> ClipMeasurement:
    """Measure a generated or real clip."""
    samples, rate = read_wav_samples(Path(path))
    duration = len(samples) / rate if rate else 0.0
    if duration <= 0:
        raise ValueError(f"{path} contains no audio")

    track = dsp.estimate_f0_track(samples, rate)
    mean, spread = dsp.f0_statistics(track)
    pauses = dsp.count_internal_pauses(samples, rate)
    speech = dsp.energy_vad(samples, rate, speech_pad_ms=0.0)
    speech_time = sum(segment.duration_s for segment in speech)

    characters = sum(1 for ch in text if "一" <= ch <= "鿿")
    return ClipMeasurement(
        duration_s=round(duration, 4),
        sample_rate=rate,
        f0_mean_hz=mean,
        f0_std_hz=spread,
        chars_per_second=dsp.chars_per_second(text, duration) if text else None,
        pause_count=pauses,
        pauses_per_second=round(pauses / duration, 4) if duration else None,
        seconds_per_char=round(duration / characters, 4) if characters else None,
        peak_dbfs=round(dsp.peak_dbfs(samples), 2),
        clipping_ratio=round(dsp.clipping_ratio(samples), 6),
        snr_db=dsp.estimate_snr_db(samples, rate),
        silence_ratio=round(max(0.0, 1.0 - speech_time / duration), 4),
    )


# --------------------------------------------------------------------------------------
# The character's real distribution
# --------------------------------------------------------------------------------------


class ProsodyProfile(CVAIModel):
    """How the character actually speaks, measured from real recordings.

    Built from the **held-out** split by preference: those are real lines no engine was
    trained on, so the comparison is against the character rather than against the
    training set a candidate may have memorised.
    """

    source: str = "heldout"
    n_samples: int = 0
    f0_mean_hz: float | None = None
    f0_std_hz: float | None = None
    #: Spread *across* clips of the per-clip pitch spread. A character with expressive
    #: range has a high within-clip σ; this records how much that itself varies.
    f0_std_of_std: float | None = None
    chars_per_second_mean: float | None = None
    chars_per_second_std: float | None = None
    pauses_per_second_mean: float | None = None
    seconds_per_char_mean: float | None = None
    silence_ratio_mean: float | None = None

    @classmethod
    def from_measurements(
        cls, measurements: Sequence[ClipMeasurement], *, source: str = "heldout"
    ) -> "ProsodyProfile":
        def stats(values: list[float]) -> tuple[float | None, float | None]:
            clean = [v for v in values if v is not None]
            if not clean:
                return None, None
            return (
                round(fmean(clean), 3),
                round(pstdev(clean) if len(clean) > 1 else 0.0, 3),
            )

        f0_means, f0_std = stats([m.f0_mean_hz for m in measurements if m.f0_mean_hz])
        spread_mean, spread_of_spread = stats(
            [m.f0_std_hz for m in measurements if m.f0_std_hz is not None]
        )
        cps_mean, cps_std = stats(
            [m.chars_per_second for m in measurements if m.chars_per_second]
        )
        pauses_mean, _ = stats(
            [m.pauses_per_second for m in measurements if m.pauses_per_second is not None]
        )
        spc_mean, _ = stats(
            [m.seconds_per_char for m in measurements if m.seconds_per_char]
        )
        silence_mean, _ = stats(
            [m.silence_ratio for m in measurements if m.silence_ratio is not None]
        )

        return cls(
            source=source,
            n_samples=len(measurements),
            f0_mean_hz=f0_means,
            f0_std_hz=spread_mean,
            f0_std_of_std=spread_of_spread,
            chars_per_second_mean=cps_mean,
            chars_per_second_std=cps_std,
            pauses_per_second_mean=pauses_mean,
            seconds_per_char_mean=spc_mean,
            silence_ratio_mean=silence_mean,
        )

    @classmethod
    def from_dataset(
        cls,
        paths: VoicePackPaths,
        dataset: DatasetManifest,
        *,
        prefer_heldout: bool = True,
        limit: int = 40,
    ) -> "ProsodyProfile":
        """Measure the character from her own recordings.

        Prefers the annotations already stored on each sample — the Voice Pack pipeline
        measured them with this same code — and only re-reads audio for samples that
        predate those fields.
        """
        from cvai_types import DatasetSplit

        samples = dataset.usable()
        heldout = [
            s
            for s in samples
            if dataset.split_of(s.sample_id) is DatasetSplit.HELDOUT
        ]
        chosen = heldout if (prefer_heldout and heldout) else samples
        source = "heldout" if chosen is heldout else "all_approved"
        chosen = chosen[:limit]

        measurements: list[ClipMeasurement] = []
        for sample in chosen:
            if sample.f0_mean_hz is not None and sample.chars_per_second is not None:
                measurements.append(
                    ClipMeasurement(
                        duration_s=sample.audio.duration_s,
                        sample_rate=sample.audio.sample_rate,
                        f0_mean_hz=sample.f0_mean_hz,
                        f0_std_hz=sample.f0_std_hz,
                        chars_per_second=sample.chars_per_second,
                        pause_count=sample.pause_count,
                        pauses_per_second=(
                            round(sample.pause_count / sample.audio.duration_s, 4)
                            if sample.pause_count is not None
                            and sample.audio.duration_s
                            else None
                        ),
                        seconds_per_char=(
                            round(1.0 / sample.chars_per_second, 4)
                            if sample.chars_per_second
                            else None
                        ),
                    )
                )
                continue
            audio = paths.root / sample.audio_path
            if audio.is_file():
                try:
                    measurements.append(measure_clip(audio, sample.transcript))
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not measure %s: %s", audio, exc)

        return cls.from_measurements(measurements, source=source)


# --------------------------------------------------------------------------------------
# Comparing a candidate against the character
# --------------------------------------------------------------------------------------


class ProsodyComparison(CVAIModel):
    candidate_id: str
    n_clips: int = 0
    f0_mean_delta_hz: float | None = None
    #: Generated σ divided by the character's σ. Below ~0.6 the delivery is flatter than
    #: the character's — spec §27's "identical intonation across all sentences".
    f0_std_ratio: float | None = None
    speaking_rate_delta_cps: float | None = None
    pauses_per_second_delta: float | None = None
    silence_ratio_delta: float | None = None
    #: Single summary in σ-like units; lower is closer to the character.
    prosody_distance: float | None = None
    notes: list[str] = Field(default_factory=list)


def compare_prosody(
    candidate_id: str,
    measurements: Sequence[ClipMeasurement],
    profile: ProsodyProfile,
) -> ProsodyComparison:
    """Compare a candidate's generated speech against the character's real distribution."""
    comparison = ProsodyComparison(candidate_id=candidate_id, n_clips=len(measurements))
    if not measurements:
        comparison.notes.append("no clips to measure")
        return comparison

    def mean_of(attribute: str) -> float | None:
        values = [
            getattr(m, attribute) for m in measurements if getattr(m, attribute) is not None
        ]
        return round(fmean(values), 4) if values else None

    f0 = mean_of("f0_mean_hz")
    f0_spread = mean_of("f0_std_hz")
    cps = mean_of("chars_per_second")
    pauses = mean_of("pauses_per_second")
    silence = mean_of("silence_ratio")

    terms: list[float] = []

    if f0 is not None and profile.f0_mean_hz:
        comparison.f0_mean_delta_hz = round(f0 - profile.f0_mean_hz, 2)
        # Normalize by the character's own pitch spread; 20 Hz matters much more for a
        # monotone speaker than for an expressive one.
        scale = profile.f0_std_hz or 20.0
        terms.append(abs(comparison.f0_mean_delta_hz) / max(5.0, scale))
        if abs(comparison.f0_mean_delta_hz) > 2 * max(5.0, scale):
            comparison.notes.append(
                f"pitch centre is {comparison.f0_mean_delta_hz:+.0f} Hz from the "
                "character's — this will not sound like her regardless of naturalness"
            )

    if f0_spread is not None and profile.f0_std_hz:
        comparison.f0_std_ratio = round(f0_spread / profile.f0_std_hz, 3)
        terms.append(abs(1.0 - comparison.f0_std_ratio))
        if comparison.f0_std_ratio < 0.6:
            comparison.notes.append(
                f"pitch variation is {comparison.f0_std_ratio:.0%} of the character's — "
                "flat delivery (spec §27: identical intonation across all sentences)"
            )
        elif comparison.f0_std_ratio > 1.6:
            comparison.notes.append(
                f"pitch variation is {comparison.f0_std_ratio:.0%} of the character's — "
                "over-acted"
            )

    if cps is not None and profile.chars_per_second_mean:
        comparison.speaking_rate_delta_cps = round(cps - profile.chars_per_second_mean, 3)
        scale = profile.chars_per_second_std or 0.5
        terms.append(abs(comparison.speaking_rate_delta_cps) / max(0.2, scale))
        ratio = cps / profile.chars_per_second_mean
        if ratio < 0.75 or ratio > 1.33:
            comparison.notes.append(
                f"speaking rate is {ratio:.0%} of the character's"
            )

    if pauses is not None and profile.pauses_per_second_mean is not None:
        comparison.pauses_per_second_delta = round(
            pauses - profile.pauses_per_second_mean, 4
        )
        terms.append(abs(comparison.pauses_per_second_delta) / 0.3)

    if silence is not None and profile.silence_ratio_mean is not None:
        comparison.silence_ratio_delta = round(silence - profile.silence_ratio_mean, 4)
        if comparison.silence_ratio_delta > 0.15:
            comparison.notes.append(
                "clips carry noticeably more silence than the character's real lines — "
                "check for padding at the start and end"
            )

    if terms:
        # RMS rather than a plain mean. A candidate that matches the character on three
        # axes and is badly wrong on the fourth is *not* three-quarters right: one
        # severe deviation — a flat pitch contour, say — is enough to stop it sounding
        # like her, and averaging would dilute exactly that signal.
        comparison.prosody_distance = round(
            (sum(term * term for term in terms) / len(terms)) ** 0.5, 4
        )
    return comparison


# --------------------------------------------------------------------------------------
# Whole-run scoring
# --------------------------------------------------------------------------------------


@dataclass
class ObjectiveBackends:
    """Model-backed metrics, each optional and each reporting its own absence."""

    speaker_similarity: object | None = None
    asr: object | None = None
    ttsds2: object | None = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def resolve_objective_backends(device: str = "cpu") -> ObjectiveBackends:
    """Pick up whatever is installed for SECS / CER / TTSDS2."""
    backends = ObjectiveBackends()
    try:
        from cvai_voice_preprocessing.backends import CamPlusPlusSpeakerBackend

        candidate = CamPlusPlusSpeakerBackend(device=device)
        if candidate.available():
            backends.speaker_similarity = candidate
        else:
            backends.notes.append(
                "speaker similarity (SECS) unavailable: " + candidate.unavailable_reason()
            )
    except ImportError:  # pragma: no cover
        backends.notes.append("speaker similarity unavailable: preprocessing extra missing")

    try:
        from cvai_voice_preprocessing.backends import FunASRTranscriber

        candidate = FunASRTranscriber(device=device)
        if candidate.available():
            backends.asr = candidate
        else:
            backends.notes.append("CER unavailable: " + candidate.unavailable_reason())
    except ImportError:  # pragma: no cover
        backends.notes.append("CER unavailable: preprocessing extra missing")

    try:  # pragma: no cover - not installed in this environment
        import ttsds  # noqa: F401

        backends.ttsds2 = ttsds
    except ImportError:
        backends.notes.append(
            "TTSDS2 unavailable: `pip install ttsds` for the factored objective score"
        )
    return backends


class RunObjectiveReport(CVAIModel):
    run_id: str
    profile: ProsodyProfile
    comparisons: list[ProsodyComparison] = Field(default_factory=list)
    metrics: dict[str, ObjectiveMetrics] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def ranked_by_prosody(self) -> list[ProsodyComparison]:
        return sorted(
            self.comparisons,
            key=lambda c: (c.prosody_distance is None, c.prosody_distance or 0.0),
        )


def score_run(
    run: BenchmarkRun,
    run_paths: RunPaths,
    profile: ProsodyProfile,
    *,
    backends: ObjectiveBackends | None = None,
) -> RunObjectiveReport:
    """Measure every generated clip and compare each candidate to the character."""
    report = RunObjectiveReport(run_id=run.run_id, profile=profile)
    if backends is not None:
        report.notes.extend(backends.notes)
    if not profile.n_samples:
        report.notes.append(
            "no real recordings were available to build a prosody profile; "
            "the comparison below is empty. Add a held-out split to the voice pack."
        )

    by_candidate: dict[str, list[ClipMeasurement]] = {}
    latencies: dict[str, list[float]] = {}

    for record in run.succeeded_records():
        if not record.audio_path:
            continue
        path = run_paths.root / record.audio_path
        if not path.is_file():
            continue
        try:
            measurement = measure_clip(path, record.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not measure %s: %s", path, exc)
            continue
        by_candidate.setdefault(record.candidate_id, []).append(measurement)
        if record.latency_ms and record.duration_s:
            latencies.setdefault(record.candidate_id, []).append(
                (record.latency_ms / 1000.0) / record.duration_s
            )

    for candidate_id, measurements in sorted(by_candidate.items()):
        comparison = compare_prosody(candidate_id, measurements, profile)
        report.comparisons.append(comparison)
        report.metrics[candidate_id] = ObjectiveMetrics(
            f0_mean_delta_hz=comparison.f0_mean_delta_hz,
            f0_std_ratio=comparison.f0_std_ratio,
            speaking_rate_delta_cps=comparison.speaking_rate_delta_cps,
            rtf=round(fmean(latencies[candidate_id]), 4)
            if latencies.get(candidate_id)
            else None,
        )

    return report


def render_objective_report(report: RunObjectiveReport) -> str:
    """Markdown for the objective section of an evaluation."""
    lines = [f"## Objective metrics · `{report.run_id}`", ""]
    profile = report.profile
    lines.append(
        f"Character reference distribution from **{profile.n_samples} "
        f"{profile.source}** recordings:"
    )
    lines.append("")
    if profile.f0_mean_hz is None:
        lines.append("- pitch: not measured")
    else:
        spread = (
            f"σ {profile.f0_std_hz:.0f}" if profile.f0_std_hz else "σ not measurable"
        )
        lines.append(f"- pitch {profile.f0_mean_hz:.0f} Hz ({spread})")
    if profile.chars_per_second_mean:
        lines.append(f"- speaking rate {profile.chars_per_second_mean:.2f} chars/s")
    if profile.pauses_per_second_mean is not None:
        lines.append(f"- pauses {profile.pauses_per_second_mean:.2f} /s")
    lines.append("")

    if not profile.f0_std_hz:
        # Without a reference spread there is nothing to compare a candidate's pitch
        # variation against, and "identical intonation across all sentences" becomes
        # undetectable. Worth saying, rather than leaving a column of dashes.
        lines.append(
            "> The reference recordings have no measurable pitch variation, so the "
            "F0 σ ratio column is empty and flat delivery cannot be detected. This is "
            "expected for synthetic demo audio; for a real pack it means the held-out "
            "clips are too few or too short."
        )
        lines.append("")

    if report.notes:
        for note in report.notes:
            lines.append(f"> {note}")
        lines.append("")

    lines.append(
        "| Candidate | clips | ΔF0 (Hz) | F0 σ ratio | Δrate (c/s) | Δpauses /s | "
        "distance | RTF |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for comparison in report.ranked_by_prosody():
        metrics = report.metrics.get(comparison.candidate_id)
        lines.append(
            "| `{id}` | {n} | {f0} | {ratio} | {rate} | {pauses} | {dist} | {rtf} |".format(
                id=comparison.candidate_id,
                n=comparison.n_clips,
                f0=_fmt(comparison.f0_mean_delta_hz, "{:+.0f}"),
                ratio=_fmt(comparison.f0_std_ratio, "{:.2f}"),
                rate=_fmt(comparison.speaking_rate_delta_cps, "{:+.2f}"),
                pauses=_fmt(comparison.pauses_per_second_delta, "{:+.2f}"),
                dist=_fmt(comparison.prosody_distance, "{:.2f}"),
                rtf=_fmt(metrics.rtf if metrics else None, "{:.2f}"),
            )
        )
    lines.append("")

    flagged = [(c.candidate_id, note) for c in report.comparisons for note in c.notes]
    if flagged:
        lines.append("### Flags")
        lines.append("")
        for candidate_id, note in flagged:
            lines.append(f"- `{candidate_id}`: {note}")
        lines.append("")

    lines.append(
        "_These rank candidates and catch regressions. They do not choose the winner: "
        "reference-free quality metrics degrade in exactly the high-quality regime this "
        "project operates in (decision D8). The listening test decides._"
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None, spec: str) -> str:
    return spec.format(value) if value is not None else "—"


def probe_audio(path: Path) -> tuple[int, float]:
    properties = read_wav_properties(Path(path))
    return properties.sample_rate, properties.duration_s
