"""The null-processing control (decision D7).

Cleaning a voice pack is not free. Separation, denoising and loudness normalisation each
remove something, and what they remove is not always noise: breath, mouth texture and
room are a large part of why a voice sounds like a person rather than a model. Spec §27
calls the result "over-processing destroys character identity", and the honest way to
find out whether it happened is to run the benchmark twice — once conditioned on the
cleaned reference clips, once on the *same lines* straight from the original recording —
and let the listening test say which is closer to her.

That control is only possible because every reference carries its provenance: which
original file it came from and at what offset. This module uses that to cut the
unprocessed version of a reference out of the original recording, sample-accurately and
with no processing whatsoever, not even loudness.

If the provenance is missing — an older pack, or one assembled by hand — the control
cannot run, and that is reported rather than quietly substituting the cleaned clip. A
control that silently becomes a copy of the thing it is controlling for is worse than no
control at all.
"""

from __future__ import annotations

from pathlib import Path

from cvai_core.audio import read_wav_samples, write_wav
from cvai_core.errors import CVAIError
from cvai_core.logging_setup import get_logger
from cvai_core.paths import VoicePackPaths
from cvai_types import ReferenceSample

log = get_logger(__name__)


class UnprocessedReferenceError(CVAIError):
    """The unprocessed version of a reference could not be rebuilt."""


def rebuild_unprocessed(
    sample: ReferenceSample,
    paths: VoicePackPaths,
    destination: Path,
) -> Path:
    """Cut this reference out of the untouched original recording.

    Returns the path written. Raises :class:`UnprocessedReferenceError` when the pack
    does not record where the clip came from, or the original file is gone.
    """
    if not sample.source_clip:
        raise UnprocessedReferenceError(
            f"reference {sample.reference_id!r} does not record which original "
            "recording it came from, so its unprocessed version cannot be rebuilt. "
            "Packs built before provenance was recorded need `cvai-prep build` re-run."
        )

    source = paths.root / sample.source_clip
    if not source.is_file():
        raise UnprocessedReferenceError(
            f"the original recording for {sample.reference_id!r} is missing: "
            f"{sample.source_clip}. raw/ is not tracked in git, so it may simply not "
            "be on this machine."
        )

    audio, rate = read_wav_samples(source)
    start = int(round((sample.source_offset_s or 0.0) * rate))
    stop = start + int(round(sample.audio.duration_s * rate))
    if start >= len(audio):
        raise UnprocessedReferenceError(
            f"reference {sample.reference_id!r} claims to start at "
            f"{sample.source_offset_s:.2f}s of a {len(audio) / rate:.2f}s recording"
        )

    cut = audio[start:stop]
    if not cut:
        raise UnprocessedReferenceError(
            f"reference {sample.reference_id!r} cuts to nothing from {sample.source_clip}"
        )

    # Deliberately no loudness match. Level is part of what the processing changed, and
    # normalising it here would hide exactly the effect the control exists to measure.
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_wav(destination, cut, rate)
    return destination


def rebuild_all(
    samples: list[ReferenceSample],
    paths: VoicePackPaths,
    out_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Rebuild every reference that can be rebuilt.

    Returns ``(by_reference_id, problems)``. Callers decide whether a partial set is
    usable; the benchmark runner refuses, because a control set that covers half the
    styles compares two different things.
    """
    rebuilt: dict[str, Path] = {}
    problems: list[str] = []
    for sample in samples:
        try:
            rebuilt[sample.reference_id] = rebuild_unprocessed(
                sample, paths, out_dir / f"{sample.reference_id}.wav"
            )
        except UnprocessedReferenceError as exc:
            problems.append(str(exc))
    return rebuilt, problems
