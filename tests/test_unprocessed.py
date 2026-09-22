"""The null-processing control (decision D7, spec §27).

Preprocessing is the one part of this project that can destroy the thing it is meant to
prepare. Separation, denoising and loudness normalisation each remove something, and
breath and room are part of why a voice sounds like a person. The only honest way to know
whether the chain preserved the character is to run the benchmark on the same lines
uncleaned and let the listening test choose.

These tests hold the control to the one property that makes it worth anything: it must be
*the same line, uncleaned*, and it must refuse rather than quietly substitute the cleaned
clip when it cannot be. A control that silently becomes a copy of the thing it controls
for does not fail loudly — it produces a confident, wrong result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cvai_core.audio import read_wav_properties, read_wav_samples, write_wav
from cvai_core.paths import VoicePackPaths
from cvai_evaluation.unprocessed import (
    UnprocessedReferenceError,
    rebuild_all,
    rebuild_unprocessed,
)
from cvai_types import AudioProperties, ReferenceSample

RATE = 24000
LEAD_S = 0.4
BODY_S = 2.0


def make_pack(tmp_path: Path, *, with_source: bool = True) -> VoicePackPaths:
    """A minimal pack: one original recording, one cleaned reference cut from it."""
    paths = VoicePackPaths(tmp_path / "demo")
    (paths.root / "raw").mkdir(parents=True)
    (paths.root / "references" / "neutral").mkdir(parents=True)

    lead = [0.01] * int(LEAD_S * RATE)
    body = [0.2] * int(BODY_S * RATE)
    if with_source:
        write_wav(paths.root / "raw" / "line.wav", lead + body + lead, RATE)
    # The "cleaned" version: louder, and without the lead-in.
    write_wav(paths.root / "references" / "neutral" / "neutral_01.wav", [0.6] * len(body), RATE)
    return paths


def make_sample(*, source: str | None = "raw/line.wav", offset: float = LEAD_S) -> ReferenceSample:
    return ReferenceSample(
        reference_id="neutral_01",
        audio_path="references/neutral/neutral_01.wav",
        transcript="今天天气不错。",
        style="neutral",
        core_style="neutral",
        audio=AudioProperties(sample_rate=RATE, channels=1, duration_s=BODY_S),
        source_clip=source,
        source_offset_s=offset if source else None,
    )


def test_the_rebuilt_clip_is_the_same_line_from_the_original(tmp_path: Path):
    paths = make_pack(tmp_path)
    out = rebuild_unprocessed(make_sample(), paths, tmp_path / "out" / "neutral_01.wav")

    properties = read_wav_properties(out)
    assert properties.sample_rate == RATE
    assert properties.duration_s == pytest.approx(BODY_S, abs=0.01)

    samples, _ = read_wav_samples(out)
    # The body of the original, not the lead-in and not the cleaned clip.
    assert samples[100] == pytest.approx(0.2, abs=0.01)


def test_the_level_is_left_alone(tmp_path: Path):
    """No loudness match, deliberately.

    Level is one of the things the cleaning chain changes. Normalising it here would
    hide exactly the effect the control exists to measure.
    """
    paths = make_pack(tmp_path)
    out = rebuild_unprocessed(make_sample(), paths, tmp_path / "out" / "neutral_01.wav")

    raw_samples, _ = read_wav_samples(out)
    clean_samples, _ = read_wav_samples(
        paths.root / "references" / "neutral" / "neutral_01.wav"
    )
    assert max(raw_samples) < max(clean_samples)


def test_a_pack_without_provenance_refuses(tmp_path: Path):
    paths = make_pack(tmp_path)
    with pytest.raises(UnprocessedReferenceError, match="does not record"):
        rebuild_unprocessed(make_sample(source=None), paths, tmp_path / "x.wav")


def test_a_missing_original_recording_says_which_one(tmp_path: Path):
    paths = make_pack(tmp_path, with_source=False)
    with pytest.raises(UnprocessedReferenceError, match="raw/line.wav"):
        rebuild_unprocessed(make_sample(), paths, tmp_path / "x.wav")


def test_an_offset_past_the_end_of_the_recording_refuses(tmp_path: Path):
    paths = make_pack(tmp_path)
    with pytest.raises(UnprocessedReferenceError, match="claims to start"):
        rebuild_unprocessed(make_sample(offset=99.0), paths, tmp_path / "x.wav")


def test_rebuild_all_reports_what_it_could_not_do(tmp_path: Path):
    """Partial success is reported as such, never as success."""
    paths = make_pack(tmp_path)
    good = make_sample()
    bad = make_sample(source=None).model_copy(update={"reference_id": "neutral_02"})

    rebuilt, problems = rebuild_all([good, bad], paths, tmp_path / "out")
    assert set(rebuilt) == {"neutral_01"}
    assert len(problems) == 1
    assert "neutral_02" in problems[0]


# --------------------------------------------------------------------------------------
# Through the benchmark
# --------------------------------------------------------------------------------------


def test_the_demo_pack_carries_provenance(demo_pack):
    """The synthetic pack has to exercise the control, or it goes untested until a real
    pack is built — which is months later and on someone else's machine."""
    from cvai_core.loaders import try_load_reference_bank

    bank = try_load_reference_bank(demo_pack)
    assert bank is not None
    for sample in bank.samples:
        assert sample.source_clip, f"{sample.reference_id} has no provenance"
        assert (demo_pack.root / sample.source_clip).is_file()
        assert sample.source_offset_s is not None


def test_the_control_candidate_is_conditioned_on_different_audio(demo_pack, tmp_path: Path):
    """The point of the whole exercise: the control must actually hear something else."""
    from cvai_core.loaders import try_load_reference_bank

    bank = try_load_reference_bank(demo_pack)
    rebuilt, problems = rebuild_all(bank.samples, demo_pack, tmp_path / "unprocessed")
    assert not problems
    assert len(rebuilt) == len(bank.samples)

    sample = bank.samples[0]
    cleaned, _ = read_wav_samples(demo_pack.root / sample.audio_path)
    control, _ = read_wav_samples(rebuilt[sample.reference_id])
    assert len(control) == pytest.approx(len(cleaned), rel=0.02)
    assert control != cleaned


def test_a_candidate_declares_which_references_it_heard():
    """Spec §23: a result that cannot say what conditioned it is not reproducible."""
    from cvai_types import BenchmarkCandidate, GenerationRecord

    candidate = BenchmarkCandidate(
        candidate_id="mock_control", engine="mock", reference_source="unprocessed"
    )
    assert candidate.reference_source == "unprocessed"
    assert GenerationRecord(
        record_id="r", candidate_id="c", sentence_id="s", text="在吗",
        engine="mock", voicepack_id="demo_zh", voicepack_version="0.1.0",
    ).reference_source == "clean"

    with pytest.raises(ValueError):
        BenchmarkCandidate(
            candidate_id="mock_control", engine="mock", reference_source="raw-ish"
        )
