"""Objective metrics (Milestone 6).

These exist to make two of spec §27's failure modes measurable rather than impressionistic:
"identical intonation across all sentences" and "robotic pacing". A candidate whose pitch
spread is half the character's is flat, whatever a listener scores it on naturalness —
and the point of measuring is to know that before the listening test rather than after.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from cvai_core.audio import write_wav
from cvai_evaluation import (
    ClipMeasurement,
    ProsodyProfile,
    compare_prosody,
    measure_clip,
    render_objective_report,
    resolve_objective_backends,
    score_run,
)

SR = 16000


def tone(
    path: Path,
    *,
    f0: float = 220.0,
    duration: float = 3.0,
    vibrato_hz: float = 0.0,
    vibrato_depth: float = 0.0,
    gaps: list[tuple[float, float]] | None = None,
) -> Path:
    """A voiced tone with optional pitch movement and silent gaps."""
    total = int(duration * SR)
    two_pi = 2.0 * math.pi
    samples = []
    phase = 0.0
    for n in range(total):
        t = n / SR
        frequency = f0
        if vibrato_hz:
            frequency = f0 * (1.0 + vibrato_depth * math.sin(two_pi * vibrato_hz * t))
        phase += two_pi * frequency / SR
        value = (math.sin(phase) + 0.4 * math.sin(2 * phase)) / 1.4
        value *= 0.6 + 0.4 * abs(math.sin(two_pi * 3.0 * t))
        if gaps and any(start <= t <= end for start, end in gaps):
            value = 0.0
        samples.append(value * 0.4)
    write_wav(path, samples, SR)
    return path


# --------------------------------------------------------------------------------------
# Measuring
# --------------------------------------------------------------------------------------


def test_measure_clip_recovers_pitch_and_rate(tmp_path: Path):
    text = "今天外面风有点大你出门记得多穿一件"
    path = tone(tmp_path / "a.wav", f0=240.0, duration=3.0)
    measurement = measure_clip(path, text)

    assert measurement.duration_s == pytest.approx(3.0, abs=0.05)
    assert measurement.f0_mean_hz == pytest.approx(240.0, rel=0.08)
    assert measurement.chars_per_second == pytest.approx(len(text) / 3.0, rel=0.05)
    assert measurement.seconds_per_char is not None
    assert 0.0 <= measurement.silence_ratio <= 1.0


def test_internal_gaps_are_counted_as_pauses(tmp_path: Path):
    none = measure_clip(tone(tmp_path / "flat.wav", duration=4.0), "一二三四五六七八")
    paused = measure_clip(
        tone(tmp_path / "paused.wav", duration=4.0, gaps=[(1.2, 2.0)]),
        "一二三四五六七八",
    )
    assert paused.pause_count > none.pause_count
    assert paused.silence_ratio > none.silence_ratio


def test_measuring_an_empty_file_fails_loudly(tmp_path: Path):
    path = tmp_path / "empty.wav"
    write_wav(path, [], SR)
    with pytest.raises(ValueError):
        measure_clip(path)


# --------------------------------------------------------------------------------------
# Profiles and comparison
# --------------------------------------------------------------------------------------


def _profile(**overrides) -> ProsodyProfile:
    data = dict(
        source="heldout",
        n_samples=10,
        f0_mean_hz=240.0,
        f0_std_hz=30.0,
        chars_per_second_mean=4.8,
        chars_per_second_std=0.5,
        pauses_per_second_mean=0.4,
        silence_ratio_mean=0.2,
    )
    data.update(overrides)
    return ProsodyProfile(**data)


def _clip(**overrides) -> ClipMeasurement:
    data = dict(
        duration_s=3.0,
        sample_rate=SR,
        f0_mean_hz=240.0,
        f0_std_hz=30.0,
        chars_per_second=4.8,
        pauses_per_second=0.4,
        silence_ratio=0.2,
    )
    data.update(overrides)
    return ClipMeasurement(**data)


def test_a_matching_candidate_has_near_zero_distance():
    comparison = compare_prosody("good", [_clip()] * 5, _profile())
    assert comparison.prosody_distance == pytest.approx(0.0, abs=0.05)
    assert comparison.notes == []


def test_a_monotone_candidate_is_flagged():
    """Spec §27: identical intonation across all sentences."""
    flat = [_clip(f0_std_hz=9.0)] * 5
    comparison = compare_prosody("flat", flat, _profile())

    assert comparison.f0_std_ratio == pytest.approx(0.3, abs=0.01)
    assert any("flat delivery" in note for note in comparison.notes)
    # And it is scored as clearly worse than a candidate that matches the character —
    # the RMS combination keeps one severe axis from being averaged away.
    matched = compare_prosody("good", [_clip()] * 5, _profile())
    assert comparison.prosody_distance > matched.prosody_distance + 0.3


def test_a_wrong_pitch_centre_is_flagged_regardless_of_naturalness():
    high = [_clip(f0_mean_hz=340.0)] * 5
    comparison = compare_prosody("high", high, _profile())

    assert comparison.f0_mean_delta_hz == pytest.approx(100.0)
    assert any("pitch centre" in note for note in comparison.notes)


def test_a_rushed_candidate_is_flagged():
    fast = [_clip(chars_per_second=7.5)] * 5
    comparison = compare_prosody("fast", fast, _profile())
    assert comparison.speaking_rate_delta_cps > 2.0
    assert any("speaking rate" in note for note in comparison.notes)


def test_excess_padding_is_flagged():
    padded = [_clip(silence_ratio=0.5)] * 5
    comparison = compare_prosody("padded", padded, _profile())
    assert any("silence" in note for note in comparison.notes)


def test_distance_normalizes_by_the_character_own_spread():
    """20 Hz matters more for a monotone character than for an expressive one."""
    off_by_20 = [_clip(f0_mean_hz=260.0)] * 3
    narrow = compare_prosody("x", off_by_20, _profile(f0_std_hz=10.0))
    wide = compare_prosody("x", off_by_20, _profile(f0_std_hz=60.0))
    assert narrow.prosody_distance > wide.prosody_distance


def test_comparison_without_clips_says_so():
    comparison = compare_prosody("none", [], _profile())
    assert comparison.n_clips == 0
    assert comparison.notes


def test_profile_from_measurements():
    clips = [_clip(f0_mean_hz=f) for f in (230.0, 240.0, 250.0)]
    profile = ProsodyProfile.from_measurements(clips)
    assert profile.n_samples == 3
    assert profile.f0_mean_hz == pytest.approx(240.0)
    assert profile.f0_std_hz == pytest.approx(30.0)


def test_profile_prefers_the_heldout_split(shared_demo_pack):
    """Held-out lines are real and untrained-on, which is what makes them the reference."""
    from cvai_core.loaders import load_dataset_manifest

    dataset = load_dataset_manifest(shared_demo_pack)
    profile = ProsodyProfile.from_dataset(shared_demo_pack, dataset)

    assert profile.source == "heldout"
    assert profile.n_samples > 0


# --------------------------------------------------------------------------------------
# Whole-run scoring
# --------------------------------------------------------------------------------------


@pytest.fixture
def scored_run(tmp_path: Path, demo_pack, repo_root_path: Path, monkeypatch):
    """A real benchmark run over the mock engine, then scored objectively."""
    import asyncio

    import yaml
    from cvai_core.config import load_config
    from cvai_evaluation import BenchmarkRunner, load_benchmark_config

    sentences = {
        "set_id": "obj",
        "sentences": [
            {"sentence_id": "o1", "text": "今天外面风有点大。", "target_style": "neutral"},
            {"sentence_id": "o2", "text": "别急，慢慢说。", "target_style": "soft"},
        ],
    }
    (tmp_path / "sentences.yaml").write_text(
        yaml.safe_dump(sentences, allow_unicode=True), encoding="utf-8"
    )
    bench = tmp_path / "bench.yaml"
    bench.write_text(
        yaml.safe_dump(
            {
                "benchmark_id": "obj",
                "voicepack_id": demo_pack.root.name,
                "sentence_set": "sentences.yaml",
                "skip_unavailable": False,
                "candidates": [
                    {"candidate_id": "m1", "engine": "mock", "adaptation_mode": "zero_shot"}
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CVAI_REPO_ROOT", str(tmp_path))
    config = load_config(
        [repo_root_path / "configs" / "app.yaml"], use_env_overrides=False
    )
    runner = BenchmarkRunner(
        config,
        load_benchmark_config(bench),
        runs_root=tmp_path / "runs",
        packs_root=demo_pack.root.parent,
    )
    run = asyncio.run(runner.run())
    return run, runner.run_paths, demo_pack


def test_score_run_measures_every_generated_clip(scored_run):
    from cvai_core.loaders import load_dataset_manifest

    run, run_paths, pack = scored_run
    profile = ProsodyProfile.from_dataset(pack, load_dataset_manifest(pack))
    report = score_run(run, run_paths, profile)

    assert report.run_id == run.run_id
    assert len(report.comparisons) == 1
    comparison = report.comparisons[0]
    assert comparison.candidate_id == "m1"
    assert comparison.n_clips == len(run.succeeded_records())
    assert "m1" in report.metrics
    assert report.metrics["m1"].rtf is not None


def test_score_run_without_a_profile_says_the_comparison_is_empty(scored_run):
    run, run_paths, _ = scored_run
    report = score_run(run, run_paths, ProsodyProfile())
    assert any("prosody profile" in note for note in report.notes)


def test_objective_report_renders_and_says_what_it_is_not(scored_run):
    from cvai_core.loaders import load_dataset_manifest

    run, run_paths, pack = scored_run
    profile = ProsodyProfile.from_dataset(pack, load_dataset_manifest(pack))
    text = render_objective_report(score_run(run, run_paths, profile))

    assert "Objective metrics" in text
    assert "F0 σ ratio" in text
    # The report must never read as if it picked a winner.
    assert "listening test decides" in text


def test_missing_model_backends_are_reported_not_hidden():
    backends = resolve_objective_backends()
    # None of SECS / CER / TTSDS2 are installed here, and each should say so.
    assert backends.notes
    assert any("SECS" in note or "speaker similarity" in note for note in backends.notes)
    assert any("TTSDS2" in note for note in backends.notes)
