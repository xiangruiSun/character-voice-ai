"""The benchmark pipeline, end to end, on the synthetic pack.

This is the test that matters most for Milestone 1: if it passes, the apparatus that
answers the project's first technical question works, and Milestone 3 only has to supply
a real engine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from cvai_core.config import load_config
from cvai_core.registry import build_tts_provider
from cvai_evaluation import (
    ALL_AXES,
    BenchmarkRunner,
    aggregate,
    build_blind_test,
    gap_to_ground_truth,
    load_benchmark_config,
    render_evaluation_report,
    render_run_report,
    simulate_ratings,
    write_blind_test,
)
from cvai_evaluation.blind import GROUND_TRUTH_CANDIDATE_ID
from cvai_types import AdaptationMode, RatingAxis

SENTENCES = {
    "set_id": "test_set",
    "sentences": [
        {"sentence_id": "t1", "text": "今天外面风有点大。", "target_style": "neutral"},
        {"sentence_id": "t2", "text": "你猜呢？", "target_style": "teasing"},
        {"sentence_id": "t3", "text": "够了。", "target_style": "angry"},
    ],
}


@pytest.fixture
def bench_env(tmp_path: Path, demo_pack, repo_root_path: Path, monkeypatch):
    """A self-contained benchmark: temp pack, temp sentence set, temp runs directory."""
    sentence_file = tmp_path / "sentences.yaml"
    sentence_file.write_text(
        yaml.safe_dump(SENTENCES, allow_unicode=True), encoding="utf-8"
    )

    benchmark_file = tmp_path / "bench.yaml"
    benchmark_file.write_text(
        yaml.safe_dump(
            {
                "benchmark_id": "unit",
                "voicepack_id": demo_pack.root.name,
                "sentence_set": str(sentence_file.relative_to(tmp_path)),
                "repeats": 1,
                "base_seed": 1000,
                "skip_unavailable": False,
                "ground_truth": {"enabled": True, "source": "heldout", "max_items": 2},
                "candidates": [
                    {
                        "candidate_id": "mock_a",
                        "engine": "mock",
                        "adaptation_mode": "zero_shot",
                        "display_name": "Mock A",
                    },
                    {
                        "candidate_id": "mock_b",
                        "engine": "mock",
                        "adaptation_mode": "zero_shot",
                        "display_name": "Mock B",
                    },
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    # `sentence_set` resolves against the repo root, so point that at tmp_path.
    monkeypatch.setenv("CVAI_REPO_ROOT", str(tmp_path))

    config = load_config([repo_root_path / "configs" / "app.yaml"], use_env_overrides=False)
    benchmark = load_benchmark_config(benchmark_file)
    runner = BenchmarkRunner(
        config,
        benchmark,
        runs_root=tmp_path / "runs",
        packs_root=demo_pack.root.parent,
    )
    return runner


def test_run_generates_every_candidate_sentence_pair(bench_env):
    run = asyncio.run(bench_env.run())

    assert len(run.records) == 6  # 2 candidates × 3 sentences
    assert len(run.failures()) == 0
    for record in run.succeeded_records():
        audio = bench_env.run_paths.root / record.audio_path
        assert audio.is_file(), record.record_id
        assert record.duration_s and record.duration_s > 0
        # Spec §23: every record must be traceable back to its configuration.
        assert record.voicepack_version
        assert record.seed is not None
        assert record.resolved_params


def test_candidates_share_seeds_and_references_per_sentence(bench_env):
    """Otherwise the comparison measures sampling noise, not the engines."""
    run = asyncio.run(bench_env.run())
    by_sentence: dict[str, list] = {}
    for record in run.succeeded_records():
        by_sentence.setdefault(record.sentence_id, []).append(record)

    for sentence_id, records in by_sentence.items():
        assert len({r.seed for r in records}) == 1, sentence_id
        assert len({r.reference_id for r in records}) == 1, sentence_id


def test_fallback_is_recorded_when_a_style_has_no_clips(bench_env):
    """The demo pack has no `angry` clips, so t3 must fall back and say so."""
    run = asyncio.run(bench_env.run())
    angry = [r for r in run.succeeded_records() if r.sentence_id == "t3"]
    assert angry and all(r.reference_was_fallback for r in angry)
    assert run.fallback_rate() > 0


def test_run_directory_captures_reproducibility_context(bench_env):
    run = asyncio.run(bench_env.run())
    paths = bench_env.run_paths

    assert paths.run_file.is_file()
    assert paths.config_file.is_file()
    assert paths.env_file.is_file()
    assert paths.events_file.is_file()

    environment = json.loads(paths.env_file.read_text(encoding="utf-8"))
    assert "python" in environment and "platform" in environment

    events = [
        json.loads(line)
        for line in paths.events_file.read_text(encoding="utf-8").splitlines()
    ]
    kinds = {event["kind"] for event in events}
    assert {"run.start", "candidate.start", "generate.ok", "run.finish"} <= kinds
    assert run.config_hash


def test_unavailable_engine_is_skipped_not_fatal(bench_env, monkeypatch):
    from cvai_types import TTSHealth

    real_factory = bench_env.provider_factory

    def factory(config, name):
        provider = real_factory(config, name)
        if getattr(provider, "instance_name", name) == "mock":
            async def unavailable():
                return TTSHealth(engine="mock", available=False, detail="sidecar down")

            provider.health = unavailable  # type: ignore[method-assign]
        return provider

    bench_env.provider_factory = factory
    bench_env.benchmark.skip_unavailable = True
    run = asyncio.run(bench_env.run())

    assert len(run.succeeded_records()) == 0
    assert len(run.failures()) == 6
    assert all("sidecar down" in (r.error or "") for r in run.failures())
    # And the report still renders, which is what makes a partial run useful.
    assert "Failures" in render_run_report(run)


def test_blind_test_hides_the_system_and_keeps_the_key_separate(bench_env):
    run = asyncio.run(bench_env.run())
    test, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=3)

    assert len(test.items) == len(run.succeeded_records()) + 2  # + ground truth anchors

    # Nothing in the rater-facing set identifies the producing system: not the candidate
    # id, not the sentence id, and not a path that encodes either.
    serialized = json.dumps(test.model_dump(mode="json"), ensure_ascii=False)
    for candidate in run.candidates:
        assert candidate.candidate_id not in serialized
    assert "ground_truth" not in serialized
    assert "heldout" not in serialized
    for item in test.items:
        assert item.audio_path == f"audio/{item.item_id}.wav"

    assert {e.item_id for e in key.entries} == {i.item_id for i in test.items}
    assert any(e.is_ground_truth for e in key.entries)
    # The key keeps the trail back to the generation record.
    assert all(e.source_audio_path for e in key.entries)


def test_blind_shuffle_is_reproducible_from_the_seed(bench_env):
    run = asyncio.run(bench_env.run())
    _, first = build_blind_test(run, bench_env.run_paths, shuffle_seed=42)
    _, second = build_blind_test(run, bench_env.run_paths, shuffle_seed=42)
    _, other = build_blind_test(run, bench_env.run_paths, shuffle_seed=43)

    # Item ids are positional now, so the assignment of *sources* to positions is what
    # the seed controls.
    order = lambda key: [e.source_audio_path for e in key.entries]  # noqa: E731
    assert order(first) == order(second)
    assert order(first) != order(other)


def test_rating_page_is_self_contained(bench_env):
    run = asyncio.run(bench_env.run())
    test, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=1)
    written = write_blind_test(bench_env.run_paths, test, key)

    html = written["page"].read_text(encoding="utf-8")
    assert "<audio" in html or "createElement(\"audio\")" in html
    # No external requests: the page has to work offline on someone else's laptop.
    assert "http://" not in html and "https://" not in html
    # The key must not leak into the page.
    assert "candidate_id" not in html


def test_aggregation_ranks_candidates_and_anchors_on_real_audio(bench_env):
    run = asyncio.run(bench_env.run())
    _, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=1)

    ratings = simulate_ratings(
        key, n_raters=3, seed=5, candidate_bias={"mock_a": 0.8, "mock_b": -0.8}
    )
    report = aggregate(run, key, ratings)

    ranked = report.ranked()
    assert ranked[0].is_ground_truth, "real recordings should top a sane simulation"
    non_anchor = [c for c in ranked if not c.is_ground_truth]
    assert non_anchor[0].candidate_id == "mock_a"

    gaps = gap_to_ground_truth(report, RatingAxis.SPEAKER_SIMILARITY)
    assert gaps["mock_a"] > gaps["mock_b"]
    assert all(value <= 0 for value in gaps.values())


def test_artifact_axis_is_inverted_when_composing(bench_env):
    """A low ai_artifact_level is good; the composite must not rank it backwards."""
    run = asyncio.run(bench_env.run())
    _, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=1)
    ratings = simulate_ratings(key, n_raters=2, seed=9)
    report = aggregate(run, key, ratings)

    for candidate in report.candidates:
        artifact = next(
            a for a in candidate.axes if a.axis is RatingAxis.AI_ARTIFACT_LEVEL
        )
        assert artifact.higher_is_better is False
        assert candidate.composite() is not None


def test_missing_anchor_is_called_out_in_the_notes(bench_env):
    run = asyncio.run(bench_env.run())
    _, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=1)
    key.entries = [e for e in key.entries if not e.is_ground_truth]
    ratings = simulate_ratings(key, n_raters=2, seed=1)
    report = aggregate(run, key, ratings)
    assert "ground-truth" in report.notes


def test_reports_render(bench_env):
    run = asyncio.run(bench_env.run())
    _, key = build_blind_test(run, bench_env.run_paths, shuffle_seed=1)
    ratings = simulate_ratings(key, n_raters=3, seed=2)
    report = aggregate(run, key, ratings)

    run_md = render_run_report(run)
    assert "Benchmark run" in run_md and "Candidates" in run_md

    eval_md = render_evaluation_report(
        run, report, agreement={axis.value: 0.5 for axis in ALL_AXES}
    )
    assert "Gap to real recordings" in eval_md
    assert "Inter-rater agreement" in eval_md


def test_adaptation_mode_unsupported_by_the_engine_is_skipped(bench_env, monkeypatch):
    from cvai_types import ProviderCapabilities

    real_factory = bench_env.provider_factory

    def factory(config, name):
        provider = real_factory(config, name)
        provider.capabilities = lambda: ProviderCapabilities(  # type: ignore[method-assign]
            engine="mock",
            supported_adaptation_modes=[AdaptationMode.FINETUNED],
        )
        return provider

    bench_env.provider_factory = factory
    for candidate in bench_env.benchmark.candidates:
        candidate.adaptation_mode = AdaptationMode.ZERO_SHOT
    run = asyncio.run(bench_env.run())
    assert all(not r.succeeded for r in run.records)
    assert "adaptation mode" in (run.records[0].error or "")


# --------------------------------------------------------------------------------------
# The null-processing control (decision D7)
# --------------------------------------------------------------------------------------


def _add_control_candidate(bench_env, **overrides):
    """Clone Mock A as a control conditioned on unprocessed references."""
    from cvai_types import BenchmarkCandidate

    control = BenchmarkCandidate(
        candidate_id="mock_a_unprocessed",
        engine="mock",
        adaptation_mode="zero_shot",
        display_name="Mock A (control)",
        reference_source="unprocessed",
        **overrides,
    )
    bench_env.benchmark.candidates.append(control)
    return control


def test_the_control_hears_the_unprocessed_clips(bench_env):
    """Same engine, same seeds, same lines — only the reference audio differs."""
    _add_control_candidate(bench_env)
    run = asyncio.run(bench_env.run())

    control = [r for r in run.records if r.candidate_id == "mock_a_unprocessed"]
    baseline = [r for r in run.records if r.candidate_id == "mock_a"]
    assert control and all(r.succeeded for r in control)
    assert all(r.reference_source == "unprocessed" for r in control)
    assert all(r.reference_source == "clean" for r in baseline)

    # It is a control, not a different experiment: seeds and chosen clips must match.
    by_sentence = {r.sentence_id: r for r in baseline}
    for record in control:
        partner = by_sentence[record.sentence_id]
        assert record.seed == partner.seed
        assert record.reference_id == partner.reference_id

    rebuilt = bench_env.run_paths.root / "unprocessed_references" / "mock_a_unprocessed"
    assert list(rebuilt.glob("*.wav"))


def test_a_control_without_provenance_is_refused_not_faked(bench_env, demo_pack):
    """The failure mode this guards against is silent success.

    If the unprocessed clips cannot be rebuilt and the control quietly falls back to the
    cleaned ones, it produces a confident result saying the cleaning changed nothing —
    which is the exact opposite of what happened.
    """
    import json

    # Strip the provenance the demo pack records, as an older pack would have.
    bank_file = demo_pack.references_file
    bank = json.loads(bank_file.read_text(encoding="utf-8"))
    for sample in bank["samples"]:
        sample["source_clip"] = None
        sample["source_offset_s"] = None
    bank_file.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")

    _add_control_candidate(bench_env)
    run = asyncio.run(bench_env.run())

    control = [r for r in run.records if r.candidate_id == "mock_a_unprocessed"]
    assert control
    assert all(not r.succeeded for r in control)
    assert "provenance" in control[0].error or "does not record" in control[0].error
    # The rest of the run is unharmed: one missing control does not cost four hours.
    assert all(r.succeeded for r in run.records if r.candidate_id == "mock_a")


def test_the_report_labels_the_control_as_a_control(bench_env):
    """Ranked next to real candidates without a label, a control reads as a competitor."""
    from cvai_evaluation.report import render_run_report

    _add_control_candidate(bench_env)
    run = asyncio.run(bench_env.run())
    report = render_run_report(run)
    assert "unprocessed (control)" in report
