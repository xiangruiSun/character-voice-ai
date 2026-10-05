"""Voice Pack preprocessing pipeline.

Runs on synthesized audio with the dependency-free backends, which covers everything
except the model calls themselves: ingest, decode, segmentation, the measurement stages,
auto-rejection, human review round-tripping, and the dataset/reference-bank build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from cvai_core.audio import write_wav
from cvai_core.paths import VoicePackPaths
from cvai_core.voicepack import scaffold_voicepack, validate_voicepack
from cvai_types import (
    CoreStyle,
    RejectionReason,
    ReviewStatus,
    SourceInfo,
    StyleDefinition,
    VoicePackManifest,
)
from cvai_voice_preprocessing import (
    Pipeline,
    PreprocessConfig,
    ReviewPatch,
    ReviewPatchEntry,
    Stage,
    apply_review_patch,
    build_dataset,
    load_state,
    resolve_backends,
    save_state,
)
from cvai_core import dsp
from cvai_voice_preprocessing.backends import STUB_TRANSCRIPT_SOURCE
from cvai_voice_preprocessing.pipeline import BuildError
from cvai_voice_preprocessing.review import render_review_page

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------------------
# Synthetic audio
# --------------------------------------------------------------------------------------


def make_speechlike(
    path: Path,
    *,
    f0: float = 210.0,
    segments: list[tuple[float, float]] | None = None,
    total_s: float = 4.0,
    amplitude: float = 0.4,
    sample_rate: int = SAMPLE_RATE,
    noise: float = 0.0005,
) -> Path:
    """Tone bursts separated by near-silence — enough structure for VAD and pitch."""
    spans = segments or [(0.6, 2.2), (2.8, 3.8)]
    total = int(total_s * sample_rate)
    samples = [0.0] * total
    two_pi = 2.0 * math.pi

    for start_s, end_s in spans:
        start, end = int(start_s * sample_rate), int(end_s * sample_rate)
        for n in range(start, min(end, total)):
            t = (n - start) / sample_rate
            value = (
                math.sin(two_pi * f0 * t)
                + 0.5 * math.sin(two_pi * 2 * f0 * t)
                + 0.25 * math.sin(two_pi * 3 * f0 * t)
            ) / 1.75
            # Syllable-rate envelope, plus fades so the burst edges are not clicks.
            value *= 0.6 + 0.4 * abs(math.sin(two_pi * 3.0 * t))
            fade = int(0.02 * sample_rate)
            if n - start < fade:
                value *= (n - start) / fade
            if end - n < fade:
                value *= max(0.0, (end - n) / fade)
            samples[n] = value * amplitude

    if noise:
        import random

        rng = random.Random(7)
        samples = [s + (rng.random() - 0.5) * noise for s in samples]

    write_wav(path, samples, sample_rate)
    return path


@pytest.fixture(autouse=True)
def builtin_backends_only(monkeypatch):
    """Pin the deterministic built-in backends, whatever is installed.

    With FunASR / Silero installed (`make install-preprocess`, or the mic's local
    STT), backend resolution would otherwise pick real models: minutes of model loading
    per test, and real transcripts where these tests expect the stub's placeholders.
    """
    from cvai_voice_preprocessing.backends import (
        CamPlusPlusSpeakerBackend,
        Emotion2VecBackend,
        FunASRTranscriber,
        SileroVADBackend,
    )

    for backend in (FunASRTranscriber, CamPlusPlusSpeakerBackend,
                    Emotion2VecBackend, SileroVADBackend):
        monkeypatch.setattr(backend, "available", lambda self: False)


@pytest.fixture
def pack(tmp_path: Path) -> VoicePackPaths:
    paths = VoicePackPaths(tmp_path / "voicepacks" / "test_pack")
    manifest = VoicePackManifest(
        voicepack_id="test_pack",
        character_id="test_pack",
        display_name="Test",
        target_sample_rate=SAMPLE_RATE,
        styles=[
            StyleDefinition(name=name, core_style=CoreStyle(name))
            for name in ("neutral", "soft", "teasing")
        ],
        source=SourceInfo(license_note="synthetic test audio"),
    )
    scaffold_voicepack(paths, manifest)
    return paths


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "incoming"
    directory.mkdir()
    make_speechlike(directory / "line_001.wav", f0=200.0)
    make_speechlike(directory / "line_002.wav", f0=240.0, segments=[(0.5, 2.6)])
    make_speechlike(
        directory / "line_003.wav", f0=180.0, segments=[(0.4, 1.6), (2.0, 3.4)]
    )
    return directory


def make_pipeline(pack: VoicePackPaths, **overrides) -> Pipeline:
    from cvai_core.loaders import load_voicepack_manifest

    manifest = load_voicepack_manifest(pack)
    options = dict(
        voicepack_id=manifest.voicepack_id,
        target_sample_rate=SAMPLE_RATE,
        min_segment_s=0.5,
        control_set_size=0,
    )
    options.update(overrides)
    return Pipeline(pack, manifest, PreprocessConfig(**options))


# --------------------------------------------------------------------------------------
# DSP
# --------------------------------------------------------------------------------------


def test_energy_vad_finds_the_bursts(tmp_path: Path):
    path = make_speechlike(tmp_path / "a.wav", segments=[(0.6, 2.2), (2.8, 3.8)])
    from cvai_voice_preprocessing.audio_io import read_samples

    samples, rate = read_samples(path)
    found = dsp.energy_vad(samples, rate, speech_pad_ms=0.0)

    assert len(found) == 2
    assert found[0].start_s == pytest.approx(0.6, abs=0.15)
    assert found[0].end_s == pytest.approx(2.2, abs=0.15)
    assert found[1].start_s == pytest.approx(2.8, abs=0.15)


def test_vad_padding_keeps_audio_before_the_onset(tmp_path: Path):
    """Padding exists to preserve the breath before a line (spec §7)."""
    path = make_speechlike(tmp_path / "a.wav", segments=[(1.0, 2.5)])
    from cvai_voice_preprocessing.audio_io import read_samples

    samples, rate = read_samples(path)
    tight = dsp.energy_vad(samples, rate, speech_pad_ms=0.0)
    padded = dsp.energy_vad(samples, rate, speech_pad_ms=200.0)

    assert padded[0].start_s < tight[0].start_s
    assert padded[0].start_s == pytest.approx(tight[0].start_s - 0.2, abs=0.05)


def test_f0_tracking_recovers_the_synthesized_pitch(tmp_path: Path):
    path = make_speechlike(tmp_path / "a.wav", f0=220.0, segments=[(0.2, 3.0)])
    from cvai_voice_preprocessing.audio_io import read_samples

    samples, rate = read_samples(path)
    mean, spread = dsp.f0_statistics(dsp.estimate_f0_track(samples, rate))

    assert mean is not None
    assert mean == pytest.approx(220.0, rel=0.08)
    assert spread is not None and spread < 40.0


def test_pitch_separates_two_different_voices(tmp_path: Path):
    low = make_speechlike(tmp_path / "low.wav", f0=140.0, segments=[(0.2, 3.0)])
    high = make_speechlike(tmp_path / "high.wav", f0=300.0, segments=[(0.2, 3.0)])
    from cvai_voice_preprocessing.audio_io import read_samples

    low_mean, _ = dsp.f0_statistics(dsp.estimate_f0_track(*read_samples(low)))
    high_mean, _ = dsp.f0_statistics(dsp.estimate_f0_track(*read_samples(high)))
    assert low_mean and high_mean and high_mean > low_mean * 1.7


def test_clipping_is_detected():
    clean = [0.5 * math.sin(i / 10.0) for i in range(1000)]
    clipped = [1.0 if i % 2 else -1.0 for i in range(1000)]
    assert dsp.clipping_ratio(clean) == 0.0
    assert dsp.clipping_ratio(clipped) > 0.9


def test_loudness_normalization_is_gain_only_and_respects_the_peak_ceiling():
    quiet = [0.02 * math.sin(i / 8.0) for i in range(SAMPLE_RATE)]
    louder, gain, measured = dsp.normalize_loudness(
        quiet, SAMPLE_RATE, -20.0, true_peak_ceiling_dbfs=-1.0
    )
    assert gain > 0
    assert dsp.peak_dbfs(louder) <= -1.0 + 0.01
    # Gain only: the waveform's shape is unchanged, so ratios between samples hold.
    ratios = [b / a for a, b in zip(quiet[100:110], louder[100:110]) if abs(a) > 1e-6]
    assert max(ratios) - min(ratios) < 1e-4
    assert measured < 0


def test_chars_per_second_counts_cjk():
    assert dsp.chars_per_second("今天天气不错", 2.0) == pytest.approx(3.0)
    assert dsp.chars_per_second("", 2.0) is None


def test_cosine_similarity_and_centroid():
    assert dsp.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert dsp.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    centroid = dsp.mean_vector([[1.0, 0.0], [0.0, 1.0]])
    assert dsp.cosine_similarity(centroid, [1.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


def test_backend_resolution_reports_what_is_missing():
    bundle = resolve_backends(allow_stub_asr=True)
    report = bundle.report()
    assert set(report) >= {"vad", "asr", "speaker", "emotion"}
    # VAD and speaker always resolve to something usable.
    assert report["vad"].available
    assert report["speaker"].available
    text = bundle.render_report()
    assert "vad" in text and "asr" in text


def test_stub_asr_is_refused_when_disallowed(monkeypatch):
    import cvai_voice_preprocessing.backends as backends_module

    monkeypatch.setattr(
        backends_module.FunASRTranscriber, "available", lambda self: False
    )
    with pytest.raises(RuntimeError):
        resolve_backends(allow_stub_asr=False)


def test_stub_transcripts_are_labelled_as_such(tmp_path: Path):
    from cvai_voice_preprocessing.backends import StubASRBackend

    path = make_speechlike(tmp_path / "a.wav")
    result = StubASRBackend().transcribe(path)
    assert result.model == STUB_TRANSCRIPT_SOURCE
    assert result.confidence == 0.0
    # Visibly a placeholder, so a reviewer cannot accept it by accident.
    assert "人工" in result.text or "ASR" in result.text


def test_spectral_speaker_embedding_separates_distinct_voices(tmp_path: Path):
    from cvai_voice_preprocessing.backends import SpectralSpeakerBackend

    backend = SpectralSpeakerBackend()
    low_a = backend.embed(make_speechlike(tmp_path / "la.wav", f0=130.0, segments=[(0.1, 3.0)]))
    low_b = backend.embed(make_speechlike(tmp_path / "lb.wav", f0=135.0, segments=[(0.1, 3.0)]))
    high = backend.embed(make_speechlike(tmp_path / "h.wav", f0=330.0, segments=[(0.1, 3.0)]))

    same = dsp.cosine_similarity(low_a, low_b)
    different = dsp.cosine_similarity(low_a, high)
    assert same > different


def test_emotion_label_normalization():
    from cvai_voice_preprocessing.backends import normalize_emotion_label

    assert normalize_emotion_label("生气/angry") == "angry"
    assert normalize_emotion_label("happy") == "happy"
    assert normalize_emotion_label("other") == "unknown"
    assert normalize_emotion_label("something_new") == "unknown"


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def test_full_pipeline_runs_and_produces_segments(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()

    assert len(state.sources) == 3
    assert state.segments, "segmentation produced nothing"
    for segment in state.segments:
        assert (pack.root / segment.audio_path).is_file()
        assert segment.duration_s > 0
        assert segment.transcript  # stub ASR still fills something
        assert segment.quality_score is not None
        assert segment.f0_mean_hz is not None
        assert segment.speaker_similarity is not None

    summary = state.summary()
    assert summary["segments"] == len(state.segments)
    assert "quality" in summary["stages_completed"]


def test_raw_audio_is_copied_and_checksummed_and_never_rewritten(
    pack: VoicePackPaths, source_dir: Path
):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    pipeline._run_one(Stage.INGEST)

    raw_files = sorted(pack.raw.rglob("*.wav"))
    assert len(raw_files) == 3
    before = {p: p.read_bytes() for p in raw_files}
    checksums = {c.clip_id: c.checksum_sha256 for c in pipeline.state.sources}
    assert all(value and len(value) == 64 for value in checksums.values())

    # Everything after ingest writes to processed/, never back to raw/.
    pipeline.run()
    for path, content in before.items():
        assert path.read_bytes() == content


def test_ingest_is_idempotent(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    pipeline._run_one(Stage.INGEST)
    first = len(pipeline.state.sources)
    pipeline._run_one(Stage.INGEST)
    assert len(pipeline.state.sources) == first


def test_segmentation_is_not_duplicated_on_rerun(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    pipeline.run([Stage.INGEST, Stage.DECODE, Stage.SEGMENT])
    first = len(pipeline.state.segments)
    pipeline._run_one(Stage.SEGMENT)
    assert len(pipeline.state.segments) == first


def test_no_segment_mode_gives_one_segment_per_file(
    pack: VoicePackPaths, source_dir: Path
):
    pipeline = make_pipeline(pack, source_dir=str(source_dir), segment=False)
    state = pipeline.run([Stage.INGEST, Stage.DECODE, Stage.SEGMENT])
    assert len(state.segments) == len(state.sources)


def test_optional_cleaning_is_off_by_default_and_says_why(
    pack: VoicePackPaths, source_dir: Path
):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    pipeline.run([Stage.INGEST, Stage.DECODE, Stage.SEPARATE, Stage.DENOISE])

    separate = pipeline.state.last_run(Stage.SEPARATE)
    denoise = pipeline.state.last_run(Stage.DENOISE)
    assert separate and separate.skipped and "off" in separate.skip_reason
    assert denoise and denoise.skipped and "breath" in denoise.skip_reason


def test_processing_chain_records_every_stage(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    chain = state.segments[0].processing_chain
    stages = {step.stage.value for step in chain}
    assert "ingest" in stages  # decode, at the same rate, so not "resample"
    assert "vad_segment" in stages
    assert "transcribe" in stages
    # No optional cleaning ran, so nothing claims to have altered the waveform. This is
    # what makes "did cleaning help?" answerable later (decision D7).
    assert not any(step.is_destructive() for step in chain)


def test_resampling_is_recorded_only_when_the_rate_actually_changes(
    pack: VoicePackPaths, tmp_path: Path
):
    source = tmp_path / "incoming"
    source.mkdir()
    make_speechlike(source / "hi.wav", segments=[(0.3, 2.5)], sample_rate=22050)

    pipeline = make_pipeline(pack, source_dir=str(source))
    pipeline.run([Stage.INGEST, Stage.DECODE])
    chain = pipeline.state.sources[0].processing_chain
    assert any(step.stage.value == "resample" for step in chain)


def test_short_and_clipped_segments_are_auto_rejected(pack: VoicePackPaths, tmp_path: Path):
    source = tmp_path / "incoming"
    source.mkdir()
    # A very short burst, and a badly clipped one.
    make_speechlike(source / "short.wav", segments=[(0.4, 0.62)], total_s=1.2)
    make_speechlike(source / "clipped.wav", segments=[(0.3, 2.5)], amplitude=4.0)

    pipeline = make_pipeline(pack, source_dir=str(source), min_segment_s=1.0)
    state = pipeline.run()

    reasons = {
        s.rejection_reason
        for s in state.segments
        if s.review_status is ReviewStatus.REJECTED
    }
    assert RejectionReason.TOO_SHORT in reasons or RejectionReason.CLIPPED_AUDIO in reasons


def test_nothing_is_approved_without_a_human_by_default(
    pack: VoicePackPaths, source_dir: Path
):
    """Spec §8 expects human correction; silent auto-approval is how bad data ships."""
    state = make_pipeline(pack, source_dir=str(source_dir)).run()
    assert all(s.review_status is not ReviewStatus.APPROVED for s in state.segments)
    assert any(s.review_status is ReviewStatus.PENDING for s in state.segments)


def test_auto_approve_is_available_when_asked_for(pack: VoicePackPaths, source_dir: Path):
    state = make_pipeline(
        pack, source_dir=str(source_dir), auto_approve=True
    ).run()
    assert any(s.review_status is ReviewStatus.APPROVED for s in state.segments)


def test_state_is_persisted_and_resumable(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    pipeline.run([Stage.INGEST, Stage.DECODE, Stage.SEGMENT])

    reloaded = load_state(pack.processed, "test_pack")
    assert len(reloaded.segments) == len(pipeline.state.segments)
    assert Stage.SEGMENT in reloaded.completed_stages()

    resumed = make_pipeline(pack, source_dir=str(source_dir))
    assert len(resumed.state.segments) == len(pipeline.state.segments)


def test_speaker_anchors_change_the_centroid(pack: VoicePackPaths, tmp_path: Path):
    source = tmp_path / "incoming"
    source.mkdir()
    make_speechlike(source / "target_a.wav", f0=300.0, segments=[(0.3, 3.0)])
    make_speechlike(source / "target_b.wav", f0=310.0, segments=[(0.3, 3.0)])
    make_speechlike(source / "intruder.wav", f0=120.0, segments=[(0.3, 3.0)])

    pipeline = make_pipeline(pack, source_dir=str(source))
    pipeline.run([Stage.INGEST, Stage.DECODE, Stage.SEGMENT, Stage.SPEAKER_FILTER])

    target_segments = [
        s for s in pipeline.state.segments if s.clip_id.startswith("target")
    ]
    intruder = next(s for s in pipeline.state.segments if "intruder" in s.clip_id)

    pipeline.state.speaker_anchor_segment_ids = [s.segment_id for s in target_segments]
    pipeline._run_one(Stage.SPEAKER_FILTER)

    anchored_target = min(s.speaker_similarity for s in target_segments)
    assert anchored_target > intruder.speaker_similarity


# --------------------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------------------


def test_review_patch_applies_and_marks_human_edited(
    pack: VoicePackPaths, source_dir: Path
):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    segment = state.segments[0]

    patch = ReviewPatch(
        voicepack_id="test_pack",
        reviewer="tester",
        entries=[
            ReviewPatchEntry(
                segment_id=segment.segment_id,
                transcript="今天外面风有点大。",
                style="soft",
                review_status=ReviewStatus.APPROVED,
                speaker_anchor=True,
            )
        ],
    )
    changed = apply_review_patch(state, patch)

    assert changed == 1
    assert segment.transcript == "今天外面风有点大。"
    assert segment.transcript_source == "human"
    assert segment.style == "soft"
    assert segment.human_edited
    # Approving an edited transcript is a *correction*, which is worth distinguishing.
    assert segment.review_status is ReviewStatus.CORRECTED
    assert segment.segment_id in state.speaker_anchor_segment_ids


def test_human_edits_survive_a_rerun(pack: VoicePackPaths, source_dir: Path):
    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    segment = state.segments[0]

    apply_review_patch(
        state,
        ReviewPatch(
            voicepack_id="test_pack",
            entries=[
                ReviewPatchEntry(
                    segment_id=segment.segment_id,
                    transcript="人工校对过的文本。",
                    style="teasing",
                    review_status=ReviewStatus.APPROVED,
                )
            ],
        ),
    )
    save_state(state, pack.processed)

    again = make_pipeline(pack, source_dir=str(source_dir))
    again.run([Stage.TRANSCRIBE, Stage.ANNOTATE, Stage.QUALITY])
    after = again.state.segment(segment.segment_id)

    assert after is not None
    assert after.transcript == "人工校对过的文本。"
    assert after.style == "teasing"
    assert after.review_status is ReviewStatus.CORRECTED


def test_patch_for_the_wrong_pack_is_refused(pack: VoicePackPaths, source_dir: Path):
    state = make_pipeline(pack, source_dir=str(source_dir)).run()
    with pytest.raises(ValueError):
        apply_review_patch(state, ReviewPatch(voicepack_id="some_other_pack"))


def test_review_page_is_self_contained_and_orders_worst_first(
    pack: VoicePackPaths, source_dir: Path
):
    from cvai_core.loaders import load_voicepack_manifest

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    html = render_review_page(state, load_voicepack_manifest(pack))

    assert "<audio" in html or 'createElement("audio")' in html
    assert "http://" not in html and "https://" not in html

    payload = json.loads(
        html.split('<script id="payload" type="application/json">')[1].split("</script>")[0]
    )
    scores = [
        s["metrics"]["quality"] for s in payload["segments"] if s["metrics"]["quality"]
    ]
    assert scores == sorted(scores), "review queue should show the worst clips first"
    # Placeholder transcripts are flagged so a reviewer cannot rubber-stamp them.
    assert all(s["is_stub"] for s in payload["segments"])


# --------------------------------------------------------------------------------------
# Dataset build
# --------------------------------------------------------------------------------------


def _approve_all(state, style_cycle=("neutral", "soft", "teasing")):
    entries = []
    for index, segment in enumerate(state.segments):
        entries.append(
            ReviewPatchEntry(
                segment_id=segment.segment_id,
                transcript=f"这是第{index + 1}条测试台词，用来构建数据集。",
                style=style_cycle[index % len(style_cycle)],
                review_status=ReviewStatus.APPROVED,
            )
        )
    apply_review_patch(state, ReviewPatch(voicepack_id=state.voicepack_id, entries=entries))


def test_build_refuses_stub_transcripts(pack: VoicePackPaths, source_dir: Path):
    from cvai_core.loaders import load_voicepack_manifest

    pipeline = make_pipeline(pack, source_dir=str(source_dir), auto_approve=True)
    state = pipeline.run()

    with pytest.raises(BuildError) as exc:
        build_dataset(
            pack, load_voicepack_manifest(pack), state, pipeline.config
        )
    assert "placeholder" in str(exc.value)


def test_build_writes_clean_dataset_and_reference_bank(
    pack: VoicePackPaths, source_dir: Path
):
    from cvai_core.loaders import load_voicepack_manifest

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    _approve_all(state)

    manifest = load_voicepack_manifest(pack)
    dataset, bank = build_dataset(pack, manifest, state, pipeline.config)

    assert dataset.usable()
    for sample in dataset.usable():
        assert (pack.root / sample.audio_path).is_file()
        assert sample.audio_path.startswith("clean/")
        assert sample.transcript
        assert not sample.audio_path.startswith("raw/")

    assert bank.samples
    for reference in bank.samples:
        assert (pack.root / reference.audio_path).is_file()
        # Decision D3: engines need the reference text.
        assert reference.transcript.strip()

    # Written where the rest of the system looks for them.
    assert pack.dataset_file.is_file()
    assert pack.references_file.is_file()


def test_heldout_lines_are_not_used_as_reference_prompts(
    pack: VoicePackPaths, source_dir: Path
):
    """Otherwise the benchmark's ground truth leaks into the systems being measured."""
    from cvai_core.loaders import load_voicepack_manifest
    from cvai_types import DatasetSplit

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    _approve_all(state)

    manifest = load_voicepack_manifest(pack)
    dataset, bank = build_dataset(pack, manifest, state, pipeline.config)

    heldout_transcripts = {
        s.transcript
        for s in dataset.samples
        if dataset.split_of(s.sample_id) is DatasetSplit.HELDOUT
    }
    reference_transcripts = {r.transcript for r in bank.samples}
    assert not (heldout_transcripts & reference_transcripts)


def test_build_normalizes_loudness_and_records_it(pack: VoicePackPaths, source_dir: Path):
    from cvai_core.loaders import load_voicepack_manifest
    from cvai_voice_preprocessing.audio_io import read_samples

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    _approve_all(state)

    dataset, _ = build_dataset(pack, load_voicepack_manifest(pack), state, pipeline.config)
    sample = dataset.usable()[0]

    samples, rate = read_samples(pack.root / sample.audio_path)
    assert dsp.peak_dbfs(samples) <= pipeline.config.true_peak_ceiling_dbfs + 0.05
    loudness_steps = [
        step for step in sample.processing_chain if step.stage.value == "loudness"
    ]
    assert loudness_steps, "loudness change must be recorded in the processing chain"
    assert "target_lufs" in loudness_steps[0].params


def test_built_pack_passes_strict_validation(pack: VoicePackPaths, source_dir: Path):
    from cvai_core.loaders import load_voicepack_manifest

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    _approve_all(state)
    build_dataset(pack, load_voicepack_manifest(pack), state, pipeline.config)

    report = validate_voicepack(pack, require_dataset=True, require_references=True)
    errors = [issue.code for issue in report.errors()]
    # A tiny synthetic pack legitimately warns about size; it must not error.
    assert not errors, report.render()


def test_build_without_approvals_explains_what_to_do(pack: VoicePackPaths, source_dir: Path):
    from cvai_core.loaders import load_voicepack_manifest

    pipeline = make_pipeline(pack, source_dir=str(source_dir))
    state = pipeline.run()
    with pytest.raises(BuildError) as exc:
        build_dataset(pack, load_voicepack_manifest(pack), state, pipeline.config)
    assert "review" in str(exc.value)
