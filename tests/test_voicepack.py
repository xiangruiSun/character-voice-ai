"""Voice pack scaffolding and validation.

The validator is the gate before training, so its job is to *fail* on the things that
quietly ruin a character voice. Most tests here assert that it complains.
"""

from __future__ import annotations

from pathlib import Path

from cvai_core.loaders import (
    load_reference_bank,
    load_voicepack_manifest,
    save_model_json,
    save_model_yaml,
)
from cvai_core.paths import VoicePackPaths
from cvai_core.voicepack import Severity, scaffold_voicepack, validate_voicepack
from cvai_types import (
    AudioProperties,
    CharacterProfile,
    CoreStyle,
    DatasetEntry,
    DatasetManifest,
    DatasetSplit,
    ReferenceBank,
    ReferenceSample,
    ReviewStatus,
    StyleDefinition,
    TrainingSample,
    VoiceBinding,
    VoicePackManifest,
)


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _manifest(pack_id: str = "demo_zh", styles=("neutral", "soft")) -> VoicePackManifest:
    return VoicePackManifest(
        voicepack_id=pack_id,
        character_id=pack_id,
        display_name=pack_id,
        styles=[StyleDefinition(name=s, core_style=CoreStyle(s)) for s in styles],
    )


def test_scaffolding_creates_the_documented_layout(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "demo_zh")
    report = scaffold_voicepack(paths, _manifest())

    for directory in ("raw", "processed", "clean", "rejected", "transcripts",
                      "metadata", "references", "datasets", "checkpoints", "evaluation"):
        assert (paths.root / directory).is_dir(), directory
    # A subdirectory per declared style, so the bank's shape is visible before it exists.
    assert (paths.references / "neutral").is_dir()
    assert paths.manifest_file.is_file()
    assert (paths.root / "README.md").is_file()

    # An empty pack is structurally valid but obviously incomplete.
    assert report.ok
    assert "references.missing" in _codes(report)
    assert "dataset.missing" in _codes(report)


def test_missing_manifest_is_an_error(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "empty")
    paths.root.mkdir(parents=True)
    report = validate_voicepack(paths)
    assert not report.ok
    assert "manifest.missing" in _codes(report)


def test_directory_name_must_match_the_manifest(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "actually_other")
    scaffold_voicepack(paths, _manifest("demo_zh"))
    report = validate_voicepack(paths)
    assert "manifest.id_mismatch" in _codes(report)
    assert not report.ok


def test_style_with_no_reference_clips_is_an_error(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "demo_zh")
    scaffold_voicepack(paths, _manifest(styles=("neutral", "soft")))

    audio = paths.references / "neutral" / "n1.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    from cvai_core.audio import write_wav

    write_wav(audio, [0.0] * 24000 * 3, 24000)

    bank = ReferenceBank(
        voicepack_id="demo_zh",
        voicepack_version="0.1.0",
        samples=[
            ReferenceSample(
                reference_id="n1",
                audio_path="references/neutral/n1.wav",
                transcript="你好。",
                style="neutral",
                core_style=CoreStyle.NEUTRAL,
                audio=AudioProperties(sample_rate=24000, channels=1, duration_s=3.0),
            )
        ],
    )
    save_model_json(bank, paths.references_file)

    report = validate_voicepack(paths)
    codes = _codes(report)
    # soft has nothing at all …
    assert "references.style_empty" in codes
    # … and neutral has only one clip, so every neutral line uses the same audio.
    assert "references.style_thin" in codes
    assert not report.ok


def test_reference_pointing_at_a_missing_file_is_an_error(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "demo_zh")
    scaffold_voicepack(paths, _manifest(styles=("neutral",)))
    bank = ReferenceBank(
        voicepack_id="demo_zh",
        voicepack_version="0.1.0",
        samples=[
            ReferenceSample(
                reference_id="ghost",
                audio_path="references/neutral/ghost.wav",
                transcript="你好。",
                style="neutral",
                core_style=CoreStyle.NEUTRAL,
                audio=AudioProperties(sample_rate=24000, channels=1, duration_s=3.0),
            )
        ],
    )
    save_model_json(bank, paths.references_file)
    assert "references.missing_audio" in _codes(validate_voicepack(paths))


def test_small_dataset_and_missing_heldout_are_warnings(demo_pack):
    report = validate_voicepack(demo_pack)
    codes = _codes(report)
    # The synthetic demo pack is deliberately tiny.
    assert "dataset.too_small" in codes
    # …but it does have a held-out split, so that warning must not fire.
    assert "dataset.no_heldout" not in codes
    assert report.stats["dataset_samples_usable"] > 0
    assert isinstance(report.stats["dataset_minutes_by_style"], dict)


def test_missing_heldout_split_warns(tmp_path: Path, demo_pack):
    dataset = DatasetManifest.model_validate_json(
        demo_pack.dataset_file.read_text(encoding="utf-8")
    )
    stripped = DatasetManifest(
        voicepack_id=dataset.voicepack_id,
        voicepack_version=dataset.voicepack_version,
        samples=dataset.samples,
        splits=[
            DatasetEntry(sample_id=e.sample_id, split=DatasetSplit.TRAIN)
            for e in dataset.splits
        ],
    )
    save_model_json(stripped, demo_pack.dataset_file)
    assert "dataset.no_heldout" in _codes(validate_voicepack(demo_pack))


def test_profile_styles_must_be_performable_by_the_pack(demo_pack):
    manifest = load_voicepack_manifest(demo_pack)
    profile = CharacterProfile(
        character_id="demo_zh",
        character_name="Demo",
        voice=VoiceBinding(voicepack_id=manifest.voicepack_id),
        available_styles=["neutral", "a_style_the_pack_cannot_do"],
    )
    report = validate_voicepack(demo_pack, profile=profile)
    assert "profile.unsupported_styles" in _codes(report)
    assert not report.ok


def test_profile_pointing_at_the_wrong_pack_is_an_error(demo_pack):
    profile = CharacterProfile(
        character_id="demo_zh",
        character_name="Demo",
        voice=VoiceBinding(voicepack_id="some_other_pack"),
    )
    assert "profile.pack_mismatch" in _codes(validate_voicepack(demo_pack, profile=profile))


def test_strict_mode_requires_references_and_a_dataset(tmp_path: Path):
    paths = VoicePackPaths(tmp_path / "demo_zh")
    scaffold_voicepack(paths, _manifest())
    lenient = validate_voicepack(paths)
    strict = validate_voicepack(paths, require_dataset=True, require_references=True)
    assert lenient.ok
    assert not strict.ok


def test_shipped_denia_pack_and_profile_agree(repo_root_path: Path):
    """The committed template pack and profile must actually validate together."""
    from cvai_core.loaders import load_character_profile

    paths = VoicePackPaths(repo_root_path / "voicepacks" / "denia_cn")
    profile = load_character_profile(
        repo_root_path / "characters" / "profiles" / "denia_cn.yaml"
    )
    report = validate_voicepack(paths, profile=profile)
    assert report.ok, report.render()


def test_report_renders_without_crashing(demo_pack):
    text = validate_voicepack(demo_pack).render()
    assert "Voice pack:" in text
    assert text.strip().endswith(("PASS", "FAIL"))
