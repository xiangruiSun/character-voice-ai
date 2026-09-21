"""Dataset exporters (Milestones 3-5).

The behaviour that matters most is negative: **no exporter may emit a held-out sample**.
Those lines are the benchmark's ground truth and the listening test's hidden anchor, and
training on them would quietly invalidate every number the project later produces. It is
checked once per engine here, and enforced in the base class so a future exporter cannot
regress it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cvai_core.loaders import load_dataset_manifest, load_voicepack_manifest
from cvai_types import DatasetSplit
from cvai_voice_training import EXPORTERS, export_dataset, get_exporter


@pytest.fixture(scope="module")
def pack_with_dataset(shared_demo_pack):
    """The exporters only read the pack, so one build serves every test here."""
    manifest = load_voicepack_manifest(shared_demo_pack)
    dataset = load_dataset_manifest(shared_demo_pack)
    return shared_demo_pack, manifest, dataset


@pytest.fixture(scope="module")
def heldout_transcripts(pack_with_dataset) -> set[str]:
    _, _, dataset = pack_with_dataset
    return {
        s.transcript
        for s in dataset.samples
        if dataset.split_of(s.sample_id) is DatasetSplit.HELDOUT
    }


def _all_text(root: Path) -> str:
    """Every text-ish file in an export, concatenated."""
    chunks: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".list", ".jsonl", ".lab", ".txt", ".scp", ""}:
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, OSError):
                continue
    return "\n".join(chunks)


@pytest.mark.parametrize("engine", sorted(EXPORTERS))
def test_no_exporter_leaks_heldout_lines(
    engine: str, pack_with_dataset, heldout_transcripts: set[str], tmp_path: Path
):
    paths, manifest, dataset = pack_with_dataset
    assert heldout_transcripts, "fixture should have held-out samples to leak"

    out = tmp_path / engine
    result = export_dataset(engine, paths, manifest, dataset, out)

    body = _all_text(out)
    for transcript in heldout_transcripts:
        assert transcript not in body, f"{engine} exported a held-out line"
    assert result.excluded_heldout == len(
        [s for s in dataset.samples if dataset.split_of(s.sample_id) is DatasetSplit.HELDOUT]
    )


@pytest.mark.parametrize("engine", sorted(EXPORTERS))
def test_every_export_records_its_provenance(engine: str, pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / engine
    export_dataset(engine, paths, manifest, dataset, out)

    payload = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert payload["engine"] == engine
    assert payload["voicepack_id"] == manifest.voicepack_id
    assert payload["voicepack_version"] == manifest.version
    assert payload["train"] > 0
    assert "minutes_by_style" in payload


@pytest.mark.parametrize("engine", sorted(EXPORTERS))
def test_exports_copy_audio_rather_than_referencing_the_pack(
    engine: str, pack_with_dataset, tmp_path: Path
):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / engine
    export_dataset(engine, paths, manifest, dataset, out)

    copied = list(out.rglob("*.wav"))
    assert copied, f"{engine} exported no audio"
    # clean/ stays canonical and untouched.
    assert all((paths.root / s.audio_path).is_file() for s in dataset.usable())


def test_gpt_sovits_list_format(pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "gs"
    export_dataset("gpt_sovits", paths, manifest, dataset, out, speaker_name="denia")

    list_file = out / "denia.list"
    rows = [line for line in list_file.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    for row in rows:
        audio, speaker, language, text = row.split("|", 3)
        assert Path(audio).is_file()
        assert speaker == "denia"
        assert language == "ZH"
        assert text and "\n" not in text


def test_fish_speech_pairs_lab_files_with_audio(pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "fish"
    export_dataset("fish_speech", paths, manifest, dataset, out, speaker_name="denia")

    speaker_dir = out / "data" / "denia"
    wavs = sorted(speaker_dir.glob("*.wav"))
    assert wavs
    for wav in wavs:
        label = wav.with_suffix(".lab")
        assert label.is_file()
        assert label.read_text(encoding="utf-8").strip()


def test_qwen_jsonl_rows_are_complete(pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "qwen"
    export_dataset("qwen3_tts", paths, manifest, dataset, out)

    rows = [
        json.loads(line)
        for line in (out / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows
    for row in rows:
        assert Path(row["audio"]).is_file()
        assert row["text"] and row["language"] == "Chinese"
        assert row["duration"] > 0
    # A reference clip and its transcript travel with the export, because this engine
    # needs ref_text at inference.
    assert (out / "reference.wav").is_file()
    assert (out / "reference.txt").read_text(encoding="utf-8").strip()


def test_cosyvoice_kaldi_files_line_up(pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "cosy"
    export_dataset("cosyvoice", paths, manifest, dataset, out, speaker_name="denia")

    train = out / "train"
    wav_ids = [line.split()[0] for line in (train / "wav.scp").read_text().splitlines() if line]
    text_ids = [line.split()[0] for line in (train / "text").read_text(encoding="utf-8").splitlines() if line]
    utt_ids = [line.split()[0] for line in (train / "utt2spk").read_text().splitlines() if line]

    assert wav_ids and wav_ids == text_ids == utt_ids
    assert (train / "spk2utt").read_text().startswith("denia ")


def test_voxcpm_rows_carry_a_prompt(pack_with_dataset, tmp_path: Path):
    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "vox"
    export_dataset("voxcpm", paths, manifest, dataset, out)

    rows = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows
    assert all("prompt_audio" in row and "prompt_text" in row for row in rows)
    assert Path(rows[0]["prompt_audio"]).is_file()


def test_index_tts_has_no_exporter_and_says_why():
    with pytest.raises(KeyError) as exc:
        get_exporter("index_tts")
    assert "zero-shot" in str(exc.value)


def test_unknown_engine_lists_the_known_ones():
    with pytest.raises(KeyError) as exc:
        get_exporter("not_an_engine")
    assert "gpt_sovits" in str(exc.value)


def test_export_without_approved_samples_explains_itself(
    pack_with_dataset, tmp_path: Path
):
    from cvai_types import DatasetManifest

    paths, manifest, dataset = pack_with_dataset
    empty = DatasetManifest(
        voicepack_id=dataset.voicepack_id,
        voicepack_version=dataset.voicepack_version,
        samples=[],
        splits=[],
    )
    with pytest.raises(ValueError) as exc:
        export_dataset("gpt_sovits", paths, manifest, empty, tmp_path / "x")
    assert "cvai-prep build" in str(exc.value)


def test_transcripts_are_flattened_to_one_line(pack_with_dataset, tmp_path: Path):
    """A newline inside a pipe-delimited list file silently corrupts the dataset."""
    from cvai_voice_training.exporters import _one_line

    assert _one_line("第一行\n第二行  多余空格") == "第一行 第二行 多余空格"

    paths, manifest, dataset = pack_with_dataset
    out = tmp_path / "gs"
    export_dataset("gpt_sovits", paths, manifest, dataset, out)
    content = (out / f"{manifest.character_id}.list").read_text(encoding="utf-8")
    for line in content.splitlines():
        if line:
            assert line.count("|") == 3
