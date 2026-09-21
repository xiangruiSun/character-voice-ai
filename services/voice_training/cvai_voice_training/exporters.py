"""Per-engine dataset exporters.

Each engine's training code wants the same data in a different shape:

============  ==============================================================
GPT-SoVITS    one list file, ``audio_path|speaker|language|text``
Fish Speech   ``SPK/clip.wav`` next to ``SPK/clip.lab`` holding the transcript
Qwen3-TTS     ``train_raw.jsonl`` with audio path + text + speaker per line
CosyVoice     Kaldi-style ``wav.scp`` / ``text`` / ``utt2spk`` / ``spk2utt``
VoxCPM        ``train.jsonl`` with audio, text and a reference clip
============  ==============================================================

All of them read from one :class:`DatasetManifest`, and none of them can see the
held-out split — :meth:`DatasetExporter.export` filters it before a subclass is called,
so a new exporter cannot leak it by forgetting to.
"""

from __future__ import annotations

import abc
import json
import shutil
from pathlib import Path

from cvai_core.paths import VoicePackPaths
from cvai_types import (
    CVAIModel,
    DatasetManifest,
    DatasetSplit,
    TrainingSample,
    VoicePackManifest,
    utcnow,
)
from pydantic import Field

EXPORTER_VERSION = "1"

#: GPT-SoVITS language tag for Mandarin.
GPT_SOVITS_LANG = "ZH"


class ExportResult(CVAIModel):
    engine: str
    root: str
    train_count: int = 0
    val_count: int = 0
    excluded_heldout: int = 0
    total_seconds: float = 0.0
    files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Exported {self.train_count} train + {self.val_count} val samples "
            f"({self.total_seconds / 60:.1f} min) for {self.engine}",
            f"  {self.root}",
        ]
        if self.excluded_heldout:
            lines.append(
                f"  {self.excluded_heldout} held-out samples excluded "
                "(benchmark ground truth — never train on these)"
            )
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


class DatasetExporter(abc.ABC):
    """Base class. Subclasses implement :meth:`write`, never the filtering."""

    engine: str = "unknown"
    #: Sample rate the engine's training code expects, when it is opinionated.
    #: ``None`` means "whatever the pack has"; the engine resamples internally.
    target_sample_rate: int | None = None

    def export(
        self,
        paths: VoicePackPaths,
        manifest: VoicePackManifest,
        dataset: DatasetManifest,
        output_dir: Path,
        *,
        speaker_name: str | None = None,
        include_val: bool = True,
    ) -> ExportResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        usable = dataset.usable()
        train: list[TrainingSample] = []
        val: list[TrainingSample] = []
        heldout = 0
        for sample in usable:
            split = dataset.split_of(sample.sample_id)
            if split is DatasetSplit.HELDOUT:
                # The single rule every exporter obeys, enforced here so no subclass
                # has to remember it.
                heldout += 1
            elif split is DatasetSplit.VAL and include_val:
                val.append(sample)
            elif split is DatasetSplit.VAL:
                train.append(sample)
            else:
                train.append(sample)

        if not train:
            raise ValueError(
                f"nothing to export for {self.engine}: the dataset has no approved "
                "training samples (run `cvai-prep build` first)"
            )

        speaker = speaker_name or manifest.character_id
        result = self.write(paths, manifest, output_dir, train, val, speaker)
        result.engine = self.engine
        result.root = str(output_dir)
        result.train_count = len(train)
        result.val_count = len(val)
        result.excluded_heldout = heldout
        result.total_seconds = round(
            sum(s.audio.duration_s for s in train + val), 2
        )

        self._write_manifest(output_dir, manifest, dataset, result)
        return result

    @abc.abstractmethod
    def write(
        self,
        paths: VoicePackPaths,
        manifest: VoicePackManifest,
        output_dir: Path,
        train: list[TrainingSample],
        val: list[TrainingSample],
        speaker: str,
    ) -> ExportResult:
        """Write the engine-specific layout. Filtering has already happened."""

    # -- helpers -------------------------------------------------------------------

    @staticmethod
    def copy_audio(paths: VoicePackPaths, sample: TrainingSample, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.root / sample.audio_path, destination)
        return destination

    def _write_manifest(
        self,
        output_dir: Path,
        manifest: VoicePackManifest,
        dataset: DatasetManifest,
        result: ExportResult,
    ) -> None:
        """Record what produced this export, so a training run is traceable (spec §23)."""
        payload = {
            "engine": self.engine,
            "exporter_version": EXPORTER_VERSION,
            "exported_at": utcnow().isoformat(),
            "voicepack_id": manifest.voicepack_id,
            "voicepack_version": manifest.version,
            "dataset_created_at": dataset.created_at,
            "train": result.train_count,
            "val": result.val_count,
            "excluded_heldout": result.excluded_heldout,
            "total_seconds": result.total_seconds,
            "minutes_by_style": dataset.minutes_by_style(),
            "target_sample_rate": self.target_sample_rate,
        }
        (output_dir / "export.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


# --------------------------------------------------------------------------------------
# GPT-SoVITS
# --------------------------------------------------------------------------------------


class GPTSoVITSExporter(DatasetExporter):
    """``audio_path|speaker_name|language|text``, one line per sample.

    The engine's own WebUI produces this from its slicer and ASR steps; exporting it
    directly lets us keep our reviewed transcripts and style labels instead of
    re-deriving them with its bundled tools.
    """

    engine = "gpt_sovits"

    def write(self, paths, manifest, output_dir, train, val, speaker) -> ExportResult:
        audio_dir = output_dir / "audio"
        lines: list[str] = []
        for sample in train + val:
            destination = audio_dir / f"{sample.sample_id}.wav"
            self.copy_audio(paths, sample, destination)
            text = _one_line(sample.transcript)
            lines.append(f"{destination.resolve()}|{speaker}|{GPT_SOVITS_LANG}|{text}")

        list_file = output_dir / f"{speaker}.list"
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        notes = [
            "Train the SoVITS and GPT stages separately; v4 for 48 kHz output, "
            "v2Pro if the source audio is rough.",
            "Register the resulting weights in the voice pack manifest as a "
            "CheckpointRef with checkpoint_id '<gpt_weights>|<sovits_weights>'.",
        ]
        return ExportResult(
            engine=self.engine,
            root=str(output_dir),
            files=[str(list_file), str(audio_dir)],
            notes=notes,
        )


# --------------------------------------------------------------------------------------
# Fish Speech
# --------------------------------------------------------------------------------------


class FishSpeechExporter(DatasetExporter):
    """``SPK/clip.wav`` + ``SPK/clip.lab``, the layout ``extract_vq.py`` expects."""

    engine = "fish_speech"

    def write(self, paths, manifest, output_dir, train, val, speaker) -> ExportResult:
        speaker_dir = output_dir / "data" / speaker
        for sample in train + val:
            audio = speaker_dir / f"{sample.sample_id}.wav"
            self.copy_audio(paths, sample, audio)
            audio.with_suffix(".lab").write_text(
                _one_line(sample.transcript) + "\n", encoding="utf-8"
            )

        notes = [
            "Next: tools/vqgan/extract_vq.py → tools/llama/build_dataset.py → "
            "train.py --config-name text2semantic_finetune with a LoRA config → "
            "tools/llama/merge_lora.py.",
            "LoRA only. Upstream advises against fine-tuning an RL-trained checkpoint.",
            "Confirm the weight licence before any commercial use of the result.",
        ]
        return ExportResult(
            engine=self.engine,
            root=str(output_dir),
            files=[str(speaker_dir)],
            notes=notes,
        )


# --------------------------------------------------------------------------------------
# Qwen3-TTS
# --------------------------------------------------------------------------------------


class QwenTTSExporter(DatasetExporter):
    """``train_raw.jsonl`` / ``val_raw.jsonl``, plus a reference clip.

    The official fine-tuning flow transcribes with WhisperX to build this file. Ours is
    already transcribed and human-reviewed, so the export skips that step — which also
    avoids a second transcription disagreeing with the one in the dataset.
    """

    engine = "qwen3_tts"

    def write(self, paths, manifest, output_dir, train, val, speaker) -> ExportResult:
        audio_dir = output_dir / "audio"
        files: list[str] = []

        for name, group in (("train_raw.jsonl", train), ("val_raw.jsonl", val)):
            if not group:
                continue
            rows: list[str] = []
            for sample in group:
                destination = audio_dir / f"{sample.sample_id}.wav"
                self.copy_audio(paths, sample, destination)
                rows.append(
                    json.dumps(
                        {
                            "audio": str(destination.resolve()),
                            "text": _one_line(sample.transcript),
                            "speaker": speaker,
                            "language": "Chinese",
                            "style": sample.voice_style,
                            "duration": round(sample.audio.duration_s, 3),
                        },
                        ensure_ascii=False,
                    )
                )
            path = output_dir / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            files.append(str(path))

        # A single reference clip is needed at inference; pick the best neutral one so
        # the exported bundle is usable on its own.
        reference = _best_reference(train, preferred_style="neutral")
        if reference is not None:
            destination = output_dir / "reference.wav"
            self.copy_audio(paths, reference, destination)
            (output_dir / "reference.txt").write_text(
                _one_line(reference.transcript) + "\n", encoding="utf-8"
            )
            files.append(str(destination))

        notes = [
            "Next: extract audio codes into train_with_codes.jsonl, then run the "
            "official finetuning scripts. ~16 GB VRAM at batch size 2.",
            "Cache create_voice_clone_prompt() per reference clip in the sidecar; "
            "rebuilding it per sentence is this engine's version of reloading weights.",
        ]
        return ExportResult(
            engine=self.engine, root=str(output_dir), files=files, notes=notes
        )


# --------------------------------------------------------------------------------------
# CosyVoice
# --------------------------------------------------------------------------------------


class CosyVoiceExporter(DatasetExporter):
    """Kaldi-style directories, which the CosyVoice training recipes consume."""

    engine = "cosyvoice"

    def write(self, paths, manifest, output_dir, train, val, speaker) -> ExportResult:
        files: list[str] = []
        for split_name, group in (("train", train), ("dev", val)):
            if not group:
                continue
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            audio_dir = output_dir / "audio"

            wav_scp: list[str] = []
            text: list[str] = []
            utt2spk: list[str] = []
            for sample in group:
                destination = audio_dir / f"{sample.sample_id}.wav"
                self.copy_audio(paths, sample, destination)
                utterance = sample.sample_id
                wav_scp.append(f"{utterance} {destination.resolve()}")
                text.append(f"{utterance} {_one_line(sample.transcript)}")
                utt2spk.append(f"{utterance} {speaker}")

            (split_dir / "wav.scp").write_text("\n".join(wav_scp) + "\n", encoding="utf-8")
            (split_dir / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
            (split_dir / "utt2spk").write_text("\n".join(utt2spk) + "\n", encoding="utf-8")
            (split_dir / "spk2utt").write_text(
                f"{speaker} " + " ".join(s.sample_id for s in group) + "\n",
                encoding="utf-8",
            )
            files.append(str(split_dir))

        return ExportResult(
            engine=self.engine,
            root=str(output_dir),
            files=files,
            notes=[
                "CosyVoice can also be benchmarked zero-shot; run that row first, it "
                "costs nothing and sets the bar the fine-tune has to beat."
            ],
        )


# --------------------------------------------------------------------------------------
# VoxCPM
# --------------------------------------------------------------------------------------


class VoxCPMExporter(DatasetExporter):
    """``train.jsonl`` with a prompt clip per row — its SFT/LoRA recipe's shape."""

    engine = "voxcpm"

    def write(self, paths, manifest, output_dir, train, val, speaker) -> ExportResult:
        audio_dir = output_dir / "audio"
        reference = _best_reference(train, preferred_style="neutral")
        reference_path: Path | None = None
        if reference is not None:
            reference_path = self.copy_audio(
                paths, reference, output_dir / "reference.wav"
            )

        files: list[str] = []
        for name, group in (("train.jsonl", train), ("val.jsonl", val)):
            if not group:
                continue
            rows: list[str] = []
            for sample in group:
                destination = audio_dir / f"{sample.sample_id}.wav"
                self.copy_audio(paths, sample, destination)
                row: dict[str, object] = {
                    "audio": str(destination.resolve()),
                    "text": _one_line(sample.transcript),
                    "speaker": speaker,
                    "style": sample.voice_style,
                }
                if reference_path is not None and reference is not None:
                    row["prompt_audio"] = str(reference_path.resolve())
                    row["prompt_text"] = _one_line(reference.transcript)
                rows.append(json.dumps(row, ensure_ascii=False))
            path = output_dir / name
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            files.append(str(path))

        return ExportResult(
            engine=self.engine,
            root=str(output_dir),
            files=files,
            notes=[
                "LoRA works from 5-10 minutes, which makes this the fallback if the "
                "pack comes in under the 20-minute target.",
                "48 kHz native output — do not resample the export down.",
            ],
        )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

EXPORTERS: dict[str, type[DatasetExporter]] = {
    "gpt_sovits": GPTSoVITSExporter,
    "fish_speech": FishSpeechExporter,
    "qwen3_tts": QwenTTSExporter,
    "cosyvoice": CosyVoiceExporter,
    "voxcpm": VoxCPMExporter,
}


def get_exporter(engine: str) -> DatasetExporter:
    try:
        return EXPORTERS[engine]()
    except KeyError as exc:
        # IndexTTS is absent on purpose, and saying why is more useful than a bare
        # KeyError: it has no documented fine-tuning path, so it competes zero-shot.
        extra = (
            " (index_tts has no documented fine-tuning path; it competes zero-shot)"
            if engine == "index_tts"
            else ""
        )
        raise KeyError(
            f"no exporter for engine {engine!r}{extra}; known: {sorted(EXPORTERS)}"
        ) from exc


def export_dataset(
    engine: str,
    paths: VoicePackPaths,
    manifest: VoicePackManifest,
    dataset: DatasetManifest,
    output_dir: Path,
    **kwargs: object,
) -> ExportResult:
    return get_exporter(engine).export(
        paths, manifest, dataset, output_dir, **kwargs  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    """Collapse whitespace. A newline in a ``|``-delimited list file corrupts it."""
    return " ".join(text.split())


def _best_reference(
    samples: list[TrainingSample], *, preferred_style: str = "neutral"
) -> TrainingSample | None:
    """Pick a clip to use as the inference prompt in the exported bundle."""
    pool = [s for s in samples if s.voice_style == preferred_style] or samples
    in_window = [s for s in pool if 3.0 <= s.audio.duration_s <= 10.0] or pool
    return max(in_window, key=lambda s: (s.quality_score, s.sample_id), default=None)
