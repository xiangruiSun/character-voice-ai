"""Voice Pack schemas — the "how the character says it" half (spec §6, §7, §8, §10).

A Voice Pack is a self-contained, versioned, relocatable directory holding everything
derived from one character's recordings:

    voicepacks/<pack_id>/
        raw/            untouched originals — never written to after ingest
        processed/      intermediate stage outputs, freely regenerable
        clean/          human-approved training audio
        rejected/       filtered-out clips, kept with the reason (auditable)
        transcripts/    ASR output and human corrections
        metadata/       manifests: voicepack.yaml, dataset.json, references.json
        references/     Reference Bank, organised by style
        datasets/       engine-specific exports
        checkpoints/    character-specific fine-tuned models
        evaluation/     benchmark runs and listening-test results

Two rules from the spec are enforced here rather than trusted:

* ``raw/`` is immutable (spec §6) — the validator flags any manifest that points a
  *processed* artifact at a raw path.
* ``processing_chain`` is recorded per clip (decision D7) — because "did our cleaning
  help or hurt?" must be answerable with data, not opinion.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import Field, field_validator, model_validator

from .base import (
    CVAIModel,
    Language,
    RelPath,
    SemVer,
    Slug,
    StyleTag,
    UnitFloat,
    utcnow,
    validate_relative_path,
)
from .style import (
    CoreStyle,
    EmotionVector,
    PauseStyle,
    PitchProfile,
    SpeakingRate,
    StyleDefinition,
    VolumeStyle,
    resolve_style_chain,
)

# --------------------------------------------------------------------------------------
# Directory layout
# --------------------------------------------------------------------------------------

RAW_DIR: Final = "raw"
PROCESSED_DIR: Final = "processed"
CLEAN_DIR: Final = "clean"
REJECTED_DIR: Final = "rejected"
TRANSCRIPTS_DIR: Final = "transcripts"
METADATA_DIR: Final = "metadata"
REFERENCES_DIR: Final = "references"
DATASETS_DIR: Final = "datasets"
CHECKPOINTS_DIR: Final = "checkpoints"
EVALUATION_DIR: Final = "evaluation"

VOICEPACK_DIRS: Final[tuple[str, ...]] = (
    RAW_DIR,
    PROCESSED_DIR,
    CLEAN_DIR,
    REJECTED_DIR,
    TRANSCRIPTS_DIR,
    METADATA_DIR,
    REFERENCES_DIR,
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    EVALUATION_DIR,
)

MANIFEST_FILENAME: Final = "voicepack.yaml"
DATASET_FILENAME: Final = "dataset.json"
REFERENCES_FILENAME: Final = "references.json"


# --------------------------------------------------------------------------------------
# Processing provenance
# --------------------------------------------------------------------------------------


class ProcessingStage(str, Enum):
    """Pipeline stages from spec §7. Optional ones default to *off* (decision D7)."""

    INGEST = "ingest"
    SOURCE_SEPARATION = "source_separation"
    DEREVERB = "dereverb"
    DENOISE = "denoise"
    VAD_SEGMENT = "vad_segment"
    SPEAKER_FILTER = "speaker_filter"
    TRANSCRIBE = "transcribe"
    ALIGN = "align"
    LOUDNESS = "loudness"
    RESAMPLE = "resample"
    ANNOTATE = "annotate"
    QUALITY_SCORE = "quality_score"
    MANUAL_REVIEW = "manual_review"


#: Stages that alter the waveform. Spec §7 warns these can strip character identity, so
#: the validator requires them to be recorded and the benchmark keeps a control set that
#: skipped all of them.
DESTRUCTIVE_STAGES: Final[frozenset[ProcessingStage]] = frozenset(
    {
        ProcessingStage.SOURCE_SEPARATION,
        ProcessingStage.DEREVERB,
        ProcessingStage.DENOISE,
        ProcessingStage.LOUDNESS,
        ProcessingStage.RESAMPLE,
    }
)


class ProcessingStep(CVAIModel):
    """One applied stage, with enough detail to reproduce it."""

    stage: ProcessingStage
    tool: str = Field(min_length=1, max_length=120, description="e.g. 'audio-separator'")
    tool_version: str = Field(default="unknown", max_length=60)
    #: Model or preset name, e.g. ``model_bs_roformer_ep_317_sdr_12.9755``.
    model: str | None = Field(default=None, max_length=200)
    params: dict[str, object] = Field(default_factory=dict)

    def is_destructive(self) -> bool:
        return self.stage in DESTRUCTIVE_STAGES


class AudioProperties(CVAIModel):
    sample_rate: int = Field(ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    duration_s: float = Field(gt=0.0, le=3600.0)
    peak_dbfs: float | None = Field(default=None, le=0.0, ge=-120.0)
    lufs: float | None = Field(default=None, ge=-70.0, le=0.0)
    #: Estimated SNR in dB where available. Used for review ordering, not rejection.
    snr_db: float | None = None


# --------------------------------------------------------------------------------------
# Training samples
# --------------------------------------------------------------------------------------


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    #: Transcript or labels were corrected by a human — spec §8 requires this to be
    #: possible and it is worth distinguishing from a clip that passed untouched.
    CORRECTED = "corrected"


class RejectionReason(str, Enum):
    OTHER_SPEAKER = "other_speaker"
    BACKGROUND_MUSIC = "background_music"
    OVERLAPPING_SPEECH = "overlapping_speech"
    CLIPPED_AUDIO = "clipped_audio"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    BAD_TRANSCRIPT = "bad_transcript"
    NON_SPEECH = "non_speech"
    LOW_QUALITY = "low_quality"
    DUPLICATE = "duplicate"
    OTHER = "other"


class TrainingSample(CVAIModel):
    """One approved utterance (spec §8).

    Not every field is expected to be automatically correct in V1; the point is that each
    has a home and a human can fix it. ``review_status`` records whether a human ever
    looked.
    """

    sample_id: str = Field(min_length=1, max_length=120)
    audio_path: RelPath
    transcript: str = Field(min_length=1, max_length=1000)
    language: Language = "zh-CN"
    audio: AudioProperties

    # --- performance annotation (spec §8) ---
    emotion: StyleTag = "neutral"
    emotion_intensity: UnitFloat = 0.5
    speaking_rate: SpeakingRate = SpeakingRate.NORMAL
    pitch_profile: PitchProfile = PitchProfile.MID
    pause_style: PauseStyle = PauseStyle.NATURAL
    volume_style: VolumeStyle = VolumeStyle.NORMAL
    voice_style: StyleTag = "neutral"
    emotion_vector: EmotionVector | None = None

    # --- measured signals, filled by the annotate/quality stages ---
    #: Characters per second. The character's real distribution of this is what the
    #: benchmark compares generated speech against.
    chars_per_second: float | None = Field(default=None, gt=0.0, le=20.0)
    f0_mean_hz: float | None = Field(default=None, gt=0.0, le=1000.0)
    f0_std_hz: float | None = Field(default=None, ge=0.0, le=500.0)
    pause_count: int | None = Field(default=None, ge=0)

    # --- quality and provenance ---
    quality_score: UnitFloat = 1.0
    #: Cosine similarity to the character's speaker centroid. The defence against the
    #: "training on other characters" failure mode (spec §27).
    speaker_similarity: UnitFloat | None = None
    source_clip: RelPath | None = None
    source_offset_s: float | None = Field(default=None, ge=0.0)
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    processing_chain: list[ProcessingStep] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    rejection_reason: RejectionReason | None = None
    notes: str = Field(default="", max_length=1000)

    @field_validator("audio_path", "source_clip")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        return None if value is None else validate_relative_path(value)

    @model_validator(mode="after")
    def _consistency(self) -> "TrainingSample":
        if self.review_status is ReviewStatus.REJECTED and self.rejection_reason is None:
            raise ValueError("a rejected sample must record a rejection_reason")
        if self.review_status is not ReviewStatus.REJECTED and self.rejection_reason:
            raise ValueError("rejection_reason is only valid on a rejected sample")
        # Approved training audio must not come straight out of raw/ — raw is immutable
        # and unsegmented; anything approved has at minimum been segmented.
        if self.audio_path.startswith(f"{RAW_DIR}/"):
            raise ValueError(
                "training samples must not point into raw/; raw audio is immutable input"
            )
        return self

    @property
    def is_usable(self) -> bool:
        return self.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED)

    @property
    def was_destructively_processed(self) -> bool:
        return any(step.is_destructive() for step in self.processing_chain)


# --------------------------------------------------------------------------------------
# Reference Bank (spec §10)
# --------------------------------------------------------------------------------------


class ReferenceSample(CVAIModel):
    """One reference utterance for prompt-based synthesis.

    ``transcript`` is required, not optional (decision D3): Qwen3-TTS needs ``ref_text``,
    Fish Speech needs ``--prompt-text``, VoxCPM2 needs the transcript for its highest
    cloning tier, and GPT-SoVITS improves markedly with ``prompt_text``. A reference bank
    without transcripts has to be rebuilt by hand.
    """

    reference_id: str = Field(min_length=1, max_length=120)
    audio_path: RelPath
    transcript: str = Field(min_length=1, max_length=500)
    language: Language = "zh-CN"
    style: StyleTag
    core_style: CoreStyle
    audio: AudioProperties
    quality_score: UnitFloat = 1.0
    #: Where this clip came from before any cleaning, and at what offset. Carried so the
    #: benchmark can rebuild the *unprocessed* version of exactly this reference and run
    #: it as a control (decision D7). Without the provenance, "did our cleaning help or
    #: did it sand her voice down?" is unanswerable after the fact.
    source_clip: RelPath | None = None
    source_offset_s: float | None = Field(default=None, ge=0.0)
    #: Free tags for finer retrieval later, e.g. ``["short", "question", "sentence_final_particle"]``.
    tags: list[str] = Field(default_factory=list)
    #: Engine-specific precomputed artifacts, keyed by engine id — e.g. Fish Speech VQ
    #: prompt tokens (``fake.npy``) extracted once at pack build time rather than per
    #: request. Values are pack-relative paths.
    precomputed: dict[str, RelPath] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=500)

    @field_validator("audio_path", "source_clip")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        return None if value is None else validate_relative_path(value)

    @model_validator(mode="after")
    def _duration_sanity(self) -> "ReferenceSample":
        # Most engines take a short prompt; very long references waste context and very
        # short ones carry too little prosody. Warn-by-validation at the extremes only.
        if self.audio.duration_s < 1.0:
            raise ValueError(
                f"reference {self.reference_id!r} is {self.audio.duration_s:.2f}s; "
                "references under 1s do not carry usable prosody"
            )
        if self.audio.duration_s > 30.0:
            raise ValueError(
                f"reference {self.reference_id!r} is {self.audio.duration_s:.1f}s; "
                "keep references under 30s (engines truncate, and long prompts drift)"
            )
        return self


class ReferenceBank(CVAIModel):
    """All reference clips for one character, indexed by style (spec §10).

    The bank exists specifically to prevent the "one reference clip for every emotion"
    failure mode in spec §27.
    """

    voicepack_id: Slug
    voicepack_version: SemVer
    samples: list[ReferenceSample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ReferenceBank":
        seen: set[str] = set()
        for sample in self.samples:
            if sample.reference_id in seen:
                raise ValueError(f"duplicate reference_id {sample.reference_id!r}")
            seen.add(sample.reference_id)
        return self

    def by_style(self, style: str) -> list[ReferenceSample]:
        """Exact-style matches, best quality first."""
        matches = [s for s in self.samples if s.style == style]
        return sorted(matches, key=lambda s: (-s.quality_score, s.reference_id))

    def styles(self) -> list[str]:
        return sorted({s.style for s in self.samples})

    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.style] = counts.get(sample.style, 0) + 1
        return dict(sorted(counts.items()))

    def resolve(
        self,
        style: str,
        style_definitions: dict[str, StyleDefinition] | None = None,
    ) -> list[ReferenceSample]:
        """Walk the fallback chain until some clips are found."""
        for candidate in resolve_style_chain(style, style_definitions or {}):
            found = self.by_style(candidate)
            if found:
                return found
        # Last resort: anything at all, best quality first. Better than failing a live
        # conversation turn, and the retriever records that a fallback happened.
        return sorted(self.samples, key=lambda s: (-s.quality_score, s.reference_id))


# --------------------------------------------------------------------------------------
# Dataset manifest
# --------------------------------------------------------------------------------------


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    #: Held out from training and used as ground truth in the benchmark. Never exported
    #: to an engine's training data.
    HELDOUT = "heldout"


class DatasetEntry(CVAIModel):
    sample_id: str
    split: DatasetSplit = DatasetSplit.TRAIN


class DatasetManifest(CVAIModel):
    """The approved dataset for one voice pack version."""

    voicepack_id: Slug
    voicepack_version: SemVer
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    samples: list[TrainingSample] = Field(default_factory=list)
    splits: list[DatasetEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_align(self) -> "DatasetManifest":
        sample_ids = {s.sample_id for s in self.samples}
        if len(sample_ids) != len(self.samples):
            raise ValueError("duplicate sample_id in dataset manifest")
        unknown = {e.sample_id for e in self.splits} - sample_ids
        if unknown:
            raise ValueError(f"splits reference unknown sample_ids: {sorted(unknown)}")
        return self

    def usable(self) -> list[TrainingSample]:
        return [s for s in self.samples if s.is_usable]

    def total_seconds(self, only_usable: bool = True) -> float:
        pool = self.usable() if only_usable else self.samples
        return sum(s.audio.duration_s for s in pool)

    def minutes_by_style(self, only_usable: bool = True) -> dict[str, float]:
        """Minutes of audio per ``voice_style``.

        Milestone 2's gate: the spec asks for 20-60 minutes overall, but a pack with 40
        minutes that is 95% neutral cannot perform a character. This is the number that
        says so.
        """
        pool = self.usable() if only_usable else self.samples
        totals: dict[str, float] = {}
        for sample in pool:
            totals[sample.voice_style] = (
                totals.get(sample.voice_style, 0.0) + sample.audio.duration_s / 60.0
            )
        return {k: round(v, 3) for k, v in sorted(totals.items())}

    def split_of(self, sample_id: str) -> DatasetSplit:
        for entry in self.splits:
            if entry.sample_id == sample_id:
                return entry.split
        return DatasetSplit.TRAIN


# --------------------------------------------------------------------------------------
# Voice pack manifest
# --------------------------------------------------------------------------------------


class SourceInfo(CVAIModel):
    """Where the audio came from. Recorded for licensing and reproducibility."""

    description: str = Field(default="", max_length=500)
    game_or_media: str = Field(default="", max_length=200)
    extraction_tool: str = Field(default="", max_length=200)
    #: Free-text note on usage rights. Not legal advice, but the place a human records
    #: what was checked before training on this audio.
    license_note: str = Field(default="", max_length=1000)
    consent_confirmed: bool = False


class CheckpointRef(CVAIModel):
    """A character-specific trained model stored in ``checkpoints/``."""

    checkpoint_id: str = Field(min_length=1, max_length=120)
    engine: str = Field(min_length=1, max_length=60)
    engine_version: str = Field(default="unknown", max_length=60)
    adaptation_mode: str = Field(default="finetuned", max_length=30)
    #: One or more pack-relative paths — GPT-SoVITS needs both a GPT and a SoVITS weight.
    paths: dict[str, RelPath] = Field(default_factory=dict)
    trained_on_dataset_version: SemVer | None = None
    training_config: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    notes: str = Field(default="", max_length=1000)

    @field_validator("paths")
    @classmethod
    def _relative(cls, value: dict[str, str]) -> dict[str, str]:
        return {k: validate_relative_path(v) for k, v in value.items()}


class VoicePackManifest(CVAIModel):
    """``voicepacks/<id>/metadata/voicepack.yaml``."""

    voicepack_id: Slug
    version: SemVer = "0.1.0"
    character_id: Slug
    display_name: str = Field(min_length=1, max_length=100)
    language: Language = "zh-CN"
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    source: SourceInfo = Field(default_factory=SourceInfo)

    #: Styles this pack can perform. The retriever, planner and validator all read this.
    styles: list[StyleDefinition] = Field(default_factory=list)
    checkpoints: list[CheckpointRef] = Field(default_factory=list)

    #: Target audio format for everything under ``clean/``.
    target_sample_rate: int = Field(default=44100, ge=16000, le=48000)
    target_lufs: float = Field(default=-20.0, ge=-40.0, le=-6.0)

    #: Default processing policy. All optional stages off (decision D7); the preprocessing
    #: pipeline turns them on per-source when the audio actually needs them.
    enable_source_separation: bool = False
    enable_dereverb: bool = False
    enable_denoise: bool = False

    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _style_names_unique(self) -> "VoicePackManifest":
        names = [s.name for s in self.styles]
        if len(names) != len(set(names)):
            raise ValueError("duplicate style name in voice pack manifest")
        return self

    def style_map(self) -> dict[str, StyleDefinition]:
        return {s.name: s for s in self.styles}

    def has_style(self, style: str) -> bool:
        return any(s.name == style for s in self.styles)

    def checkpoint_for(self, engine: str) -> CheckpointRef | None:
        for checkpoint in self.checkpoints:
            if checkpoint.engine == engine:
                return checkpoint
        return None
