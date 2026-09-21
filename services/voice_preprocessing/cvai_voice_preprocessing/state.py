"""Pipeline state — the work-in-progress record between ``raw/`` and ``clean/``.

Held in one JSON file (``processed/state.json``) rather than inferred from what happens
to be on disk. Three reasons, all learned the expensive way in data pipelines:

* **Resumability.** ASR over an hour of audio is slow. A pipeline that cannot resume
  gets run less often, and a pipeline that is run less often stops matching the data.
* **Provenance.** Each segment accumulates the stages that touched it, so "why does this
  clip sound over-processed?" is answerable from the record.
* **Human edits survive re-runs.** A reviewer's corrected transcript and style label live
  in the state and are never overwritten by a later automatic pass.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from cvai_types import (
    CVAIModel,
    ProcessingStep,
    RejectionReason,
    ReviewStatus,
    Slug,
    StyleTag,
    UnitFloat,
    utcnow,
)
from pydantic import Field

#: Subdirectories under ``processed/``, numbered so the order on disk matches the order
#: of operations — which is the first thing anyone wants when debugging a pack.
DECODED_DIR = "01_decoded"
SEPARATED_DIR = "02_separated"
DENOISED_DIR = "03_denoised"
SEGMENTS_DIR = "04_segments"
STATE_FILENAME = "state.json"


class Stage(str, Enum):
    INGEST = "ingest"
    DECODE = "decode"
    SEPARATE = "separate"
    DENOISE = "denoise"
    SEGMENT = "segment"
    TRANSCRIBE = "transcribe"
    SPEAKER_FILTER = "speaker_filter"
    ANNOTATE = "annotate"
    QUALITY = "quality"
    REVIEW = "review"
    BUILD = "build"


#: The default order. ``separate`` and ``denoise`` are in the list but skipped unless the
#: voice pack manifest turns them on (decision D7).
DEFAULT_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.INGEST,
    Stage.DECODE,
    Stage.SEPARATE,
    Stage.DENOISE,
    Stage.SEGMENT,
    Stage.TRANSCRIBE,
    Stage.SPEAKER_FILTER,
    Stage.ANNOTATE,
    Stage.QUALITY,
)


class SourceClip(CVAIModel):
    """One original file. Immutable after ingest."""

    clip_id: str = Field(min_length=1, max_length=160)
    #: Pack-relative path under ``raw/``.
    raw_path: str
    original_name: str = ""
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sample_rate: int | None = None
    channels: int | None = None
    duration_s: float | None = None
    #: Pack-relative path to the decoded/processed working copy currently in use.
    working_path: str | None = None
    processing_chain: list[ProcessingStep] = Field(default_factory=list)
    error: str | None = None
    #: Free metadata carried from the source, e.g. the original Wwise event name. Game
    #: voice filenames often encode speaker and line id, which is free labelling.
    source_metadata: dict[str, str] = Field(default_factory=dict)


class SegmentRecord(CVAIModel):
    """One candidate utterance, accumulating measurements as stages run."""

    segment_id: str = Field(min_length=1, max_length=160)
    clip_id: str
    #: Pack-relative path to this segment's audio (under ``processed/04_segments``).
    audio_path: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)
    sample_rate: int

    # --- transcription ---
    transcript: str | None = None
    transcript_source: str | None = None  # asr model id, "human", or "stub"
    transcript_confidence: UnitFloat | None = None

    # --- speaker identity ---
    speaker_similarity: float | None = None
    speaker_cluster: int | None = None

    # --- performance annotation ---
    emotion_auto: str | None = None
    emotion_scores: dict[str, float] = Field(default_factory=dict)
    style: StyleTag | None = None
    chars_per_second: float | None = None
    f0_mean_hz: float | None = None
    f0_std_hz: float | None = None
    pause_count: int | None = None

    # --- quality ---
    lufs: float | None = None
    peak_dbfs: float | None = None
    snr_db: float | None = None
    clipping_ratio: float | None = None
    quality_score: UnitFloat | None = None
    quality_notes: list[str] = Field(default_factory=list)

    # --- review ---
    review_status: ReviewStatus = ReviewStatus.PENDING
    rejection_reason: RejectionReason | None = None
    reviewer_notes: str = Field(default="", max_length=1000)
    #: Set once a human has touched this segment. Automatic stages must not overwrite
    #: ``transcript`` or ``style`` when this is true.
    human_edited: bool = False

    processing_chain: list[ProcessingStep] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED)

    def add_step(self, step: ProcessingStep) -> None:
        self.processing_chain.append(step)


class StageRun(CVAIModel):
    stage: Stage
    started_at: str
    finished_at: str | None = None
    backend: str = ""
    items: int = 0
    skipped: bool = False
    skip_reason: str = ""
    error: str | None = None


class PipelineState(CVAIModel):
    """Everything the preprocessing pipeline knows about one voice pack."""

    voicepack_id: Slug
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: utcnow().isoformat())
    sources: list[SourceClip] = Field(default_factory=list)
    segments: list[SegmentRecord] = Field(default_factory=list)
    runs: list[StageRun] = Field(default_factory=list)
    #: Reference clips a human picked out as definitely this character. The speaker
    #: filter builds its centroid from these; without them it has nothing to compare to.
    speaker_anchor_segment_ids: list[str] = Field(default_factory=list)

    # -- lookups -------------------------------------------------------------------

    def source(self, clip_id: str) -> SourceClip | None:
        return next((s for s in self.sources if s.clip_id == clip_id), None)

    def segment(self, segment_id: str) -> SegmentRecord | None:
        return next((s for s in self.segments if s.segment_id == segment_id), None)

    def segments_of(self, clip_id: str) -> list[SegmentRecord]:
        return [s for s in self.segments if s.clip_id == clip_id]

    def usable_segments(self) -> list[SegmentRecord]:
        return [s for s in self.segments if s.is_usable]

    def completed_stages(self) -> set[Stage]:
        return {r.stage for r in self.runs if r.finished_at and not r.error}

    def last_run(self, stage: Stage) -> StageRun | None:
        matching = [r for r in self.runs if r.stage is stage]
        return matching[-1] if matching else None

    # -- summary -------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        usable = self.usable_segments()
        pending = [s for s in self.segments if s.review_status is ReviewStatus.PENDING]
        rejected = [s for s in self.segments if s.review_status is ReviewStatus.REJECTED]
        minutes = sum(s.duration_s for s in usable) / 60.0
        by_style: dict[str, float] = {}
        for segment in usable:
            key = segment.style or segment.emotion_auto or "unassigned"
            by_style[key] = by_style.get(key, 0.0) + segment.duration_s / 60.0
        return {
            "sources": len(self.sources),
            "segments": len(self.segments),
            "approved": len(usable),
            "pending_review": len(pending),
            "rejected": len(rejected),
            "approved_minutes": round(minutes, 2),
            "minutes_by_style": {k: round(v, 2) for k, v in sorted(by_style.items())},
            "stages_completed": sorted(s.value for s in self.completed_stages()),
            "transcribed": sum(1 for s in self.segments if s.transcript),
            "human_edited": sum(1 for s in self.segments if s.human_edited),
        }


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def state_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / STATE_FILENAME


def load_state(processed_dir: Path, voicepack_id: str) -> PipelineState:
    """Load existing state, or start a new one."""
    path = state_path(processed_dir)
    if not path.is_file():
        return PipelineState(voicepack_id=voicepack_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    state = PipelineState.model_validate(data)
    if state.voicepack_id != voicepack_id:
        raise ValueError(
            f"{path} holds state for {state.voicepack_id!r}, not {voicepack_id!r}"
        )
    return state


def save_state(state: PipelineState, processed_dir: Path) -> Path:
    """Write atomically: a truncated state file loses hours of ASR."""
    path = state_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = utcnow().isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
