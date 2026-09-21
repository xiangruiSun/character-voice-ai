"""Preprocessing configuration.

Defaults encode the project's cleaning philosophy (spec §7): optional waveform-altering
stages off, padding generous, auto-rejection limited to defects a human would never
overrule, and approval left to a person.
"""

from __future__ import annotations

from cvai_types import CVAIModel, Slug
from pydantic import Field


class PreprocessConfig(CVAIModel):
    voicepack_id: Slug

    #: Directory of original audio to ingest. ``None`` means "use what is already in
    #: ``raw/``", which is the normal case on a re-run.
    source_dir: str | None = None
    #: Process only the first N source clips. For a trial pass over a big dump before
    #: committing hours of ASR.
    limit: int | None = Field(default=None, ge=1)

    target_sample_rate: int = Field(default=44100, ge=16000, le=48000)
    target_lufs: float = Field(default=-20.0, ge=-40.0, le=-6.0)
    true_peak_ceiling_dbfs: float = Field(default=-1.0, le=0.0)

    # --- segmentation ---
    #: Game dialogue is often already one line per file. When false, each source clip
    #: becomes exactly one segment (still trimmed, still padded).
    segment: bool = True
    speech_pad_ms: int = Field(default=180, ge=0, le=1000)
    min_segment_s: float = Field(default=1.0, gt=0.0)
    #: Long segments hurt both training and the later chunker. 20 s is generous for
    #: character dialogue.
    max_segment_s: float = Field(default=20.0, gt=1.0)

    # --- optional cleaning (decision D7: off unless the audio needs it) ---
    enable_separation: bool = False
    enable_denoise: bool = False
    denoise_attenuation_db: float = Field(default=12.0, ge=0.0, le=40.0)
    #: Clips that skip every optional stage, kept as a control set so the effect of
    #: cleaning on this character is measurable rather than assumed.
    control_set_size: int = Field(default=8, ge=0, le=200)

    # --- models ---
    device: str = "cpu"
    asr_model: str = "paraformer-zh"
    emotion_model: str = "iic/emotion2vec_plus_large"
    #: Character and world proper nouns, passed to ASR. Generic Chinese ASR mangles
    #: them, and a mangled name in the transcript teaches the model a wrong pronunciation.
    hotwords: list[str] = Field(default_factory=list)

    # --- filtering ---
    #: Cosine similarity to the character centroid below which a segment is rejected.
    #: Only enforced when a real speaker model is in use; the built-in spectral
    #: embedding is advisory and only ever flags.
    speaker_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    max_clipping_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    min_snr_db: float = Field(default=8.0)

    #: Approve everything that passes the automatic checks. Off by default: spec §8
    #: expects human correction, and an unreviewed dataset is how wrong transcripts and
    #: other characters' voices end up in training.
    auto_approve: bool = False

    # --- dataset build ---
    references_per_style: int = Field(default=4, ge=1, le=20)
    reference_min_s: float = Field(default=3.0, gt=0.0)
    reference_max_s: float = Field(default=10.0, gt=1.0)
    #: Real lines held out of training, used as benchmark ground truth and as the
    #: listening test's hidden anchor.
    heldout_per_style: int = Field(default=2, ge=0, le=50)
    val_fraction: float = Field(default=0.05, ge=0.0, le=0.5)
