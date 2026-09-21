"""Voice Pack preprocessing pipeline (Milestone 2, spec §7).

Turns a pile of original character audio into an approved, annotated, versioned dataset
plus a Reference Bank, with a human in the loop and a full provenance record.

The governing principle is spec §7's, and most of the defaults here exist to honour it:

    Remove interference that hurts training, but preserve the voice actor's natural
    vocal performance.

So source separation and denoising are off unless the audio genuinely needs them, VAD
padding is generous enough to keep the breath before a line, automatic stages reject only
unambiguous defects, and approval is a human act.

Entry point: ``cvai-prep`` (see :mod:`cvai_voice_preprocessing.cli`).
"""

from __future__ import annotations

from .backends import BackendBundle, resolve_backends
from .config import PreprocessConfig
from .pipeline import BuildError, Pipeline, build_dataset
from .review import ReviewPatch, ReviewPatchEntry, apply_review_patch, write_review_page
from .state import PipelineState, SegmentRecord, SourceClip, Stage, load_state, save_state

__all__ = [
    "BackendBundle",
    "BuildError",
    "Pipeline",
    "PipelineState",
    "PreprocessConfig",
    "ReviewPatch",
    "ReviewPatchEntry",
    "SegmentRecord",
    "SourceClip",
    "Stage",
    "apply_review_patch",
    "build_dataset",
    "load_state",
    "resolve_backends",
    "save_state",
    "write_review_page",
]
