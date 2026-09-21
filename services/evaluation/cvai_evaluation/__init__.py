"""Benchmark and evaluation pipeline (spec §17-§19).

The first technical question the project has to answer is which TTS adaptation approach
reproduces the target character most faithfully. This package is the apparatus for
answering it: a runner that generates the same sentences from every candidate under
identical conditions, a blind test assembler, rating aggregation against a real-recording
anchor, and reports.
"""

from __future__ import annotations

from .blind import (
    GROUND_TRUTH_CANDIDATE_ID,
    build_blind_test,
    export_webmushra_config,
    load_ground_truth,
    render_rating_page,
    write_blind_test,
)
from .config import BenchmarkConfig, GroundTruthConfig, load_benchmark_config
from .objective import (
    ClipMeasurement,
    ObjectiveBackends,
    ProsodyComparison,
    ProsodyProfile,
    RunObjectiveReport,
    compare_prosody,
    measure_clip,
    render_objective_report,
    resolve_objective_backends,
    score_run,
)
from .report import render_evaluation_report, render_run_report
from .runner import BenchmarkRunner
from .scoring import (
    ALL_AXES,
    aggregate,
    gap_to_ground_truth,
    load_ratings,
    rater_agreement,
)
from .simulate import simulate_ratings, write_ratings

__all__ = [
    "ALL_AXES",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "ClipMeasurement",
    "ObjectiveBackends",
    "ProsodyComparison",
    "ProsodyProfile",
    "RunObjectiveReport",
    "compare_prosody",
    "measure_clip",
    "render_objective_report",
    "resolve_objective_backends",
    "score_run",
    "GROUND_TRUTH_CANDIDATE_ID",
    "GroundTruthConfig",
    "aggregate",
    "build_blind_test",
    "export_webmushra_config",
    "gap_to_ground_truth",
    "load_benchmark_config",
    "load_ground_truth",
    "load_ratings",
    "rater_agreement",
    "render_evaluation_report",
    "render_rating_page",
    "render_run_report",
    "simulate_ratings",
    "write_blind_test",
    "write_ratings",
]
