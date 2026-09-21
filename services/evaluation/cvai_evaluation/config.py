"""Benchmark configuration (spec §17).

One YAML file defines an experiment: which voice pack, which sentence set, which
candidates, how many repeats, which seed. Everything needed to re-run it later is in that
file plus the recorded commit — which is the whole point of spec §23.
"""

from __future__ import annotations

from pathlib import Path

from cvai_core.errors import ConfigError
from cvai_core.loaders import load_model_yaml
from cvai_types import BenchmarkCandidate, CVAIModel, SemVer, Slug
from pydantic import Field, model_validator


class GroundTruthConfig(CVAIModel):
    """Real character recordings to mix into the blind test as hidden anchors.

    ``auto`` pulls from the dataset's held-out split, which is the correct source: those
    lines are real, and no engine was trained on them.
    """

    enabled: bool = True
    source: str = Field(default="heldout", pattern=r"^(heldout|explicit|none)$")
    #: Used when ``source == "explicit"``; pack-relative audio paths.
    sample_ids: list[str] = Field(default_factory=list)
    max_items: int = Field(default=6, ge=0, le=100)


class BenchmarkConfig(CVAIModel):
    benchmark_id: Slug
    description: str = Field(default="", max_length=1000)
    voicepack_id: Slug
    #: Pinned so a result always states which pack version produced it. ``None`` means
    #: "whatever the pack says now", which is convenient during development and wrong
    #: for a recorded experiment.
    voicepack_version: SemVer | None = None
    #: Path to the sentence-set YAML, relative to the repository root.
    sentence_set: str
    candidates: list[BenchmarkCandidate] = Field(default_factory=list)

    repeats: int = Field(default=1, ge=1, le=10)
    base_seed: int = Field(default=20260921, ge=0)
    output_sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    skip_unavailable: bool = True
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)

    #: Reference-clip rotation. Off makes every candidate use the single best clip per
    #: style, which is a cleaner A/B of the engines; on is closer to production. Default
    #: on, because "one reference clip for everything" is the failure mode being guarded
    #: against (spec §27) and the benchmark should exercise the production path.
    rotate_references: bool = True

    @model_validator(mode="after")
    def _at_least_one_enabled(self) -> "BenchmarkConfig":
        if not any(c.enabled for c in self.candidates):
            raise ConfigError(
                f"benchmark {self.benchmark_id!r} has no enabled candidates"
            )
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ConfigError("duplicate candidate_id in benchmark config")
        return self

    def enabled_candidates(self) -> list[BenchmarkCandidate]:
        return [c for c in self.candidates if c.enabled]


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    return load_model_yaml(BenchmarkConfig, path)
