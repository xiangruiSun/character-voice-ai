"""Rating aggregation (spec §18).

Two things this module refuses to do quietly:

* **Treat every axis as "higher is better".** ``ai_artifact_level`` asks how obvious it
  is that a clip is generated, so low is good. The direction lives in
  ``AXIS_HIGHER_IS_BETTER`` and is applied when composing a ranking score.
* **Report a winner without the ground-truth anchor.** Real character recordings are in
  the same blind test; if a candidate's naturalness is 3.1 and the real recordings score
  3.2, the raters were not discriminating and the run says nothing. The anchor row is
  always in the report.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from cvai_types import (
    AXIS_HIGHER_IS_BETTER,
    AggregateReport,
    BenchmarkRun,
    BlindKey,
    CandidateAggregate,
    HumanRating,
    RatingAxis,
    aggregate_axis,
)

from .blind import GROUND_TRUTH_CANDIDATE_ID

ALL_AXES = (
    RatingAxis.SPEAKER_SIMILARITY,
    RatingAxis.NATURALNESS,
    RatingAxis.CHARACTER_SIMILARITY,
    RatingAxis.AI_ARTIFACT_LEVEL,
)


def load_ratings(path: Path) -> list[HumanRating]:
    """Load one or many rating files.

    Accepts a single exported file or a directory of them, because in practice each
    rater hands back their own download.
    """
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    ratings: list[HumanRating] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("ratings", [])
        for entry in data:
            ratings.append(HumanRating.model_validate(entry))
    return ratings


def aggregate(
    run: BenchmarkRun,
    key: BlindKey,
    ratings: list[HumanRating],
) -> AggregateReport:
    """Fold ratings into per-candidate aggregates."""
    item_to_candidate = {e.item_id: e.candidate_id for e in key.entries}
    ground_truth_items = {e.item_id for e in key.entries if e.is_ground_truth}
    labels = {c.candidate_id: c.label for c in run.candidates}

    per_candidate: dict[str, list[HumanRating]] = defaultdict(list)
    items_seen: dict[str, set[str]] = defaultdict(set)
    orphans = 0
    for rating in ratings:
        candidate_id = item_to_candidate.get(rating.item_id)
        if candidate_id is None:
            orphans += 1
            continue
        per_candidate[candidate_id].append(rating)
        items_seen[candidate_id].add(rating.item_id)

    aggregates: list[CandidateAggregate] = []
    for candidate_id, group in per_candidate.items():
        axes = [
            aggregate_axis([r.score(axis) for r in group], axis) for axis in ALL_AXES
        ]
        believed = [r.believed_real for r in group if r.believed_real is not None]
        aggregates.append(
            CandidateAggregate(
                candidate_id=candidate_id,
                label=labels.get(candidate_id, candidate_id),
                is_ground_truth=candidate_id == GROUND_TRUTH_CANDIDATE_ID
                or all(r.item_id in ground_truth_items for r in group),
                n_ratings=len(group),
                n_items=len(items_seen[candidate_id]),
                axes=axes,
                believed_real_rate=(
                    round(sum(1 for b in believed if b) / len(believed), 4)
                    if believed
                    else None
                ),
            )
        )

    notes = []
    if orphans:
        notes.append(
            f"{orphans} ratings referenced item ids that are not in this test's key "
            "and were ignored — check that the ratings and the key come from the same run"
        )
    if GROUND_TRUTH_CANDIDATE_ID not in per_candidate:
        notes.append(
            "no ground-truth anchor was rated; scores cannot be calibrated against real "
            "recordings and should not be used to pick a winner"
        )

    return AggregateReport(
        run_id=run.run_id,
        test_id=key.test_id,
        candidates=aggregates,
        n_raters=len({r.rater_id for r in ratings}),
        notes=" | ".join(notes),
    )


def rater_agreement(ratings: list[HumanRating], axis: RatingAxis) -> float | None:
    """Mean pairwise Pearson correlation between raters on commonly rated items.

    A crude but honest check. If raters do not agree, no amount of averaging makes the
    ranking meaningful, and the right response is clearer instructions or better anchors
    — not a bigger sample.
    """
    by_rater: dict[str, dict[str, int]] = defaultdict(dict)
    for rating in ratings:
        by_rater[rating.rater_id][rating.item_id] = rating.score(axis)

    rater_ids = sorted(by_rater)
    if len(rater_ids) < 2:
        return None

    correlations: list[float] = []
    for i, left in enumerate(rater_ids):
        for right in rater_ids[i + 1 :]:
            shared = sorted(set(by_rater[left]) & set(by_rater[right]))
            if len(shared) < 3:
                continue
            xs = [by_rater[left][item] for item in shared]
            ys = [by_rater[right][item] for item in shared]
            correlation = _pearson(xs, ys)
            if correlation is not None:
                correlations.append(correlation)
    return round(statistics.fmean(correlations), 4) if correlations else None


def _pearson(xs: list[int], ys: list[int]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        # A rater who gave every item the same score carries no information here.
        return None
    return cov / ((var_x**0.5) * (var_y**0.5))


def gap_to_ground_truth(report: AggregateReport, axis: RatingAxis) -> dict[str, float]:
    """Each candidate's distance from the real recordings on one axis.

    This, not the raw mean, is the number spec §18's success condition is about: how
    close a generated line got to a new recording from the original character.
    """
    anchor = next((c for c in report.candidates if c.is_ground_truth), None)
    if anchor is None:
        return {}
    anchor_mean = anchor.axis_mean(axis)
    if anchor_mean is None:
        return {}
    higher_better = AXIS_HIGHER_IS_BETTER[axis]
    gaps: dict[str, float] = {}
    for candidate in report.candidates:
        if candidate.is_ground_truth:
            continue
        value = candidate.axis_mean(axis)
        if value is None:
            continue
        delta = value - anchor_mean
        gaps[candidate.candidate_id] = round(delta if higher_better else -delta, 4)
    return gaps
