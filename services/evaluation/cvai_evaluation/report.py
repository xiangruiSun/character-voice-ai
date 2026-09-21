"""Markdown reporting for a benchmark run.

Two reports, because they answer different questions at different times:

* the **run report**, available the moment generation finishes, which says whether the
  experiment itself is sound — did every candidate produce audio, did reference styles
  resolve, were controls dropped, how fast was each engine;
* the **evaluation report**, after humans have rated, which says which candidate won and
  by how much against the real-recording anchor.

Keeping them separate matters: a run whose fallback rate is 60% should be fixed and
re-run, not listened to, and that is visible before anyone spends an hour rating.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from cvai_types import AggregateReport, BenchmarkRun, RatingAxis

from .scoring import ALL_AXES, gap_to_ground_truth

_AXIS_LABEL = {
    RatingAxis.SPEAKER_SIMILARITY: "Speaker sim.",
    RatingAxis.NATURALNESS: "Naturalness",
    RatingAxis.CHARACTER_SIMILARITY: "Character sim.",
    RatingAxis.AI_ARTIFACT_LEVEL: "AI artifacts (low=good)",
}


def render_run_report(run: BenchmarkRun) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark run `{run.run_id}`")
    lines.append("")
    lines.append(f"- Voice pack: `{run.voicepack_id}` v{run.voicepack_version}")
    lines.append(f"- Sentence set: `{run.sentence_set_id}`")
    lines.append(f"- Base seed: `{run.base_seed}`")
    lines.append(f"- Config hash: `{run.config_hash or 'n/a'}`")
    lines.append(f"- Git commit: `{run.git_commit or 'n/a'}`")
    lines.append(f"- Created: {run.created_at}")
    lines.append("")

    succeeded = run.succeeded_records()
    failed = run.failures()
    lines.append(
        f"**{len(succeeded)} generated, {len(failed)} failed, "
        f"reference fallback rate {run.fallback_rate():.1%}.**"
    )
    if run.fallback_rate() > 0.25:
        lines.append("")
        lines.append(
            "> The reference bank could not serve the requested style for more than a "
            "quarter of the lines. Every candidate was therefore partly evaluated on "
            "fallback styles, which flattens the differences this benchmark exists to "
            "measure. Add reference clips for the missing styles before rating."
        )
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    lines.append(
        "| Candidate | Engine | Adaptation | Checkpoint | OK | Failed | "
        "Median duration | Median latency | RTF | Licence |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")

    by_candidate: dict[str, list] = defaultdict(list)
    for record in run.records:
        by_candidate[record.candidate_id].append(record)

    for candidate in run.candidates:
        records = by_candidate.get(candidate.candidate_id, [])
        ok = [r for r in records if r.succeeded]
        durations = [r.duration_s for r in ok if r.duration_s]
        latencies = [r.latency_ms for r in ok if r.latency_ms]
        rtf = (
            fmean([(la / 1000.0) / du for la, du in zip(latencies, durations) if du])
            if durations and latencies
            else None
        )
        lines.append(
            "| `{id}` | {engine} | {mode} | {ckpt} | {ok} | {failed} | {dur} | {lat} | "
            "{rtf} | {lic} |".format(
                id=candidate.candidate_id,
                engine=candidate.engine,
                mode=candidate.adaptation_mode.value,
                ckpt=f"`{candidate.checkpoint_id}`" if candidate.checkpoint_id else "—",
                ok=len(ok),
                failed=len(records) - len(ok),
                dur=f"{_median(durations):.2f}s" if durations else "—",
                lat=f"{_median(latencies):.0f}ms" if latencies else "—",
                rtf=f"{rtf:.2f}" if rtf else "—",
                lic=candidate.license,
            )
        )
    lines.append("")

    dropped = defaultdict(set)
    for record in succeeded:
        for control in record.dropped_controls:
            dropped[record.candidate_id].add(control)
    if dropped:
        lines.append("## Controls the engines ignored")
        lines.append("")
        lines.append(
            "These style directions were sent but not honoured. Differences on those "
            "dimensions are not attributable in the listening test."
        )
        lines.append("")
        for candidate_id, controls in sorted(dropped.items()):
            lines.append(f"- `{candidate_id}`: {', '.join(sorted(controls))}")
        lines.append("")

    if failed:
        lines.append("## Failures")
        lines.append("")
        reasons: dict[str, int] = defaultdict(int)
        for record in failed:
            reasons[f"{record.candidate_id}: {record.error}"] += 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:20]:
            lines.append(f"- ×{count} {reason}")
        lines.append("")

    lines.append("## Reference styles used")
    lines.append("")
    style_counts: dict[str, int] = defaultdict(int)
    for record in succeeded:
        style_counts[record.reference_style or "—"] += 1
    for style, count in sorted(style_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{style}`: {count}")
    lines.append("")
    return "\n".join(lines)


def render_evaluation_report(
    run: BenchmarkRun,
    report: AggregateReport,
    *,
    agreement: dict[str, float | None] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Listening-test results · `{report.run_id}`")
    lines.append("")
    lines.append(f"- Raters: {report.n_raters}")
    lines.append(f"- Blind test: `{report.test_id}`")
    lines.append(f"- Voice pack: `{run.voicepack_id}` v{run.voicepack_version}")
    lines.append("")
    if report.notes:
        lines.append(f"> {report.notes}")
        lines.append("")

    lines.append("## Scores (1-5)")
    lines.append("")
    header = ["Candidate", "n"] + [_AXIS_LABEL[a] for a in ALL_AXES] + [
        "Believed real",
        "Composite",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for candidate in report.ranked():
        row = [
            ("**real recordings**" if candidate.is_ground_truth else f"`{candidate.label}`"),
            str(candidate.n_ratings),
        ]
        for axis in ALL_AXES:
            mean = candidate.axis_mean(axis)
            std = next((a.std for a in candidate.axes if a.axis is axis), None)
            row.append(f"{mean:.2f} ±{std:.2f}" if mean is not None else "—")
        row.append(
            f"{candidate.believed_real_rate:.0%}"
            if candidate.believed_real_rate is not None
            else "—"
        )
        composite = candidate.composite()
        row.append(f"{composite:.2f}" if composite is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Gap to real recordings")
    lines.append("")
    lines.append(
        "Negative means worse than a real recording of the character on that axis. "
        "Spec §18's success condition is that this gap approaches zero on speaker and "
        "character similarity — not that any absolute score is high."
    )
    lines.append("")
    anchor_present = any(c.is_ground_truth for c in report.candidates)
    if not anchor_present:
        lines.append("_No ground-truth anchor in this test; gaps cannot be computed._")
    else:
        lines.append("| Candidate | " + " | ".join(_AXIS_LABEL[a] for a in ALL_AXES) + " |")
        lines.append("|" + "---|" * (len(ALL_AXES) + 1))
        gaps = {axis: gap_to_ground_truth(report, axis) for axis in ALL_AXES}
        for candidate in report.ranked():
            if candidate.is_ground_truth:
                continue
            row = [f"`{candidate.label}`"]
            for axis in ALL_AXES:
                value = gaps[axis].get(candidate.candidate_id)
                row.append(f"{value:+.2f}" if value is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if agreement:
        lines.append("## Inter-rater agreement")
        lines.append("")
        lines.append(
            "Mean pairwise Pearson correlation between raters. Below ~0.4 the ranking "
            "is not trustworthy regardless of sample size."
        )
        lines.append("")
        for axis_name, value in agreement.items():
            lines.append(
                f"- {axis_name}: {value:.2f}" if value is not None else f"- {axis_name}: —"
            )
        lines.append("")

    lines.append("## What this does and does not establish")
    lines.append("")
    lines.append(
        "- A high composite with a large remaining gap to the real recordings means the "
        "best candidate is still recognisably synthetic."
    )
    lines.append(
        "- High naturalness with low character similarity is the specific failure spec §5 "
        "warns about: a good generic voice, not this character."
    )
    lines.append(
        "- Before declaring a winner, confirm its licence permits the intended use and "
        "record the decision in `docs/decisions/`."
    )
    lines.append("")
    return "\n".join(lines)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
