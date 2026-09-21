"""``cvai-bench`` — command line for the benchmark pipeline.

    cvai-bench run configs/benchmarks/denia_cn_v1.yaml
    cvai-bench blind runs/<run_id>
    cvai-bench aggregate runs/<run_id> --ratings ratings/
    cvai-bench report runs/<run_id>
    cvai-bench demo

``demo`` chains everything with the mock engine and simulated ratings, which is how the
whole apparatus stays exercised on a machine with no GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from cvai_core.config import load_config
from cvai_core.errors import CVAIError
from cvai_core.logging_setup import configure_logging, get_logger
from cvai_core.paths import repo_root
from cvai_core.runlog import RunPaths
from cvai_types import BenchmarkRun, BlindKey, BlindTestSet, RatingAxis

from .blind import build_blind_test, export_webmushra_config, write_blind_test
from .config import load_benchmark_config
from .demo_pack import build_demo_voicepack
from .objective import (
    ProsodyProfile,
    render_objective_report,
    resolve_objective_backends,
    score_run,
)
from .report import render_evaluation_report, render_run_report
from .runner import BenchmarkRunner
from .scoring import ALL_AXES, aggregate, load_ratings, rater_agreement
from .simulate import simulate_ratings, write_ratings

log = get_logger("cvai-bench")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _load_run(run_dir: Path) -> tuple[RunPaths, BenchmarkRun]:
    paths = RunPaths(run_dir)
    if not paths.run_file.is_file():
        raise CVAIError(f"no run.json in {run_dir}")
    return paths, BenchmarkRun.model_validate_json(
        paths.run_file.read_text(encoding="utf-8")
    )


def _load_blind(paths: RunPaths) -> tuple[BlindTestSet, BlindKey]:
    items = paths.blind / "items.json"
    key = paths.blind / "key.json"
    if not items.is_file() or not key.is_file():
        raise CVAIError(
            f"{paths.root} has no blind test yet; run `cvai-bench blind {paths.root}`"
        )
    return (
        BlindTestSet.model_validate_json(items.read_text(encoding="utf-8")),
        BlindKey.model_validate_json(key.read_text(encoding="utf-8")),
    )


def _app_config(args: argparse.Namespace):
    paths = [Path(args.config)] if args.config else None
    return load_config(paths)


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    config = _app_config(args)
    benchmark = load_benchmark_config(Path(args.benchmark))
    runner = BenchmarkRunner(config, benchmark, run_id=args.run_id)
    run = asyncio.run(runner.run())

    report = render_run_report(run)
    runner.run_paths.report_file.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nRun directory: {runner.run_paths.root}")
    failed = len(run.failures())
    if failed:
        print(f"note: {failed} generations failed — see events.jsonl", file=sys.stderr)
    return 0


def cmd_blind(args: argparse.Namespace) -> int:
    paths, run = _load_run(Path(args.run_dir))
    test, key = build_blind_test(run, paths, shuffle_seed=args.seed)
    written = write_blind_test(paths, test, key)
    if args.webmushra:
        written["webmushra"] = export_webmushra_config(test, paths)

    print(f"Blind test `{test.test_id}` with {len(test.items)} items")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(
        "\nOpen the rating page in a browser and hand it to a listener. "
        "Do not give them key.json."
    )
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    paths, run = _load_run(Path(args.run_dir))
    test, key = _load_blind(paths)

    if args.simulate:
        ratings = simulate_ratings(key, n_raters=args.raters, seed=args.seed)
        written = write_ratings(ratings, paths.root / "ratings-simulated.json")
        print(f"! using SIMULATED ratings ({written}) — not evidence of anything")
    else:
        if not args.ratings:
            raise CVAIError("pass --ratings <file-or-directory>, or --simulate")
        ratings = load_ratings(Path(args.ratings))

    if not ratings:
        raise CVAIError("no ratings loaded")

    report = aggregate(run, key, ratings)
    paths.ratings_file.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    agreement = {
        axis.value: rater_agreement(ratings, axis) for axis in ALL_AXES
    }
    rendered = render_evaluation_report(run, report, agreement=agreement)
    (paths.root / "evaluation.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    paths, run = _load_run(Path(args.run_dir))
    rendered = render_run_report(run)
    if args.objective:
        rendered += "\n" + _objective_section(run, paths, args)
    paths.report_file.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def _objective_section(run, paths, args) -> str:
    """Measure generated audio against the character's real prosody distribution."""
    from cvai_core.loaders import open_voicepack, try_load_dataset_manifest

    packs_root = Path(args.packs_root) if getattr(args, "packs_root", None) else None
    try:
        pack_paths, _ = open_voicepack(run.voicepack_id, packs_root)
    except CVAIError as exc:
        return f"## Objective metrics\n\n> unavailable: {exc}\n"

    dataset = try_load_dataset_manifest(pack_paths)
    if dataset is None:
        return (
            "## Objective metrics\n\n> unavailable: the voice pack has no dataset, so "
            "there is nothing to compare generated prosody against.\n"
        )

    profile = ProsodyProfile.from_dataset(pack_paths, dataset)
    backends = resolve_objective_backends()
    report = score_run(run, paths, profile, backends=backends)
    return render_objective_report(report)


def cmd_demo(args: argparse.Namespace) -> int:
    root = repo_root()
    pack = root / "voicepacks" / "demo_zh"
    if args.rebuild_pack or not (pack / "metadata" / "voicepack.yaml").is_file():
        print("Building the synthetic demo voice pack …")
        build_demo_voicepack(pack, overwrite=True)

    config = load_config([root / "configs" / "app.yaml"])
    benchmark = load_benchmark_config(root / "configs" / "benchmarks" / "demo_mock.yaml")
    runner = BenchmarkRunner(config, benchmark)
    run = asyncio.run(runner.run())
    runner.run_paths.report_file.write_text(render_run_report(run), encoding="utf-8")

    test, key = build_blind_test(run, runner.run_paths, shuffle_seed=11)
    write_blind_test(runner.run_paths, test, key)

    ratings = simulate_ratings(
        key,
        n_raters=4,
        seed=11,
        candidate_bias={"mock_a": 0.0, "mock_b": 0.35, "mock_c": -0.35},
    )
    write_ratings(ratings, runner.run_paths.root / "ratings-simulated.json")
    aggregated = aggregate(run, key, ratings)
    agreement = {axis.value: rater_agreement(ratings, axis) for axis in ALL_AXES}
    rendered = render_evaluation_report(run, aggregated, agreement=agreement)
    (runner.run_paths.root / "evaluation.md").write_text(rendered, encoding="utf-8")

    print(render_run_report(run))
    print(rendered)
    print(f"Run directory: {runner.run_paths.root}")
    print(f"Rating page:   {runner.run_paths.blind / 'rate.html'}")
    print(
        "\nThe ratings above are simulated. They demonstrate that the pipeline works; "
        "they say nothing about any engine."
    )
    return 0


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-bench", description=__doc__)
    parser.add_argument("--config", help="path to app.yaml (default: configs/app.yaml)")
    parser.add_argument("--packs-root", help="override the voicepacks directory")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="generate audio for every candidate")
    run_parser.add_argument("benchmark", help="path to a benchmark YAML")
    run_parser.add_argument("--run-id", help="override the generated run id")
    run_parser.set_defaults(func=cmd_run)

    blind_parser = sub.add_parser("blind", help="assemble the blind listening test")
    blind_parser.add_argument("run_dir")
    blind_parser.add_argument("--seed", type=int, default=0)
    blind_parser.add_argument(
        "--webmushra", action="store_true", help="also emit a webMUSHRA config"
    )
    blind_parser.set_defaults(func=cmd_blind)

    agg_parser = sub.add_parser("aggregate", help="fold ratings into a report")
    agg_parser.add_argument("run_dir")
    agg_parser.add_argument("--ratings", help="rating file or directory of them")
    agg_parser.add_argument(
        "--simulate",
        action="store_true",
        help="invent ratings to test the pipeline (never evidence)",
    )
    agg_parser.add_argument("--raters", type=int, default=3)
    agg_parser.add_argument("--seed", type=int, default=7)
    agg_parser.set_defaults(func=cmd_aggregate)

    report_parser = sub.add_parser("report", help="re-render the run report")
    report_parser.add_argument("run_dir")
    report_parser.add_argument(
        "--objective",
        action="store_true",
        help="also measure generated prosody against the character's real distribution",
    )
    report_parser.set_defaults(func=cmd_report)

    demo_parser = sub.add_parser(
        "demo", help="end-to-end pipeline demo with the mock engine"
    )
    demo_parser.add_argument(
        "--rebuild-pack",
        action="store_true",
        help="regenerate the synthetic demo voice pack before running",
    )
    demo_parser.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except CVAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
