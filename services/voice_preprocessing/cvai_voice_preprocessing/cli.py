"""``cvai-prep`` — the Voice Pack preprocessing command line (Milestone 2).

    cvai-prep backends                              what is installed
    cvai-prep run denia_cn --source ~/denia_voice   ingest → … → quality
    cvai-prep status denia_cn                       where the pack stands
    cvai-prep review denia_cn                       write the review page
    cvai-prep apply denia_cn review-patch.json      apply a reviewer's decisions
    cvai-prep anchors denia_cn add seg_a seg_b      mark confirmed character clips
    cvai-prep build denia_cn                        clean/ + dataset + reference bank

Stages are resumable: re-running ``run`` picks up where the last one stopped and never
overwrites a human's corrections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvai_core.config import load_config
from cvai_core.errors import CVAIError
from cvai_core.loaders import load_voicepack_manifest
from cvai_core.logging_setup import configure_logging
from cvai_core.paths import repo_root, voicepack_root
from cvai_core.voicepack import validate_voicepack
from cvai_types import ReviewStatus

from .backends import resolve_backends
from .config import PreprocessConfig
from .pipeline import BuildError, Pipeline, build_dataset
from .review import apply_review_patch, load_review_patch, write_review_page
from .state import Stage, load_state, save_state


def _pack(args: argparse.Namespace):
    packs_root = (
        Path(args.packs_root)
        if args.packs_root
        else repo_root() / load_config().paths.voicepacks
    )
    paths = voicepack_root(args.voicepack_id, packs_root)
    if not paths.manifest_file.is_file():
        raise CVAIError(
            f"no voice pack at {paths.root}; create one with "
            f"`cvai-voicepack init {args.voicepack_id}`"
        )
    return paths, load_voicepack_manifest(paths)


def _config(args: argparse.Namespace, manifest) -> PreprocessConfig:
    return PreprocessConfig(
        voicepack_id=manifest.voicepack_id,
        source_dir=getattr(args, "source", None),
        limit=getattr(args, "limit", None),
        target_sample_rate=manifest.target_sample_rate,
        target_lufs=manifest.target_lufs,
        segment=not getattr(args, "no_segment", False),
        enable_separation=getattr(args, "separate", False) or manifest.enable_source_separation,
        enable_denoise=getattr(args, "denoise", False) or manifest.enable_denoise,
        device=getattr(args, "device", "cpu"),
        hotwords=list(getattr(args, "hotword", None) or []),
        auto_approve=getattr(args, "auto_approve", False),
        speech_pad_ms=getattr(args, "speech_pad_ms", 180),
    )


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def cmd_backends(args: argparse.Namespace) -> int:
    bundle = resolve_backends(
        prefer_device=args.device,
        enable_separation=args.separate,
        enable_denoise=args.denoise,
    )
    print(bundle.render_report())
    if bundle.asr_is_stub:
        print(
            "\nNo ASR model is installed, so transcripts would be placeholders.\n"
            "Run `make install-preprocess` before preparing a real pack."
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    config = _config(args, manifest)
    pipeline = Pipeline(paths, manifest, config)

    print(pipeline.backends.render_report())
    print()

    stages = None
    if args.stages:
        try:
            stages = [Stage(name.strip()) for name in args.stages.split(",") if name.strip()]
        except ValueError as exc:
            raise CVAIError(
                f"unknown stage: {exc}. Known: {', '.join(s.value for s in Stage)}"
            ) from exc

    state = pipeline.run(stages)
    _print_summary(state.summary())
    print(f"\nState: {paths.processed / 'state.json'}")
    print(f"Next:  cvai-prep review {manifest.voicepack_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    state = load_state(paths.processed, manifest.voicepack_id)
    print(f"Voice pack: {manifest.voicepack_id}  v{manifest.version}")
    _print_summary(state.summary())

    if state.runs:
        print("\nStage history")
        for run in state.runs[-12:]:
            mark = "✗" if run.error else ("·" if run.skipped else "✓")
            detail = run.error or run.skip_reason or f"{run.items} items"
            print(f"  {mark} {run.stage.value:<16} {detail}")

    print()
    print(validate_voicepack(paths).render())
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    state = load_state(paths.processed, manifest.voicepack_id)
    if not state.segments:
        raise CVAIError("no segments yet; run `cvai-prep run` first")

    page = write_review_page(
        state,
        manifest,
        paths.processed,
        limit=args.limit,
        status_filter=ReviewStatus.PENDING if args.pending_only else None,
    )
    print(f"Review page: {page}")
    print(
        "Open it in a browser, correct the transcripts and styles, then export the "
        "patch and apply it with:\n"
        f"  cvai-prep apply {manifest.voicepack_id} review-patch.json"
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    state = load_state(paths.processed, manifest.voicepack_id)
    patch = load_review_patch(Path(args.patch))
    changed = apply_review_patch(state, patch)
    save_state(state, paths.processed)
    print(f"Applied {changed} segment edits from {args.patch}")
    _print_summary(state.summary())
    return 0


def cmd_anchors(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    state = load_state(paths.processed, manifest.voicepack_id)

    if args.action == "list":
        if not state.speaker_anchor_segment_ids:
            print(
                "No speaker anchors. Without them the speaker filter compares every "
                "segment against the mean of all segments, which only identifies the "
                "character if the pack is already single-speaker."
            )
        for segment_id in state.speaker_anchor_segment_ids:
            print(segment_id)
        return 0

    anchors = set(state.speaker_anchor_segment_ids)
    if args.action == "clear":
        anchors.clear()
    else:
        unknown = [sid for sid in args.segment_ids if state.segment(sid) is None]
        if unknown:
            raise CVAIError(f"unknown segment ids: {unknown}")
        if args.action == "add":
            anchors.update(args.segment_ids)
        else:
            anchors.difference_update(args.segment_ids)

    state.speaker_anchor_segment_ids = sorted(anchors)
    save_state(state, paths.processed)
    print(f"{len(state.speaker_anchor_segment_ids)} anchors")
    print("Re-run the speaker stage: cvai-prep run", manifest.voicepack_id,
          "--stages speaker_filter")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    paths, manifest = _pack(args)
    state = load_state(paths.processed, manifest.voicepack_id)
    config = _config(args, manifest)

    try:
        dataset, bank = build_dataset(
            paths,
            manifest,
            state,
            config,
            allow_stub_transcripts=args.allow_stub_transcripts,
        )
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset:    {paths.dataset_file}")
    print(f"References: {paths.references_file}")
    print(f"  {len(dataset.usable())} samples, {dataset.total_seconds() / 60:.1f} minutes")
    print(f"  minutes by style: {json.dumps(dataset.minutes_by_style(), ensure_ascii=False)}")
    print(f"  {len(bank.samples)} reference clips: {json.dumps(bank.coverage(), ensure_ascii=False)}")
    print()
    report = validate_voicepack(paths, require_dataset=True, require_references=True)
    print(report.render())
    return 0 if report.ok else 1


def _print_summary(summary: dict) -> None:
    print("\nPack state")
    for key in (
        "sources",
        "segments",
        "transcribed",
        "approved",
        "pending_review",
        "rejected",
        "human_edited",
        "approved_minutes",
    ):
        print(f"  {key:<16} {summary[key]}")
    if summary["minutes_by_style"]:
        print("  minutes by style")
        for style, minutes in summary["minutes_by_style"].items():
            print(f"    {style:<18} {minutes}")


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-prep", description=__doc__)
    parser.add_argument("--packs-root")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    backends = sub.add_parser("backends", help="show which backends are installed")
    backends.add_argument("--device", default="cpu")
    backends.add_argument("--separate", action="store_true")
    backends.add_argument("--denoise", action="store_true")
    backends.set_defaults(func=cmd_backends)

    run = sub.add_parser("run", help="run the preprocessing stages")
    run.add_argument("voicepack_id")
    run.add_argument("--source", help="directory of original audio to ingest")
    run.add_argument("--stages", help="comma-separated subset, e.g. transcribe,annotate")
    run.add_argument("--limit", type=int, help="process only the first N source clips")
    run.add_argument("--device", default="cpu")
    run.add_argument(
        "--no-segment",
        action="store_true",
        help="one segment per file (game dialogue is often already one line per file)",
    )
    run.add_argument("--speech-pad-ms", type=int, default=180)
    run.add_argument(
        "--separate",
        action="store_true",
        help="isolate vocals — only when music or effects are mixed under the dialogue",
    )
    run.add_argument(
        "--denoise",
        action="store_true",
        help="denoise — attenuates breaths and texture; usually leave off",
    )
    run.add_argument("--hotword", action="append", help="proper noun for ASR; repeatable")
    run.add_argument(
        "--auto-approve",
        action="store_true",
        help="approve everything that passes the automatic checks (skips human review)",
    )
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="summarise pack state")
    status.add_argument("voicepack_id")
    status.set_defaults(func=cmd_status)

    review = sub.add_parser("review", help="write the human review page")
    review.add_argument("voicepack_id")
    review.add_argument("--limit", type=int)
    review.add_argument("--pending-only", action="store_true")
    review.set_defaults(func=cmd_review)

    apply_parser = sub.add_parser("apply", help="apply a review patch")
    apply_parser.add_argument("voicepack_id")
    apply_parser.add_argument("patch")
    apply_parser.set_defaults(func=cmd_apply)

    anchors = sub.add_parser("anchors", help="manage speaker anchor segments")
    anchors.add_argument("voicepack_id")
    anchors.add_argument("action", choices=["add", "remove", "list", "clear"])
    anchors.add_argument("segment_ids", nargs="*")
    anchors.set_defaults(func=cmd_anchors)

    build = sub.add_parser("build", help="write clean/, dataset.json and references.json")
    build.add_argument("voicepack_id")
    build.add_argument(
        "--allow-stub-transcripts",
        action="store_true",
        help="build even though transcripts are placeholders (pipeline testing only)",
    )
    build.set_defaults(func=cmd_build)

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
