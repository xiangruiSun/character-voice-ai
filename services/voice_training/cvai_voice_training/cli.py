"""``cvai-export`` — turn an approved Voice Pack dataset into an engine's training format.

    cvai-export list
    cvai-export denia_cn gpt_sovits
    cvai-export denia_cn qwen3_tts --out /data/exports/denia_qwen
    cvai-export denia_cn all

Exports land in ``voicepacks/<id>/datasets/<engine>/`` by default, next to the pack they
came from, each with an ``export.json`` recording the pack version and split counts so a
training run can be traced back to its data (spec §23).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvai_core.config import load_config
from cvai_core.errors import CVAIError
from cvai_core.loaders import load_dataset_manifest, load_voicepack_manifest
from cvai_core.logging_setup import configure_logging
from cvai_core.paths import repo_root, voicepack_root

from .exporters import EXPORTERS, export_dataset


def cmd_list(args: argparse.Namespace) -> int:
    print("Engines with a dataset exporter:")
    for engine in sorted(EXPORTERS):
        print(f"  {engine}")
    print(
        "\nindex_tts has no exporter: it has no documented fine-tuning path and "
        "competes zero-shot."
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    packs_root = (
        Path(args.packs_root)
        if args.packs_root
        else repo_root() / load_config().paths.voicepacks
    )
    paths = voicepack_root(args.voicepack_id, packs_root)
    if not paths.manifest_file.is_file():
        raise CVAIError(f"no voice pack at {paths.root}")
    if not paths.dataset_file.is_file():
        raise CVAIError(
            f"{args.voicepack_id} has no dataset yet; run `cvai-prep build "
            f"{args.voicepack_id}` first"
        )

    manifest = load_voicepack_manifest(paths)
    dataset = load_dataset_manifest(paths)

    engines = sorted(EXPORTERS) if args.engine == "all" else [args.engine]
    failures = 0
    for engine in engines:
        output = (
            Path(args.out)
            if args.out and len(engines) == 1
            else paths.datasets / engine
        )
        try:
            result = export_dataset(
                engine,
                paths,
                manifest,
                dataset,
                output,
                speaker_name=args.speaker,
            )
        except (KeyError, ValueError) as exc:
            print(f"error exporting {engine}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(result.render())
        print()
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-export", description=__doc__,
        # The module docstrings are written as usage examples; the default
        # formatter reflows them into one unreadable paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--packs-root")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show available exporters")
    listing.set_defaults(func=cmd_list)

    export = sub.add_parser("export", help="export a dataset")
    export.add_argument("voicepack_id")
    export.add_argument("engine", help="engine key, or 'all'")
    export.add_argument("--out", help="output directory (single engine only)")
    export.add_argument("--speaker", help="speaker name in the export (default: character_id)")
    export.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `cvai-export denia_cn gpt_sovits` reads better than forcing the `export`
    # subcommand, so insert it when the first token is not a known command.
    if argv and argv[0] not in {"list", "export", "-h", "--help"} and not argv[0].startswith("-"):
        argv.insert(0, "export")

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
