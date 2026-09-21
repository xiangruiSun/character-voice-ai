"""``cvai-voicepack`` — create and check voice packs.

    cvai-voicepack init denia_cn --character denia_cn --display-name "迪尼娅"
    cvai-voicepack validate denia_cn
    cvai-voicepack validate denia_cn --strict      # require references and a dataset
    cvai-voicepack list

``validate`` is the gate before any training run: it answers "is this pack good enough
to spend GPU hours on?" and exits non-zero when it is not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_config
from ..errors import CVAIError
from ..loaders import FilesystemCharacterProvider
from ..logging_setup import configure_logging
from ..paths import repo_root, voicepack_root
from ..voicepack import scaffold_voicepack, validate_voicepack
from cvai_types import CoreStyle, SourceInfo, StyleDefinition, VoicePackManifest


def _packs_root(args: argparse.Namespace) -> Path:
    if args.packs_root:
        return Path(args.packs_root)
    config = load_config([Path(args.config)] if args.config else None)
    return repo_root() / config.paths.voicepacks


def cmd_init(args: argparse.Namespace) -> int:
    paths = voicepack_root(args.voicepack_id, _packs_root(args))
    manifest = VoicePackManifest(
        voicepack_id=args.voicepack_id,
        character_id=args.character or args.voicepack_id,
        display_name=args.display_name or args.voicepack_id,
        target_sample_rate=args.sample_rate,
        styles=[
            StyleDefinition(name=style.value, core_style=style) for style in CoreStyle
        ],
        source=SourceInfo(
            description=args.source or "",
            license_note="TODO — record the usage-rights check before training.",
        ),
        notes=(
            "Scaffolded by cvai-voicepack init. Fill raw/ with original audio, then run "
            "the Milestone 2 preprocessing pipeline."
        ),
    )
    report = scaffold_voicepack(paths, manifest, overwrite=args.force)
    print(f"Created voice pack at {paths.root}")
    print(report.render())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = voicepack_root(args.voicepack_id, _packs_root(args))

    profile = None
    if args.check_profile:
        provider = FilesystemCharacterProvider()
        try:
            profile = provider.get(args.check_profile)
        except CVAIError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    report = validate_voicepack(
        paths,
        profile=profile,
        require_dataset=args.strict,
        require_references=args.strict,
    )
    print(report.render())
    return 0 if report.ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    root = _packs_root(args)
    if not root.is_dir():
        print(f"no voice packs directory at {root}")
        return 0
    found = False
    for entry in sorted(root.iterdir()):
        if not (entry / "metadata" / "voicepack.yaml").is_file():
            continue
        found = True
        report = validate_voicepack(voicepack_root(entry.name, root))
        status = "ok" if report.ok else f"{len(report.errors())} error(s)"
        minutes = report.stats.get("dataset_minutes_usable", "—")
        clips = report.stats.get("reference_clips", 0)
        print(f"{entry.name:20s}  {status:14s}  {minutes} min usable  {clips} refs")
    if not found:
        print(f"no voice packs found under {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-voicepack", description=__doc__)
    parser.add_argument("--config", help="path to app.yaml")
    parser.add_argument("--packs-root", help="override the voicepacks directory")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="scaffold a new voice pack")
    init_parser.add_argument("voicepack_id")
    init_parser.add_argument("--character", help="character_id (default: voicepack_id)")
    init_parser.add_argument("--display-name")
    init_parser.add_argument("--source", help="where the audio comes from")
    init_parser.add_argument("--sample-rate", type=int, default=44100)
    init_parser.add_argument("--force", action="store_true", help="overwrite a manifest")
    init_parser.set_defaults(func=cmd_init)

    validate_parser = sub.add_parser("validate", help="check a voice pack")
    validate_parser.add_argument("voicepack_id")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="require a reference bank and a dataset (use before training)",
    )
    validate_parser.add_argument(
        "--check-profile",
        metavar="CHARACTER_ID",
        help="also check that a character profile's styles are supported by this pack",
    )
    validate_parser.set_defaults(func=cmd_validate)

    list_parser = sub.add_parser("list", help="list voice packs and their status")
    list_parser.set_defaults(func=cmd_list)
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
