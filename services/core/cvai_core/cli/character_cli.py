"""``cvai-character`` — write and check the "what she says" half.

    cvai-character init denia_cn --name 迪尼娅 --voicepack denia_cn
    cvai-character lint denia_cn
    cvai-character lint denia_cn --pack denia_cn   # also check the voice can perform it
    cvai-character list

``lint`` is the counterpart to ``cvai-voicepack validate``: one asks whether the voice is
good enough to spend GPU hours on, the other whether the character is written well enough
to be worth hearing. Both exit non-zero on findings that will produce wrong output, and
both are meant to be run long before anyone opens a listening test.

``init`` writes a commented template rather than an empty file, because the fields that
matter most — the original dialogue examples, the world-knowledge boundary — are the ones
people leave out when a schema does not insist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..character_lint import lint_profile
from ..config import load_config
from ..errors import CVAIError
from ..loaders import FilesystemCharacterProvider, open_voicepack
from ..logging_setup import configure_logging
from ..paths import repo_root
from ..voicepack import validate_voicepack

_TEMPLATE = """# Character profile — the "what she says" half of the system (spec §11).
#
# The dialogue examples are the highest-value part of this file. V1 does not fine-tune
# the LLM, so her original lines are the only thing in the prompt carrying her voice:
# personality adjectives say what to aim for, real lines show it. Twelve is a floor,
# not a target, and they should cover every style listed in available_styles.
#
# This file is valid as written and deliberately empty — `cvai-character lint` will tell
# you what is missing and why each thing matters. Fill it in and run it again:
#
#   cvai-character lint {character_id} --pack {voicepack_id}

character_id: {character_id}
character_name: {character_name}
language: zh-CN

# Short traits, one sentence or less each.
# e.g. 表面疏离，实际上会记住你说过的小事
personality: []

# Who she is, where she is from, what has happened to her.
background: ""

# Facts she knows about her world. This doubles as a boundary: the prompt tells her not
# to assert anything outside it. Without one she invents lore, which breaks the
# character fastest for exactly the people who know her best.
world_knowledge: []

# How she treats *this* user specifically.
relationship_style: ""

speaking_habits:
  typical_sentence_length: short
  # Catchphrases. These also become the speech recogniser's hotwords, so her name and
  # the world's proper nouns belong here or ASR will mangle them.
  frequent_expressions:
    - {character_name}
  verbal_tics: []            # fillers, laughs, sighs, written as TTS should read them
  punctuation_habits: []     # e.g. 经常用「……」拖长句尾
  pronunciation_overrides: {{}}   # per character, e.g. {{"重": "chong2"}}

# What would break *this* character. The universal rules are applied separately.
forbidden_behavior: []

# Her actual lines. `user` may be empty for a standalone line (a battle cry, an idle
# bark). `source` is how anyone can audit that these are hers and not an LLM's
# impression of her — quest name, cutscene id, voice file name.
#
#   - user: 你叫什么名字
#     character: 迪尼娅。记住了。
#     style: neutral
#     source: ch01_intro_014.wav
dialogue_examples: []

# Every style here must have examples, and her voice pack must have reference clips for
# it. A style with neither still produces speech — it just will not be her.
available_styles: [neutral]

voice:
  voicepack_id: {voicepack_id}
  default_reference_style: neutral
  # preferred_engine: filled in after the Milestone 7 decision.

llm:
  model: gpt-4o
  temperature: 0.8
  max_chars_per_reply: 120
"""


def _config(args: argparse.Namespace):
    return load_config([Path(args.config)] if args.config else None)


def _profiles_dir(args: argparse.Namespace) -> Path:
    if args.profiles_dir:
        return Path(args.profiles_dir)
    return repo_root() / _config(args).paths.characters


def _packs_root(args: argparse.Namespace) -> Path:
    if args.packs_root:
        return Path(args.packs_root)
    return repo_root() / _config(args).paths.voicepacks


def cmd_init(args: argparse.Namespace) -> int:
    directory = _profiles_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.character_id}.yaml"
    if path.exists() and not args.force:
        print(f"error: {path} already exists (use --force)", file=sys.stderr)
        return 2

    path.write_text(
        _TEMPLATE.format(
            character_id=args.character_id,
            character_name=args.name or args.character_id,
            voicepack_id=args.voicepack or args.character_id,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {path}")
    print("Fill in the dialogue examples first — they carry the voice.")
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    provider = FilesystemCharacterProvider(_profiles_dir(args))
    try:
        profile = provider.get(args.character_id)
    except CVAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = lint_profile(profile)
    print(report.render())

    ok = report.ok
    if args.pack:
        # The two halves only meet here: a style she is written to express but her voice
        # cannot perform degrades silently to neutral, which is spec §27's "identical
        # intonation" arrived at by accident.
        try:
            paths, _ = open_voicepack(args.pack, _packs_root(args))
        except CVAIError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            return 2
        pack_report = validate_voicepack(paths, profile=profile)
        print()
        print(pack_report.render())
        ok = ok and pack_report.ok

    if args.strict and report.warnings():
        return 1
    return 0 if ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    provider = FilesystemCharacterProvider(_profiles_dir(args))
    ids = provider.list_ids()
    if not ids:
        print(f"no character profiles in {provider.profiles_dir}")
        return 0
    for character_id in ids:
        try:
            profile = provider.get(character_id)
        except CVAIError as exc:
            print(f"{character_id:20s}  unreadable: {exc}")
            continue
        report = lint_profile(profile)
        status = "ok" if report.ok else f"{len(report.errors())} error(s)"
        warnings = len(report.warnings())
        print(
            f"{character_id:20s}  {status:14s}  {warnings} warning(s)  "
            f"{len(profile.dialogue_examples)} examples  "
            f"{len(profile.available_styles)} styles"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-character", description=__doc__)
    parser.add_argument("--config", help="path to app.yaml")
    parser.add_argument("--profiles-dir", help="override the character profiles directory")
    parser.add_argument("--packs-root", help="override the voicepacks directory")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="write a profile template")
    init_parser.add_argument("character_id")
    init_parser.add_argument("--name", help="display name, e.g. 迪尼娅")
    init_parser.add_argument("--voicepack", help="voice pack id (default: character id)")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init)

    lint_parser = sub.add_parser("lint", help="check a profile")
    lint_parser.add_argument("character_id")
    lint_parser.add_argument(
        "--pack",
        metavar="VOICEPACK_ID",
        help="also check that her voice pack can perform the styles she declares",
    )
    lint_parser.add_argument(
        "--strict", action="store_true", help="fail on warnings as well as errors"
    )
    lint_parser.set_defaults(func=cmd_lint)

    list_parser = sub.add_parser("list", help="list profiles and their state")
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
