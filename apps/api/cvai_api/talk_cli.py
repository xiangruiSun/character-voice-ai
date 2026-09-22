"""``cvai-talk`` — the conversation from a terminal.

    cvai-talk denia_cn                      # type at her, hear her answer
    cvai-talk denia_cn --say "在吗"          # one line and exit
    cvai-talk denia_cn --audition           # one line per style, as a page to listen to

The same orchestrator the API drives, with a different transport. It exists because the
browser client answers "does the whole thing work end to end?" and this answers "does she
sound right?" — and the second question is asked twenty times a day while a voice pack is
being built, on a machine that may have no browser at all.

``--audition`` is the one worth knowing about. It takes a line in every style the
character declares and writes a single self-contained page with the clips side by side,
which is how style collapse (spec §27 — "every line sounds the same") becomes obvious in
thirty seconds. It is not a substitute for the benchmark: no blind ordering, no anchors,
no real recordings to compare against, so it tells you a voice pack is wrong but never
that it is right. Milestone 7 is decided by ``cvai-bench``, not here.

Audio plays through whatever the system has (``afplay``, ``aplay``, ``ffplay``, ``play``,
``paplay``). With none of them, the paths are printed — the files are the point, and a
missing player must not stop the run.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import shutil
import subprocess
import sys
from pathlib import Path

from cvai_conversation import OrchestratorConfig
from cvai_core.errors import CVAIError
from cvai_core.loaders import load_sentence_set
from cvai_core.logging_setup import configure_logging
from cvai_core.paths import repo_root
from cvai_types import TurnEvent, TurnEventType

from .sessions import Session, SessionManager

#: Players in the order they are tried. Each takes a path as its last argument.
_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("afplay", ()),                               # macOS
    ("paplay", ()),                               # PulseAudio
    ("aplay", ("-q",)),                           # ALSA
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
    ("play", ("-q",)),                            # SoX
)


def find_player() -> tuple[str, tuple[str, ...]] | None:
    for name, flags in _PLAYERS:
        if shutil.which(name):
            return name, flags
    return None


def play(path: str, player: tuple[str, tuple[str, ...]] | None) -> None:
    if player is None:
        return
    name, flags = player
    try:
        subprocess.run([name, *flags, path], check=False, capture_output=True)
    except OSError as exc:  # pragma: no cover - platform specific
        print(f"  (could not play {path}: {exc})", file=sys.stderr)


# --------------------------------------------------------------------------------------
# Turns
# --------------------------------------------------------------------------------------


def report(events: list[TurnEvent], *, player, verbose: bool) -> list[str]:
    """Print what happened and play the audio. Returns the clips, in order."""
    text = next(
        (e.text for e in events if e.type is TurnEventType.TEXT_DELTA and e.is_final), ""
    )
    if text:
        print(f"  {text}")

    if verbose:
        # The performance direction never reaches a user (spec §12), but the person
        # building the voice is not a user — this is exactly what they need to see when
        # a line comes out wrong.
        for event in events:
            if event.type is TurnEventType.PLAN and event.detail:
                print(f"  · plan: {event.detail}")

    clips: list[str] = []
    for event in events:
        if event.type is TurnEventType.AUDIO and event.audio_path:
            clips.append(event.audio_path)
            if verbose:
                print(
                    f"  · chunk {event.chunk_index}: {event.text}  "
                    f"[{event.reference_id} · {event.duration_s:.2f}s]"
                )
            if player is not None:
                play(event.audio_path, player)

    for event in events:
        if event.type is TurnEventType.ERROR:
            print(f"  ! {event.error}", file=sys.stderr)
    if not clips:
        print("  (no audio — is the engine configured and running?)", file=sys.stderr)
    return clips


async def one_turn(session: Session, text: str, *, player, verbose: bool) -> list[str]:
    return report(await session.orchestrator.collect(text), player=player, verbose=verbose)


async def repl(session: Session, *, player, verbose: bool) -> int:
    print(f"Talking to {session.profile.character_name} ({session.engine}).")
    print("Ctrl-D or an empty line to leave.\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        try:
            await one_turn(session, text, player=player, verbose=verbose)
        except CVAIError as exc:
            print(f"  ! {exc}", file=sys.stderr)
        print()
    return 0


# --------------------------------------------------------------------------------------
# Audition
# --------------------------------------------------------------------------------------


def audition_lines(session: Session, sentence_set_path: Path) -> list[tuple[str, str]]:
    """One line per style the character declares, taken from the test sentence set.

    Reusing the benchmark's sentences rather than inventing lines here keeps the
    audition and the benchmark listening to the same material, and those sentences were
    chosen to expose what separates a character voice from a competent generic one.
    """
    sentences = load_sentence_set(sentence_set_path).sentences
    by_style: dict[str, str] = {}
    for sentence in sentences:
        by_style.setdefault(sentence.target_style, sentence.text)

    fallback = next((s.text for s in sentences), "今天外面风有点大。")
    return [
        (style, by_style.get(style, fallback))
        for style in session.profile.available_styles
    ]


async def audition(
    session: Session, sentence_set_path: Path, out: Path, *, player, verbose: bool
) -> int:
    rows: list[tuple[str, str, list[str]]] = []
    for style, text in audition_lines(session, sentence_set_path):
        print(f"[{style}] {text}")
        events = await session.orchestrator.collect(text)
        clips = report(events, player=player, verbose=verbose)
        rows.append((style, text, clips))
        print()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_audition(session, rows, out.parent), encoding="utf-8")
    print(f"Wrote {out}")
    print(
        "Listen for one thing first: whether the styles are actually different. If they "
        "are not, the fault is upstream of the engine — the Reference Bank or the "
        "profile's styles."
    )
    return 0


def render_audition(session: Session, rows, base: Path) -> str:
    """A self-contained page. No build step, no network, opens from a file:// URL."""
    parts = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(session.profile.character_name)} · audition</title>",
        "<style>body{font:15px/1.6 system-ui,'PingFang SC','Microsoft YaHei',sans-serif;"
        "max-width:760px;margin:40px auto;padding:0 16px}"
        "h1{font-size:19px}tr{border-bottom:1px solid #e3e6ec}"
        "table{border-collapse:collapse;width:100%}td{padding:10px 6px;vertical-align:top}"
        ".style{font-family:ui-monospace,monospace;color:#5b6270;white-space:nowrap}"
        "audio{height:32px}.note{color:#5b6270;font-size:13px}</style></head><body>",
        f"<h1>{html.escape(session.profile.character_name)} — style audition</h1>",
        f"<p class=note>Engine <code>{html.escape(session.engine)}</code>, voice pack "
        f"<code>{html.escape(session.profile.voice.voicepack_id)}</code>. "
        "Not a benchmark: nothing here is blind, and there are no real recordings to "
        "compare against. It answers whether the styles differ, not whether they are "
        "hers.</p><table>",
    ]
    for style, text, clips in rows:
        players = "".join(
            f'<audio controls src="{html.escape(_relative(clip, base))}"></audio> '
            for clip in clips
        ) or "<span class=note>no audio</span>"
        parts.append(
            f"<tr><td class=style>{html.escape(style)}</td>"
            f"<td>{html.escape(text)}<br>{players}</td></tr>"
        )
    parts.append("</table></body></html>")
    return "\n".join(parts)


def _relative(clip: str, base: Path) -> str:
    try:
        return Path(clip).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(clip).resolve().as_uri()


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cvai-talk", description=__doc__,
        # The module docstrings are written as usage examples; the default
        # formatter reflows them into one unreadable paragraph.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("character_id", nargs="?", help="defaults to the configured one")
    parser.add_argument("--say", help="one line, then exit")
    parser.add_argument(
        "--audition",
        action="store_true",
        help="speak one line per declared style and write a page to listen through",
    )
    parser.add_argument("--engine", help="override the TTS engine for this session")
    parser.add_argument("--config", help="path to app.yaml")
    parser.add_argument("--out", help="where the audition page goes")
    parser.add_argument(
        "--sentences",
        help="test sentence set for --audition "
        "(default: configs/benchmarks/sentences_zh_v1.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the performance plan, chunking and reference clips",
    )
    parser.add_argument("--no-play", action="store_true", help="write audio, play nothing")
    parser.add_argument("--log-level", default="WARNING")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    manager = SessionManager.from_config_file(
        Path(args.config) if args.config else None,
        orchestrator_config=OrchestratorConfig(),
    )
    try:
        session = manager.create(args.character_id, engine=args.engine)
    except CVAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    player = None if args.no_play else find_player()
    if player is None and not args.no_play:
        print(
            "note: no audio player found (afplay/aplay/ffplay/play/paplay); "
            "clips are written to disk instead.\n",
            file=sys.stderr,
        )

    try:
        if args.audition:
            sentences = (
                Path(args.sentences)
                if args.sentences
                else repo_root() / "configs" / "benchmarks" / "sentences_zh_v1.yaml"
            )
            out = Path(args.out) if args.out else Path("runs") / "audition.html"
            return asyncio.run(
                audition(session, sentences, out, player=player, verbose=args.verbose)
            )
        if args.say:
            asyncio.run(one_turn(session, args.say, player=player, verbose=args.verbose))
            return 0
        return asyncio.run(repl(session, player=player, verbose=args.verbose))
    except CVAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        asyncio.run(manager.close_all())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
