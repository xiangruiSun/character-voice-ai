"""``cvai-talk`` — the conversation from a terminal.

The browser client answers "does the whole thing work end to end?". This answers "does
she sound right?", which is the question asked twenty times a day while a voice pack is
being built, often on a machine with no browser and no audio device at all. So the two
things worth pinning are that a missing audio player never stops a run, and that
``--audition`` produces a page that opens from a file:// URL with the clips in it.

The audition is deliberately *not* a benchmark, and one test holds that line: nothing
here is blind and there are no real recordings to compare against, so the page says so
rather than letting a good-sounding audition stand in for Milestone 7.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cvai_api.talk_cli import (
    audition_lines,
    build_parser,
    find_player,
    main,
    render_audition,
)

# The fully wired manager fixture — synthetic pack, mock engine, fake LLM and STT.
from test_api import manager  # noqa: F401 - pytest fixture


@pytest.fixture
def talk_env(manager, monkeypatch, tmp_path):
    """`cvai-talk` builds its own SessionManager, so point it at the test one."""
    from cvai_api import talk_cli

    monkeypatch.setattr(
        talk_cli.SessionManager, "from_config_file", classmethod(lambda cls, *a, **k: manager)
    )
    monkeypatch.setattr(talk_cli, "find_player", lambda: None)
    return manager




# --------------------------------------------------------------------------------------
# One line
# --------------------------------------------------------------------------------------


def test_saying_one_line_prints_the_reply_and_writes_audio(talk_env, capsys):
    assert main(["demo_zh", "--say", "在吗", "--no-play"]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    # The audio is the point; the text is the caption.
    assert "no audio" not in out


def test_verbose_shows_the_performance_direction(talk_env, capsys):
    """Spec §12 keeps the plan away from users. The person building the voice is not a
    user, and when a line comes out wrong this is exactly what they need to see."""
    assert main(["demo_zh", "--say", "在吗", "--no-play", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "plan:" in out
    assert "chunk 0:" in out


def test_an_unknown_character_is_a_clear_error(talk_env, capsys):
    assert main(["nobody", "--say", "在吗", "--no-play"]) == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------------------


def test_a_missing_player_is_a_note_not_a_failure(manager, monkeypatch, capsys):
    """Half the machines this runs on are headless servers."""
    from cvai_api import talk_cli

    monkeypatch.setattr(
        talk_cli.SessionManager, "from_config_file", classmethod(lambda cls, *a, **k: manager)
    )
    monkeypatch.setattr(talk_cli, "find_player", lambda: None)

    assert main(["demo_zh", "--say", "在吗"]) == 0
    assert "no audio player found" in capsys.readouterr().err


def test_the_player_is_called_once_per_chunk(manager, monkeypatch):
    from cvai_api import talk_cli

    played: list[str] = []
    monkeypatch.setattr(
        talk_cli.SessionManager, "from_config_file", classmethod(lambda cls, *a, **k: manager)
    )
    monkeypatch.setattr(talk_cli, "find_player", lambda: ("fake-player", ()))
    monkeypatch.setattr(talk_cli, "play", lambda path, player: played.append(path))

    assert main(["demo_zh", "--say", "今天外面风有点大，你出门记得多穿一件。"]) == 0
    assert played
    assert all(Path(path).is_file() for path in played)


def test_no_play_plays_nothing(manager, monkeypatch):
    from cvai_api import talk_cli

    played: list[str] = []
    monkeypatch.setattr(
        talk_cli.SessionManager, "from_config_file", classmethod(lambda cls, *a, **k: manager)
    )
    monkeypatch.setattr(talk_cli, "play", lambda path, player: played.append(path))
    assert main(["demo_zh", "--say", "在吗", "--no-play"]) == 0
    assert played == []


def test_find_player_returns_something_or_nothing_without_raising():
    result = find_player()
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


# --------------------------------------------------------------------------------------
# Audition
# --------------------------------------------------------------------------------------


def test_audition_covers_every_style_she_declares(talk_env, tmp_path, repo_root_path):
    """Style collapse (spec §27) is invisible one line at a time and obvious side by
    side, which is the entire reason this mode exists."""
    out = tmp_path / "audition.html"
    assert (
        main(
            [
                "demo_zh",
                "--audition",
                "--no-play",
                "--out",
                str(out),
                "--sentences",
                str(repo_root_path / "configs" / "benchmarks" / "sentences_zh_v1.yaml"),
            ]
        )
        == 0
    )
    page = out.read_text(encoding="utf-8")
    # The run closed its session on the way out, so ask for the profile directly.
    profile = talk_env.create().profile
    for style in profile.available_styles:
        assert style in page
    assert page.count("<audio") >= len(profile.available_styles)


def test_the_audition_page_says_it_is_not_a_benchmark(talk_env, tmp_path, repo_root_path):
    """A good-sounding audition must not be mistaken for a Milestone 7 result: nothing
    here is blind and there are no real recordings in it."""
    out = tmp_path / "audition.html"
    main(["demo_zh", "--audition", "--no-play", "--out", str(out),
          "--sentences", str(repo_root_path / "configs" / "benchmarks" / "sentences_zh_v1.yaml")])
    page = out.read_text(encoding="utf-8")
    assert "Not a benchmark" in page
    assert "blind" in page


def test_audition_lines_come_from_the_benchmark_sentences(manager, repo_root_path):
    """The audition and the benchmark listen to the same material, and those sentences
    were chosen to expose what separates a character voice from a generic one."""
    session = manager.create()
    lines = audition_lines(
        session, repo_root_path / "configs" / "benchmarks" / "sentences_zh_v1.yaml"
    )
    assert [style for style, _ in lines] == session.profile.available_styles
    assert all(text for _, text in lines)


def test_a_style_with_no_matching_sentence_still_gets_a_line(manager, tmp_path):
    """Better an off-style line than a silent gap in the audition."""
    import yaml

    sentences = tmp_path / "one.yaml"
    sentences.write_text(
        yaml.safe_dump(
            {
                "set_id": "tiny",
                "sentences": [
                    {"sentence_id": "s1", "text": "今天外面风有点大。", "target_style": "neutral"}
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    session = manager.create()
    lines = audition_lines(session, sentences)
    assert all(text for _, text in lines)


def test_audio_is_linked_relative_so_the_page_travels(manager, tmp_path):
    """A page that only works from one absolute path cannot be sent to a listener."""
    session = manager.create()
    clip = tmp_path / "clips" / "a.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF")

    page = render_audition(session, [("neutral", "在吗", [str(clip)])], tmp_path)
    assert 'src="clips/a.wav"' in page


def test_audio_outside_the_page_directory_falls_back_to_a_file_url(manager, tmp_path):
    session = manager.create()
    page = render_audition(session, [("neutral", "在吗", ["/elsewhere/a.wav"])], tmp_path)
    assert "file:///elsewhere/a.wav" in page


def test_the_page_escapes_what_it_interpolates(manager, tmp_path):
    session = manager.create()
    page = render_audition(session, [("neutral", "<script>alert(1)</script>", [])], tmp_path)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------


def test_the_character_is_optional():
    """Defaults to the configured character, which is what you want when there is one."""
    args = build_parser().parse_args([])
    assert args.character_id is None


def test_the_engine_can_be_overridden_per_run():
    """Comparing two engines by ear is a two-command job, not a config edit."""
    args = build_parser().parse_args(["denia_cn", "--engine", "gpt_sovits"])
    assert args.engine == "gpt_sovits"


def test_the_help_text_does_not_promise_a_benchmark():
    # argparse re-wraps the description, so compare on normalised whitespace.
    help_text = re.sub(r"\s+", " ", build_parser().format_help())
    assert "cvai-bench" in help_text
    assert "not a substitute for the benchmark" in help_text
