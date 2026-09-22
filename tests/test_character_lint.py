"""Checking a character profile before it costs anything (spec §11, §27).

The profile is written by hand and nothing about it fails loudly. A style declared with
no examples still produces replies; an empty world-knowledge list still talks confidently
about a world it is inventing. Each of those yields a character who is subtly not her,
which is the failure the whole project exists to avoid — and catching it here costs
nothing, while catching it in a listening test costs a GPU week and a set of listeners.

The distinction these tests hold is between *malformed* (the schema's job) and
*well-formed and wrong* (this linter's job), and between findings that will produce wrong
output (errors) and findings that will produce weaker output (warnings). A profile early
in collection should be all warnings: the linter's job is to guide, not to block.
"""

from __future__ import annotations

import pytest
from cvai_core.character_lint import lint_profile
from cvai_core.voicepack import Severity
from cvai_types import CharacterProfile, DialogueExample, SpeakingHabits, VoiceBinding


#: Thirteen genuinely different lines. Numbered variations of one sentence would trip
#: the near-duplicate check, which is the linter working rather than a bug.
_LINES = [
    "迪尼娅。记住了。",
    "随便，什么都行。",
    "外面风大，你穿这么少？",
    "冷，记得加件衣服。",
    "会一点，不过别抱太大期望。",
    "……路上小心。",
    "看书。有事？",
    "习惯了。人多了吵。",
    "不用。我自己能行。",
    "你又来了。",
    "别站在门口，进来吧。",
    "我没说不行，只是懒得动。",
    "吵死了，安静点。",
]


def example(character: str, style: str = "neutral", user: str = "在吗", source: str = "ch01.wav"):
    return DialogueExample(user=user, character=character, style=style, source=source)


def make_profile(**overrides) -> CharacterProfile:
    base = {
        "character_id": "denia_cn",
        "character_name": "迪尼娅",
        "personality": ["表面疏离"],
        "background": "她在城里做零工。",
        "world_knowledge": ["这座城市叫瑞恩"],
        "speaking_habits": SpeakingHabits(frequent_expressions=["迪尼娅"]),
        "forbidden_behavior": ["不要提到现实世界"],
        "voice": VoiceBinding(voicepack_id="denia_cn", default_reference_style="neutral"),
        "available_styles": ["neutral"],
        "dialogue_examples": [example(line) for line in _LINES],
    }
    base.update(overrides)
    return CharacterProfile(**base)


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def severity_of(report, code: str) -> Severity:
    return next(issue.severity for issue in report.issues if issue.code == code)


# --------------------------------------------------------------------------------------
# A good profile
# --------------------------------------------------------------------------------------


def test_a_filled_in_profile_passes_cleanly():
    report = lint_profile(make_profile())
    assert report.ok
    assert report.warnings() == []


def test_the_report_counts_what_matters():
    report = lint_profile(make_profile())
    assert report.stats["dialogue_examples"] == len(_LINES)
    assert report.stats["distinct_examples"] == len(_LINES)
    assert report.stats["examples_by_style"] == {"neutral": len(_LINES)}


# --------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------


def test_no_examples_is_an_error():
    """Without her lines the prompt is adjectives, and adjectives produce a generic
    anime character in a voice that happens to be hers."""
    report = lint_profile(make_profile(dialogue_examples=[]))
    assert not report.ok
    assert "profile.no_examples" in codes(report)


def test_no_examples_does_not_also_complain_about_every_style():
    """Said once, the finding is actionable. Repeated per style it buries the rest."""
    report = lint_profile(
        make_profile(dialogue_examples=[], available_styles=["neutral", "soft", "teasing"])
    )
    assert "profile.style_without_examples" not in codes(report)


def test_a_handful_of_examples_is_a_warning_not_a_block():
    """Profiles are written before the lines are all collected. Guiding beats blocking."""
    report = lint_profile(make_profile(dialogue_examples=[example("就这一句。")]))
    assert report.ok
    assert "profile.few_examples" in codes(report)


def test_repeated_barks_are_counted_honestly():
    """A game rip of two hundred lines that dedupes to nine is a profile of nine, and
    the prompt builder drops the repeats whatever the profile claims."""
    repeated = [example("冲啊！", user="") for _ in range(20)]
    report = lint_profile(make_profile(dialogue_examples=repeated))
    assert "profile.duplicate_examples" in codes(report)
    assert report.stats["distinct_examples"] == 1


def test_examples_without_provenance_are_flagged():
    """Spec §11 wants *original* dialogue. Unsourced lines cannot be audited, and a
    profile quietly seeded with LLM-written lines teaches an imitation of an imitation."""
    unsourced = [example(line, source="") for line in _LINES]
    report = lint_profile(make_profile(dialogue_examples=unsourced))
    assert "profile.unsourced_examples" in codes(report)
    assert severity_of(report, "profile.unsourced_examples") is Severity.WARNING


def test_a_few_unsourced_examples_are_tolerated():
    mixed = [
        example(line, source="" if index < 4 else "ch01.wav")
        for index, line in enumerate(_LINES)
    ]
    assert "profile.unsourced_examples" not in codes(lint_profile(make_profile(dialogue_examples=mixed)))


# --------------------------------------------------------------------------------------
# Style coverage — the failure spec §27 names
# --------------------------------------------------------------------------------------


def test_a_style_with_no_examples_is_an_error():
    report = lint_profile(
        make_profile(available_styles=["neutral", "teasing"])
    )
    assert not report.ok
    assert "profile.style_without_examples" in codes(report)


def test_a_style_with_one_example_is_a_warning():
    """One line is a coincidence, not a register."""
    examples = [example(line) for line in _LINES] + [example("哼，随你便。", "teasing")]
    report = lint_profile(
        make_profile(dialogue_examples=examples, available_styles=["neutral", "teasing"])
    )
    assert report.ok
    assert "profile.thin_style" in codes(report)


def test_examples_in_styles_she_cannot_be_asked_for_are_flagged():
    """Unreachable range: the lines are there, but the planner can never request them."""
    examples = [example(line) for line in _LINES] + [
        example("哼，随你便。", "teasing"),
        example("你说什么？", "teasing"),
    ]
    report = lint_profile(make_profile(dialogue_examples=examples, available_styles=["neutral"]))
    assert "profile.examples_in_undeclared_styles" in codes(report)


def test_a_default_style_she_does_not_declare_is_an_error():
    """Every fallback lands there, so it is the most-used style in the conversation."""
    report = lint_profile(
        make_profile(
            voice=VoiceBinding(voicepack_id="denia_cn", default_reference_style="sad"),
        )
    )
    assert not report.ok
    assert "profile.default_style_unavailable" in codes(report)


# --------------------------------------------------------------------------------------
# Boundaries and habits
# --------------------------------------------------------------------------------------


def test_missing_world_knowledge_is_flagged():
    report = lint_profile(make_profile(world_knowledge=[]))
    assert "profile.no_world_knowledge" in codes(report)
    assert report.ok  # a warning: she still works, she just invents more


def test_missing_frequent_expressions_is_flagged():
    """They are the prompt's catchphrases and the recogniser's hotwords; without them
    her own name comes back from ASR mangled and the LLM answers a question nobody
    asked."""
    report = lint_profile(make_profile(speaking_habits=SpeakingHabits()))
    assert "profile.no_frequent_expressions" in codes(report)


def test_a_bad_pronunciation_override_is_an_error():
    """Silently unapplied, it reads the character the standard way — which is the exact
    mistake the override existed to prevent."""
    habits = SpeakingHabits(
        frequent_expressions=["迪尼娅"], pronunciation_overrides={"重": "chóng"}
    )
    report = lint_profile(make_profile(speaking_habits=habits))
    assert not report.ok
    assert "profile.bad_pronunciation" in codes(report)


def test_a_valid_pronunciation_override_passes():
    habits = SpeakingHabits(
        frequent_expressions=["迪尼娅"], pronunciation_overrides={"重": "chong2"}
    )
    assert lint_profile(make_profile(speaking_habits=habits)).ok


def test_a_multi_character_override_is_flagged():
    habits = SpeakingHabits(
        frequent_expressions=["迪尼娅"], pronunciation_overrides={"重要": "zhong4"}
    )
    report = lint_profile(make_profile(speaking_habits=habits))
    assert "profile.multi_char_pronunciation" in codes(report)


def test_over_long_replies_are_flagged():
    """Spec §27 lists over-long TTS chunks as a failure mode of its own."""
    from cvai_types import LLMSettings

    report = lint_profile(make_profile(llm=LLMSettings(max_chars_per_reply=800)))
    assert "profile.long_replies" in codes(report)


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------


def test_init_writes_a_profile_that_parses(tmp_path):
    """An `init` that produces a file the loader rejects is worse than no template."""
    from cvai_core.cli.character_cli import main
    from cvai_core.loaders import FilesystemCharacterProvider

    assert main(["--profiles-dir", str(tmp_path), "init", "denia_cn", "--name", "迪尼娅"]) == 0
    profile = FilesystemCharacterProvider(tmp_path).get("denia_cn")
    assert profile.character_name == "迪尼娅"
    assert "迪尼娅" in profile.speaking_habits.frequent_expressions


def test_init_refuses_to_overwrite(tmp_path):
    from cvai_core.cli.character_cli import main

    assert main(["--profiles-dir", str(tmp_path), "init", "denia_cn"]) == 0
    assert main(["--profiles-dir", str(tmp_path), "init", "denia_cn"]) == 2
    assert main(["--profiles-dir", str(tmp_path), "init", "denia_cn", "--force"]) == 0


def test_lint_exits_non_zero_on_an_empty_template(tmp_path, capsys):
    """The template is valid YAML and an unusable character, and it says which."""
    from cvai_core.cli.character_cli import main

    main(["--profiles-dir", str(tmp_path), "init", "denia_cn"])
    assert main(["--profiles-dir", str(tmp_path), "lint", "denia_cn"]) == 1
    assert "no dialogue examples" in capsys.readouterr().out


def test_lint_strict_fails_on_warnings(tmp_path):
    import yaml
    from cvai_core.cli.character_cli import main

    profile = make_profile(world_knowledge=[])
    (tmp_path / "denia_cn.yaml").write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), allow_unicode=True),
        encoding="utf-8",
    )
    assert main(["--profiles-dir", str(tmp_path), "lint", "denia_cn"]) == 0
    assert main(["--profiles-dir", str(tmp_path), "lint", "denia_cn", "--strict"]) == 1


def test_lint_reports_an_unknown_character_clearly(tmp_path, capsys):
    from cvai_core.cli.character_cli import main

    assert main(["--profiles-dir", str(tmp_path), "lint", "nobody"]) == 2
    assert "not found" in capsys.readouterr().err


def test_list_shows_every_profile(tmp_path, capsys):
    from cvai_core.cli.character_cli import main

    main(["--profiles-dir", str(tmp_path), "init", "denia_cn"])
    main(["--profiles-dir", str(tmp_path), "init", "other_cn"])
    assert main(["--profiles-dir", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "denia_cn" in out and "other_cn" in out


@pytest.mark.parametrize("command", ["lint", "list"])
def test_commands_do_not_need_a_voice_pack(tmp_path, command):
    """The two halves stay separate: a character can be written before any audio
    exists, and checking her must not require a pack."""
    from cvai_core.cli.character_cli import main

    main(["--profiles-dir", str(tmp_path), "init", "denia_cn"])
    assert main(["--profiles-dir", str(tmp_path), command, *(["denia_cn"] if command == "lint" else [])]) in (0, 1)
