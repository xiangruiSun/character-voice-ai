"""Chinese text front-end: normalization (spec §13) and chunking (spec §14).

The test sentences in `configs/benchmarks/sentences_zh_v1.yaml` include a whole
category for this — numbers, dates, percentages, Latin tokens — because a front-end
mistake is indistinguishable from a bad acoustic model when you only hear the output.
These tests pin the readings so that when the benchmark sounds wrong, the cause is not
here.
"""

from __future__ import annotations

import pytest
from cvai_text_normalizer import (
    ChineseTextNormalizer,
    ChunkerConfig,
    LatinConfig,
    LatinPolicy,
    NormalizerConfig,
    SpeechChunker,
    decimal_to_chinese,
    digit_by_digit,
    ends_completely,
    estimate_chunks,
    integer_to_chinese,
    normalize_latin,
    time_to_chinese,
)
from cvai_text_normalizer.numbers import apply_liang

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "零"),
        (2, "二"),
        (9, "九"),
        # 10 is 十, never 一十 — the most common giveaway of a naive implementation.
        (10, "十"),
        (11, "十一"),
        (20, "二十"),
        (100, "一百"),
        # …but 110 keeps its 一十.
        (110, "一百一十"),
        (1250, "一千二百五十"),
        (1000, "一千"),
        # Zero insertion across a group boundary.
        (10001, "一万零一"),
        (100001, "十万零一"),
        (110000, "十一万"),
        (1000000, "一百万"),
        (100000000, "一亿"),
        (-42, "负四十二"),
    ],
)
def test_integer_readings(value: int, expected: str):
    assert integer_to_chinese(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("3.14", "三点一四"),
        ("78.5", "七十八点五"),
        ("0.5", "零点五"),
        ("-1.25", "负一点二五"),
        ("12", "十二"),
    ],
)
def test_decimal_readings(value: str, expected: str):
    """Cardinal before the point, digit by digit after it."""
    assert decimal_to_chinese(value) == expected


def test_digit_by_digit():
    assert digit_by_digit("2026") == "二零二六"
    assert digit_by_digit("138-0013") == "一三八杠零零一三"


@pytest.mark.parametrize(
    "reading,following,expected",
    [
        ("二", "个", "两"),
        ("二", "点", "两"),
        ("二", "小时", "两"),
        # Not final in the number, so it stays 二.
        ("十二", "个", "十二"),
        ("二十", "个", "二十"),
        # No measure word follows.
        ("二", "月", "两"),
        ("二", "", "二"),
    ],
)
def test_er_versus_liang(reading: str, following: str, expected: str):
    assert apply_liang(reading, following) == expected


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        ("14", "30", "十四点半"),
        ("2", "05", "两点零五分"),
        ("9", "00", "九点整"),
        ("23", "45", "二十三点四十五分"),
    ],
)
def test_time_readings(hour: str, minute: str, expected: str):
    assert time_to_chinese(hour, minute) == expected


# --------------------------------------------------------------------------------------
# Normalizer, end to end
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normalizer() -> ChineseTextNormalizer:
    return ChineseTextNormalizer()


@pytest.mark.parametrize(
    "text,expected_fragments,forbidden",
    [
        # Years are digit-by-digit, month and day are cardinal. Getting this the wrong
        # way round is the classic Chinese TTS tell.
        (
            "会议定在2026年3月17日下午两点半，别迟到。",
            ["二零二六年", "三月", "十七日"],
            ["二千零二十六"],
        ),
        ("这批一共1250件，其中37件要单独检查。", ["一千二百五十件", "三十七件"], []),
        ("成功率大概是78.5%，比上次高了20%。", ["百分之七十八点五", "百分之二十"], ["%"]),
        ("他说他用的是GPT-5.6，还装了CUDA 12.8。", ["G P T", "五点六", "十二点八"], ["负"]),
        ("把那个PDF发到我邮箱，编号是A7-3B。", ["P D F", "杠"], []),
        ("第3次了，2个人。", ["第三次", "两个人"], []),
        ("价格¥1250。", ["一千二百五十元"], ["¥"]),
        ("温度25°C。", ["二十五摄氏度"], []),
        ("请拨13800138000。", ["一三八零零一三八零零零"], []),
    ],
)
def test_normalization_cases(
    normalizer: ChineseTextNormalizer,
    text: str,
    expected_fragments: list[str],
    forbidden: list[str],
):
    result = normalizer.normalize(text)
    for fragment in expected_fragments:
        assert fragment in result.text, f"{fragment!r} missing from {result.text!r}"
    for fragment in forbidden:
        assert fragment not in result.text, f"{fragment!r} survived in {result.text!r}"


def test_no_digits_survive_a_normal_sentence(normalizer: ChineseTextNormalizer):
    result = normalizer.normalize("2026年3月17日，1250件，78.5%，第3次。")
    assert not any(ch.isdigit() for ch in result.text), result.text
    assert not result.warnings


def test_warnings_flag_leftover_digits():
    """A digit that survives is a reading the engine will guess at."""
    normalizer = ChineseTextNormalizer(
        NormalizerConfig(use_wetext=False, use_cn2an=False)
    )
    result = normalizer.normalize("型号")
    assert not result.warnings
    # Force a leftover by disabling the rules that would consume it.
    result = normalizer.normalize("IPv6")
    assert "IPv6" not in result.text or result.warnings


def test_emoji_and_markdown_are_stripped(normalizer: ChineseTextNormalizer):
    result = normalizer.normalize("**今天**天气不错 😊，真的。")
    assert "*" not in result.text
    assert "😊" not in result.text
    assert "今天天气不错" in result.text


def test_ellipsis_is_collapsed(normalizer: ChineseTextNormalizer):
    result = normalizer.normalize("我...我不知道......算了。")
    assert "..." not in result.text
    assert "……" in result.text
    # A five-second silence is not a pause.
    assert "…………" not in result.text


def test_ascii_punctuation_becomes_chinese(normalizer: ChineseTextNormalizer):
    result = normalizer.normalize("你好,世界!真的吗?")
    assert "，" in result.text and "！" in result.text and "？" in result.text
    assert "," not in result.text


def test_full_width_input_is_folded(normalizer: ChineseTextNormalizer):
    """Full-width digits otherwise slip past every rule."""
    result = normalizer.normalize("一共１２５０件")
    assert "一千二百五十" in result.text


def test_normalizer_reports_what_it_changed(normalizer: ChineseTextNormalizer):
    result = normalizer.normalize("2026年有78.5%的人用PDF。")
    assert result.changed
    rules = {r.rule for r in result.replacements}
    assert rules  # at minimum the latin pass and one number rule
    assert result.backend  # 'builtin', or the libraries when installed


def test_empty_and_whitespace_input(normalizer: ChineseTextNormalizer):
    assert normalizer.normalize("").text == ""
    assert normalizer.normalize("   \n  ").text == ""


# --------------------------------------------------------------------------------------
# Latin policy
# --------------------------------------------------------------------------------------


def test_latin_policy_keep_leaves_acronyms_alone():
    text, _ = normalize_latin(
        "用PDF发给我", LatinConfig(policy=LatinPolicy.KEEP, lexicon={})
    )
    assert "PDF" in text


def test_latin_policy_spell_separates_letters():
    text, _ = normalize_latin(
        "用PDF发给我", LatinConfig(policy=LatinPolicy.SPELL, lexicon={})
    )
    assert "P D F" in text


def test_latin_policy_drop_removes_it():
    text, _ = normalize_latin(
        "用PDF发给我", LatinConfig(policy=LatinPolicy.DROP, lexicon={})
    )
    assert "PDF" not in text and "P D F" not in text


def test_lexicon_wins_over_policy():
    text, _ = normalize_latin(
        "用PDF发给我",
        LatinConfig(policy=LatinPolicy.SPELL, lexicon={"PDF": "PDF文件"}),
    )
    assert "PDF文件" in text


def test_lowercase_words_are_not_spelled_out():
    """Spelling 'hello' letter by letter is much worse than leaving it."""
    text, _ = normalize_latin("说hello", LatinConfig(policy=LatinPolicy.SPELL, lexicon={}))
    assert "h e l l o" not in text


# --------------------------------------------------------------------------------------
# Character pronunciation
# --------------------------------------------------------------------------------------


def test_pronunciation_overrides_become_hints(repo_root_path):
    from cvai_core.loaders import load_character_profile

    profile = load_character_profile(
        repo_root_path / "characters" / "profiles" / "denia_cn.yaml"
    )
    profile = profile.model_copy(
        update={
            "speaking_habits": profile.speaking_habits.model_copy(
                update={"pronunciation_overrides": {"重": "chong2"}}
            )
        }
    )
    normalizer = ChineseTextNormalizer.from_character(profile)

    with_override = normalizer.normalize("这件事得重新做。")
    without = normalizer.normalize("今天天气不错。")

    assert with_override.pinyin_hints == {"重": "chong2"}
    assert without.pinyin_hints == {}


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def test_chunks_break_on_chinese_sentence_endings():
    chunks = estimate_chunks("今天外面风有点大。你出门记得多穿一件。别感冒了。")
    assert len(chunks) == 3
    assert chunks[0].endswith("。")


def test_short_pieces_are_merged_rather_than_spoken_alone():
    """哼。 as its own 0.4 s request costs more in seams than it saves in latency."""
    config = ChunkerConfig(first_chunk_min_chars=8, min_chars=8)
    chunks = estimate_chunks("哼。我才不在乎呢。", config)
    assert len(chunks) == 1


def test_the_first_chunk_may_be_short_because_it_sets_perceived_latency():
    chunks = estimate_chunks("好的。我这就去准备需要的东西，你稍微等我一下。")
    assert chunks[0] == "好的。"


def test_long_text_is_split_at_a_comma_rather_than_run_on():
    long_line = "如果你真的打算一个人过去，那至少把该带的都带上，别等出了事才想起来后悔，我可不想再替你收拾烂摊子。"
    chunks = estimate_chunks(long_line, ChunkerConfig(max_chars=30, hard_max_chars=50))
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_nothing_exceeds_the_hard_ceiling_even_without_punctuation():
    """Spec §27 lists extremely long TTS chunks as a failure mode."""
    runon = "啊" * 250
    chunks = estimate_chunks(runon, ChunkerConfig(max_chars=40, hard_max_chars=60))
    assert chunks
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert "".join(chunks) == runon


def test_no_text_is_lost_or_duplicated():
    text = "第一句话。第二句话，还有后半句。第三句！"
    chunks = estimate_chunks(text)
    rebuilt = "".join(chunks)
    # Punctuation stays with its sentence; the characters themselves all survive.
    assert rebuilt.replace(" ", "") == text.replace(" ", "")


def test_closing_quotes_stay_with_their_sentence():
    chunks = estimate_chunks("她说“我不去。”我也没办法。")
    assert chunks[0].endswith("”")


def test_a_conjunction_is_not_stranded_at_the_start_of_a_chunk():
    """Cutting before 但是 leaves the listener waiting across an audible gap."""
    chunks = estimate_chunks("我知道了。但是这件事没那么简单。")
    assert not any(chunk.startswith("但是") for chunk in chunks)


def test_streaming_yields_chunks_as_they_complete():
    """A chunk leaves as soon as it is complete, not when the reply ends."""
    chunker = SpeechChunker()
    emitted: list = []

    emitted += chunker.feed("今天外面")
    assert emitted == [], "an incomplete sentence must not be sent to TTS"

    emitted += chunker.feed("风有点大。")
    assert [c.text for c in emitted] == ["今天外面风有点大。"]

    emitted += chunker.feed("你出门记")
    emitted += chunker.feed("得多穿一件。")
    assert [c.text for c in emitted] == [
        "今天外面风有点大。",
        "你出门记得多穿一件。",
    ]

    # Nothing left to emit, but the turn still has to be marked as ended.
    assert chunker.flush() == []
    assert emitted[-1].is_final


def test_streaming_and_one_shot_agree():
    text = "第一句。第二句，带个逗号。第三句！"
    one_shot = estimate_chunks(text)

    chunker = SpeechChunker()
    streamed: list[str] = []
    for char in text:
        streamed += [chunk.text for chunk in chunker.feed(char)]
    streamed += [chunk.text for chunk in chunker.flush()]

    assert streamed == one_shot


def test_the_last_chunk_is_marked_final():
    chunks = SpeechChunker().split("第一句。第二句。")
    assert chunks[-1].is_final
    assert not any(chunk.is_final for chunk in chunks[:-1])


def test_chunk_indices_are_sequential():
    chunks = SpeechChunker().split("一句。两句。三句。四句。")
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_reset_prevents_an_interrupted_reply_leaking_into_the_next():
    """Barge-in drops a partial reply; its tail must not prefix the following one."""
    chunker = SpeechChunker()
    chunker.feed("这句话还没说完")
    assert chunker.pending
    chunker.reset()
    assert chunker.pending == ""
    assert [c.text for c in chunker.split("新的回答。")] == ["新的回答。"]


def test_chunks_carry_the_style_controls():
    from cvai_types import StyleControls

    controls = StyleControls(emotion="soft_teasing")
    chunks = SpeechChunker(controls=controls).split("你猜呢？我才不告诉你。")
    assert all(c.controls.emotion == "soft_teasing" for c in chunks)


def test_synthesis_text_requires_normalization_first():
    """A chunk that skipped the normalizer must not reach an engine."""
    chunks = SpeechChunker().split("今天天气不错。")
    with pytest.raises(ValueError):
        _ = chunks[0].synthesis_text

    normalized = chunks[0].model_copy(update={"normalized_text": "今天天气不错。"})
    assert normalized.synthesis_text == "今天天气不错。"


@pytest.mark.parametrize(
    "text,complete",
    [
        ("今天天气不错。", True),
        # 呢 is a sentence-final particle, so this is complete even with no 。
        ("你说呢", True),
        ("你说呢吧", True),
        ("今天天气", False),
        ("她说“别去了。”", True),
        ("我觉得", False),
    ],
)
def test_semantic_completeness_heuristic(text: str, complete: bool):
    assert ends_completely(text) is complete


def test_normalizer_and_chunker_compose():
    """The realistic path: normalize a reply, then chunk it for streaming."""
    normalizer = ChineseTextNormalizer()
    reply = "会议定在2026年3月17日下午两点半。记得带上那个PDF，编号A7-3B。"
    normalized = normalizer.normalize(reply)
    chunks = estimate_chunks(normalized.text)

    assert len(chunks) >= 2
    assert not any(ch.isdigit() for chunk in chunks for ch in chunk)
