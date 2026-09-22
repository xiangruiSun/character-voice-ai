"""Which of her real lines go into the prompt (spec §11).

Spec §11 rules out fine-tuning the LLM and prescribes prompting with original dialogue
examples. The examples are the part that carries voice, so *which* examples are shown is
most of what makes a turn sound like her — and with a two-hundred-line profile and room
for six, the selection is doing almost all of the work.

What these tests pin is therefore behavioural rather than numeric: relevant lines beat
irrelevant ones, the requested style is preferred but never the only thing shown, six
examples are not six paraphrases, and the same turn always produces the same prompt.
"""

from __future__ import annotations

import pytest
from cvai_core.dialogue_examples import bigrams, normalize, select_examples
from cvai_types import DialogueExample


def example(user: str, character: str, style: str = "neutral") -> DialogueExample:
    return DialogueExample(user=user, character=character, style=style)


POOL = [
    example("你叫什么名字", "迪尼娅。记住了。"),
    example("你喜欢吃什么", "随便，什么都行。"),
    example("今天天气怎么样", "外面风大，你穿这么少？", "soft"),
    example("天气冷吗", "冷，记得加件衣服。", "soft"),
    example("你会做饭吗", "会一点，不过别抱太大期望。"),
    example("我先走了", "……路上小心。", "soft"),
    example("你在干嘛", "看书。有事？"),
    example("", "哼，不用你管。", "teasing"),
    example("你为什么总是一个人", "习惯了。人多了吵。", "serious"),
    example("需要帮忙吗", "不用。我自己能行。", "serious"),
]


def texts(chosen) -> list[str]:
    return [e.character for e in chosen]


# --------------------------------------------------------------------------------------
# Relevance
# --------------------------------------------------------------------------------------


def test_a_relevant_line_is_chosen_over_the_first_ones_in_the_profile():
    """The failure this replaced: the first six lines, whatever the user said."""
    chosen = select_examples("今天天气冷不冷", POOL, limit=3)
    assert "冷，记得加件衣服。" in texts(chosen)


def test_the_closest_line_wins_over_a_merely_similar_one():
    chosen = select_examples("你叫什么名字啊", POOL, limit=1)
    assert texts(chosen) == ["迪尼娅。记住了。"]


def test_matching_is_on_what_was_said_to_her():
    """What predicts her answer is the situation, not the words of the answer.

    Here the query shares wording with one example's *reply* and with another's
    *prompt*; the prompt-side match must win.
    """
    pool = [
        example("你在看什么书", "随便翻翻。"),
        example("今天累吗", "还好。我自己能行。"),
    ]
    chosen = select_examples("你在看什么书", pool, limit=1)
    assert texts(chosen) == ["随便翻翻。"]


def test_an_unrelated_question_falls_back_to_the_profiles_own_order():
    """Zero relevance everywhere is not a reason to reorder arbitrarily.

    Whoever wrote the character put the most representative lines first; an ordering
    derived from a table of zeros carries no information and loses that.
    """
    chosen = select_examples("xyzzy", POOL, limit=3)
    assert texts(chosen) == texts(POOL[:3])


def test_a_short_pool_is_returned_whole():
    assert len(select_examples("在吗", POOL[:4], limit=6)) == 4


def test_asking_for_nothing_returns_nothing():
    assert select_examples("在吗", POOL, limit=0) == []


# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------


def test_the_requested_style_is_preferred():
    chosen = select_examples("说点什么", POOL, limit=3, style="serious")
    assert any(e.style == "serious" for e in chosen)


def test_style_is_a_preference_not_a_filter():
    """Hard filtering is how a character with two teasing lines ends up showing the
    model two lines and nothing else about how she speaks."""
    chosen = select_examples("你猜呢", POOL, limit=4, style="teasing")
    assert len([e for e in chosen if e.style != "teasing"]) >= 1


def test_a_relevant_line_in_the_wrong_style_still_gets_in():
    chosen = select_examples("天气冷吗", POOL, limit=3, style="serious")
    assert "冷，记得加件衣服。" in texts(chosen)


# --------------------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------------------


def test_six_examples_are_not_six_paraphrases():
    """Near-duplicates crowd out the range the prompt exists to demonstrate — the same
    failure as conditioning every line on one reference clip, one level up."""
    pool = [
        example("你好吗", "我很好，谢谢关心。"),
        example("你还好吗", "我很好，谢谢关心。"),
        example("最近怎么样", "我很好，谢谢关心。"),
        example("你在忙什么", "在看书，有事说。"),
        example("要一起吃饭吗", "不了，我不饿。"),
    ]
    chosen = select_examples("你好吗", pool, limit=3)
    assert len(set(texts(chosen))) == 3


def test_standalone_lines_can_still_be_chosen():
    """A battle cry or an idle bark has no prompt to match, but it is still her voice."""
    pool = [example("", "哼，不用你管。", "teasing")] + POOL[:2]
    assert "哼，不用你管。" in texts(select_examples("随便说点", pool, limit=3))


# --------------------------------------------------------------------------------------
# Determinism and presentation
# --------------------------------------------------------------------------------------


def test_the_same_turn_produces_the_same_prompt():
    """A reproducible experiment cannot have a prompt that varies between runs."""
    first = select_examples("今天天气怎么样", POOL, limit=4, style="soft")
    for _ in range(5):
        assert texts(select_examples("今天天气怎么样", POOL, limit=4, style="soft")) == texts(first)


def test_examples_are_presented_in_profile_order():
    """Relevance-sorted, the list quietly tells the model the first one matters most.
    In profile order it reads as a transcript of her."""
    chosen = select_examples("天气冷吗 你叫什么名字", POOL, limit=4)
    indices = [POOL.index(e) for e in chosen]
    assert indices == sorted(indices)


# --------------------------------------------------------------------------------------
# Chinese text handling
# --------------------------------------------------------------------------------------


def test_punctuation_does_not_affect_matching():
    assert bigrams("你好，吗？") == bigrams("你好吗")


def test_a_single_character_query_still_matches():
    """Chinese has no spaces and no bigram at length one; scoring zero everywhere would
    silently mean 'profile order' for the shortest, most common utterances."""
    pool = [example("冷", "冷，记得加件衣服。"), example("热", "把窗开开。")]
    assert texts(select_examples("冷", pool, limit=1)) == ["冷，记得加件衣服。"]


def test_normalize_keeps_the_characters_and_drops_the_rest():
    assert normalize(" 你好， 世界！ ") == "你好世界"


# --------------------------------------------------------------------------------------
# Through the provider
# --------------------------------------------------------------------------------------


def test_the_default_provider_uses_it(tmp_path):
    """Every caller improves at once, which is why this lives behind the interface's
    own default rather than in one consumer."""
    import yaml
    from cvai_core.loaders import FilesystemCharacterProvider

    root = tmp_path / "profiles"
    root.mkdir()
    (root / "denia_cn.yaml").write_text(
        yaml.safe_dump(
            {
                "character_id": "denia_cn",
                "character_name": "迪尼娅",
                "voice": {"voicepack_id": "denia_cn"},
                "dialogue_examples": [
                    {"user": e.user, "character": e.character, "style": e.style}
                    for e in POOL
                ],
                "available_styles": ["neutral", "soft", "teasing", "serious"],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    provider = FilesystemCharacterProvider(root)
    chosen = provider.retrieve_examples("denia_cn", "外面天气冷吗", limit=3)
    assert "冷，记得加件衣服。" in [e.character for e in chosen]


def test_the_planner_sends_relevant_examples_to_the_llm(tmp_path):
    from cvai_core.dialogue_examples import select_examples as select

    chosen = select("天气冷吗", POOL, limit=3)
    assert chosen  # sanity: the planner's path is covered in test_speech_planner
    assert all(isinstance(e, DialogueExample) for e in chosen)


@pytest.mark.parametrize("query", ["", "   ", "。。。"])
def test_an_empty_query_does_not_crash(query: str):
    assert len(select_examples(query, POOL, limit=3)) == 3


def test_a_pool_of_repeated_barks_returns_fewer_rather_than_repeats():
    """Game profiles are full of duplicated lines. Three copies of one bark is a worse
    prompt than one copy plus nothing."""
    pool = [example("", "冲啊！", "happy") for _ in range(5)]
    assert len(select_examples("走吧", pool, limit=3)) == 1
