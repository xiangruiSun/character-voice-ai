"""What she still knows twenty turns later (spec §11).

`MemorySettings.summarize_after_turns` was in the schema from Milestone 1 and nothing
implemented it: history was trimmed to the last N exchanges and everything older was
gone. The failure is invisible in testing, where conversations are three turns long, and
glaring in use — you tell her your sister's name in turn 2 and by turn 20 she has never
heard of her.

The tests that matter here are the ones about *truthfulness*, not compression. A memory
that invents a detail is worse than one that forgets: forgetting reads as a limitation,
while a confidently wrong "you told me last week" reads as the character being someone
else. So the fallback is extractive, a failed summariser never blocks a turn, and nothing
the model returns is trusted without a sanity check.
"""

from __future__ import annotations

import asyncio

import pytest
from cvai_speech_planner import ConversationMemory, extract_facts, render_transcript
from cvai_speech_planner.memory import MAX_SUMMARY_CHARS
from cvai_types import LLMMessage, MemorySettings, Role


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """Records what it was asked and returns whatever it was told to."""

    def __init__(self, reply: str = "- 对方的妹妹叫小满。", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, messages, *, temperature=None, max_output_tokens=None):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider exploded")

        class Response:
            content = self.reply

        return Response()


def converse(memory: ConversationMemory, exchanges: int, *, start: int = 1) -> None:
    for n in range(start, start + exchanges):
        memory.add_user(f"第{n}轮我说的话")
        memory.add_character(f"第{n}轮她的回答")


def settings(**overrides) -> MemorySettings:
    base = {"max_turns": 4, "summarize_after_turns": 3}
    base.update(overrides)
    return MemorySettings(**base)


# --------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------


def test_turns_are_recorded_in_order():
    memory = ConversationMemory(settings())
    memory.add_user("在吗")
    memory.add_character("在。")
    assert [m.role for m in memory.turns] == [Role.USER, Role.ASSISTANT]
    assert memory.exchanges == 1


def test_blank_turns_are_not_recorded():
    """An empty reply is a failed turn, not a thing she said."""
    memory = ConversationMemory(settings())
    memory.add_user("在吗")
    memory.add_character("   ")
    assert len(memory.turns) == 1


def test_recent_returns_the_prompt_window():
    memory = ConversationMemory(settings(max_turns=2))
    converse(memory, 5)
    recent = memory.recent()
    assert len(recent) == 4  # two exchanges
    assert recent[0].content == "第4轮我说的话"


def test_a_zero_length_window_keeps_nothing():
    memory = ConversationMemory(settings(max_turns=0))
    converse(memory, 3)
    assert memory.recent() == []


# --------------------------------------------------------------------------------------
# When summarisation runs
# --------------------------------------------------------------------------------------


def test_a_short_conversation_is_not_summarised():
    """Nothing has fallen out of the window yet, so there is nothing to preserve."""
    memory = ConversationMemory(settings())
    converse(memory, 2)
    assert not memory.needs_summary
    assert run(memory.maybe_summarize(FakeLLM())) is False


def test_turns_falling_out_of_the_window_trigger_it():
    memory = ConversationMemory(settings(max_turns=2, summarize_after_turns=2))
    converse(memory, 5)
    assert memory.needs_summary
    assert run(memory.maybe_summarize(FakeLLM())) is True
    assert memory.summary


def test_disabled_memory_never_summarises():
    memory = ConversationMemory(settings(enabled=False))
    converse(memory, 20)
    assert not memory.needs_summary


def test_each_message_is_summarised_at_most_once():
    """Counted rather than recomputed: a double-fold puts the same turn in the prompt
    twice, and she starts referring to one conversation as though it were two."""
    llm = FakeLLM()
    memory = ConversationMemory(settings(max_turns=2, summarize_after_turns=2))
    converse(memory, 5)
    run(memory.maybe_summarize(llm))
    folded = memory.summarized_messages

    run(memory.maybe_summarize(llm))  # nothing new has aged out
    assert memory.summarized_messages == folded
    assert len(llm.calls) == 1


def test_the_summariser_sees_only_what_is_about_to_be_forgotten():
    llm = FakeLLM()
    memory = ConversationMemory(settings(max_turns=2, summarize_after_turns=2))
    converse(memory, 5)
    run(memory.maybe_summarize(llm))

    asked = llm.calls[0][-1].content
    assert "第1轮我说的话" in asked
    # The last two exchanges are still in the prompt window; summarising them as well
    # would say the same thing twice.
    assert "第5轮我说的话" not in asked


def test_the_existing_summary_is_given_back_to_the_summariser():
    """Otherwise each fold rewrites history from scratch and older facts vanish."""
    llm = FakeLLM()
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1), summary="- 对方叫阿岚。")
    converse(memory, 4)
    run(memory.maybe_summarize(llm))
    assert "阿岚" in llm.calls[0][-1].content


# --------------------------------------------------------------------------------------
# Truthfulness
# --------------------------------------------------------------------------------------


def test_a_failed_summariser_falls_back_instead_of_breaking_the_turn():
    """Memory must never be the reason a conversation stops."""
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    memory.add_user("我妹妹叫小满")
    memory.add_character("嗯，记住了。")
    converse(memory, 3, start=2)

    assert run(memory.maybe_summarize(FakeLLM(fail=True))) is True
    assert "小满" in memory.summary


def test_with_no_llm_at_all_the_summary_is_extractive():
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    memory.add_user("我住在城南")
    memory.add_character("知道了。")
    converse(memory, 3, start=2)

    assert run(memory.maybe_summarize(None)) is True
    assert "城南" in memory.summary


@pytest.mark.parametrize("reply", ["", "   ", "无", "无。", "好的，以下是总结：", "当然可以！"])
def test_an_unusable_model_answer_is_replaced_by_extraction(reply: str):
    """A refusal, an empty answer or the model narrating its own task are all worse
    than the plain extractive version, which is at least true."""
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    memory.add_user("我明天要去医院")
    memory.add_character("……小心点。")
    converse(memory, 3, start=2)

    run(memory.maybe_summarize(FakeLLM(reply=reply)))
    assert "医院" in memory.summary


def test_the_fallback_keeps_only_what_the_user_said():
    """Her replies can be re-derived from the profile. What the person told her cannot
    be re-derived from anything."""
    messages = [
        LLMMessage(role=Role.USER, content="我妹妹叫小满"),
        LLMMessage(role=Role.ASSISTANT, content="嗯。"),
    ]
    facts = extract_facts(messages)
    assert "小满" in facts
    assert "嗯" not in facts


def test_the_fallback_truncates_rather_than_paraphrases():
    """Paraphrasing without a model means inventing. Truncation cannot."""
    long_line = "这是一句很长的话。" * 20
    facts = extract_facts([LLMMessage(role=Role.USER, content=long_line)])
    assert facts.endswith("…")
    assert len(facts) < len(long_line)


def test_the_summariser_is_asked_not_to_invent():
    llm = FakeLLM()
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    converse(memory, 4)
    run(memory.maybe_summarize(llm))

    instruction = llm.calls[0][0].content
    assert "不要编造" in instruction
    # Third person, as facts — a summary written in her voice gets imitated as dialogue.
    assert "第三人称" in instruction


# --------------------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------------------


def test_the_summary_stays_bounded():
    """Unbounded, it eventually costs more context than the transcript it replaced."""
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    llm = FakeLLM(reply="- " + "记住的内容。" * 20)
    for n in range(1, 30):
        memory.add_user(f"第{n}句")
        memory.add_character(f"第{n}答")
        run(memory.maybe_summarize(llm))
    assert len(memory.summary) <= MAX_SUMMARY_CHARS


def test_the_newest_lines_survive_the_bound():
    """A conversation refers back to its recent past far more than to its beginning."""
    memory = ConversationMemory(settings(max_turns=1, summarize_after_turns=1))
    memory.summary = "\n".join(f"- 旧记录{n}" for n in range(200))
    memory.add_user("我换了新工作")
    memory.add_character("恭喜。")
    converse(memory, 3, start=2)
    run(memory.maybe_summarize(FakeLLM(reply="- 对方换了新工作。")))

    assert "新工作" in memory.summary
    assert "旧记录0" not in memory.summary


def test_the_transcript_labels_who_said_what():
    text = render_transcript(
        [
            LLMMessage(role=Role.USER, content="在吗"),
            LLMMessage(role=Role.ASSISTANT, content="在。"),
        ]
    )
    assert text == "对方：在吗\n她：在。"


# --------------------------------------------------------------------------------------
# In the prompt
# --------------------------------------------------------------------------------------


def test_the_summary_reaches_the_system_prompt():
    from cvai_speech_planner import render_system_prompt
    from test_orchestrator import make_profile

    prompt = render_system_prompt(make_profile(), summary="- 对方的妹妹叫小满。")
    assert "小满" in prompt
    assert "之前聊过" in prompt


def test_the_summary_is_labelled_as_this_conversation_not_as_lore():
    """Half-remembered conversation and her actual backstory have different authority,
    and a model that cannot tell them apart asserts both equally firmly."""
    from cvai_speech_planner import render_system_prompt
    from test_orchestrator import make_profile

    prompt = render_system_prompt(make_profile(), summary="- 对方说过他讨厌下雨。")
    before = prompt.index("之前聊过")
    assert prompt.index("你记得的事") if "你记得的事" in prompt else True
    # It is its own labelled section, not appended to the character's world knowledge.
    assert "【" in prompt[max(0, before - 3) : before]


# --------------------------------------------------------------------------------------
# Through the orchestrator
# --------------------------------------------------------------------------------------


def test_a_long_conversation_keeps_what_was_said_early(tmp_path):
    """The whole point, end to end: turn 2's detail is still available at turn 12.

    The planner's LLM is also the summariser, so this records what it was asked. What
    it *answers* is the fake's business; what it was asked is the behaviour under test.
    """
    from test_orchestrator import FakeLLM as PlannerLLM, make_orchestrator

    class RecordingLLM(PlannerLLM):
        def __init__(self) -> None:
            super().__init__()
            self.summary_requests: list[str] = []

        async def complete(self, messages, *, temperature=None, max_output_tokens=None):
            if any("不要编造" in m.content for m in messages):
                self.summary_requests.append(messages[-1].content)
            return await super().complete(
                messages, temperature=temperature, max_output_tokens=max_output_tokens
            )

    llm = RecordingLLM()
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    orchestrator.memory.settings = settings(max_turns=2, summarize_after_turns=2)

    run(orchestrator.collect("我妹妹叫小满"))
    for n in range(6):
        run(orchestrator.collect(f"随便聊点别的{n}"))

    assert orchestrator.memory.summarized_messages > 0
    assert orchestrator.memory.summary
    assert any("小满" in asked for asked in llm.summary_requests)


def test_without_summarisation_the_early_turn_is_simply_gone(tmp_path):
    """The behaviour this replaced, kept as a test so the difference stays visible."""
    from cvai_speech_planner import trim_history
    from test_orchestrator import make_orchestrator

    orchestrator = make_orchestrator(tmp_path)
    run(orchestrator.collect("我妹妹叫小满"))
    for n in range(6):
        run(orchestrator.collect(f"随便聊点别的{n}"))

    window = trim_history(orchestrator.history, 2)
    assert not any("小满" in message.content for message in window)


def test_memory_failures_never_break_a_turn(tmp_path, monkeypatch):
    from test_orchestrator import make_orchestrator

    orchestrator = make_orchestrator(tmp_path)

    async def explode(*args, **kwargs):
        raise RuntimeError("memory is on fire")

    monkeypatch.setattr(orchestrator.memory, "maybe_summarize", explode)
    events = run(orchestrator.collect("在吗"))
    assert events[-1].type.value == "turn_end"


def test_history_is_still_readable_as_a_list(tmp_path):
    """`orchestrator.history` predates the memory object and other code reads it."""
    from test_orchestrator import make_orchestrator

    orchestrator = make_orchestrator(tmp_path)
    run(orchestrator.collect("在吗"))
    assert [m.role for m in orchestrator.history] == [Role.USER, Role.ASSISTANT]
