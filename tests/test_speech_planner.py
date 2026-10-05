"""Character Brain and Speech Planner (spec §11, §12).

The planner's job is to produce a line *and* its delivery, and the guards' job is to
make sure the delivery is something this character's voice pack can actually perform.
Both are tested against a fake LLM, because what matters here is the handling of what
comes back — including the ways it comes back wrong.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from cvai_core.interfaces.character import CharacterProvider
from cvai_core.interfaces.llm import LLMProvider
from cvai_speech_planner import (
    FALLBACK,
    PARSED,
    STRUCTURED,
    CharacterSpeechPlanner,
    apply_guards,
    render_system_prompt,
    speech_plan_json_schema,
    trim_history,
)
from cvai_types import (
    CharacterProfile,
    CharacterSpeechPlan,
    DialogueExample,
    LLMCapabilities,
    LLMMessage,
    LLMResponse,
    Role,
    SpeakingRate,
    SpeakingHabits,
    VoiceBinding,
    VolumeStyle,
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def make_profile(**overrides: Any) -> CharacterProfile:
    data: dict[str, Any] = {
        "character_id": "denia_cn",
        "character_name": "迪尼娅",
        "personality": ["表面疏离", "熟悉之后会流露关心"],
        "background": "她在旧城区长大。",
        "world_knowledge": ["旧城区在河的南岸"],
        "relationship_style": "对用户半信半疑，但愿意搭话。",
        "speaking_habits": SpeakingHabits(
            typical_sentence_length="short",
            frequent_expressions=["随你便"],
            verbal_tics=["哼"],
            punctuation_habits=["常用省略号"],
        ),
        "forbidden_behavior": ["不要使用网络流行语"],
        "dialogue_examples": [
            DialogueExample(user="在吗", character="哼，怎么了。", style="teasing", source="q1"),
            DialogueExample(character="随你便。", style="cold", source="q2"),
        ],
        "voice": VoiceBinding(voicepack_id="denia_cn", default_reference_style="neutral"),
        "available_styles": ["neutral", "soft", "teasing", "cold", "soft_teasing"],
    }
    data.update(overrides)
    return CharacterProfile(**data)


class FakeCharacters(CharacterProvider):
    def __init__(self, profile: CharacterProfile) -> None:
        self.profile = profile

    def get(self, character_id: str) -> CharacterProfile:
        return self.profile

    def list_ids(self) -> list[str]:
        return [self.profile.character_id]


class FakeLLM(LLMProvider):
    provider = "fake"

    def __init__(
        self,
        *,
        structured: dict | None = None,
        text: str = "",
        supports_structured: bool = True,
        structured_raises: bool = False,
    ) -> None:
        self.structured = structured
        self.text = text
        self.supports_structured = supports_structured
        self.structured_raises = structured_raises
        self.seen_messages: list[LLMMessage] = []

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            provider=self.provider,
            model="fake-1",
            supports_structured_output=self.supports_structured,
        )

    async def complete(self, messages, *, temperature=None, max_output_tokens=None):
        self.seen_messages = list(messages)
        return LLMResponse(content=self.text, model="fake-1", provider=self.provider)

    async def stream(self, messages, *, temperature=None, max_output_tokens=None):
        yield  # pragma: no cover - unused here

    async def complete_structured(self, messages, schema, *, temperature=None):
        self.seen_messages = list(messages)
        if self.structured_raises:
            raise RuntimeError("provider refused the schema")
        return self.structured or {}


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------


def test_system_prompt_carries_the_profile():
    prompt = render_system_prompt(
        make_profile(), examples=make_profile().dialogue_examples
    )
    assert "迪尼娅" in prompt
    assert "表面疏离" in prompt
    assert "旧城区在河的南岸" in prompt
    assert "随你便" in prompt
    # Original lines are the highest-value content in the prompt.
    assert "哼，怎么了。" in prompt


def test_world_knowledge_doubles_as_a_boundary():
    """Otherwise the model invents lore, which breaks the character instantly."""
    prompt = render_system_prompt(make_profile())
    assert "不要以肯定语气断言" in prompt


def test_universal_rules_cannot_be_removed_by_a_profile():
    prompt = render_system_prompt(make_profile(forbidden_behavior=[]))
    assert "不要提及人工智能" in prompt


def test_prompt_states_that_only_text_is_spoken():
    """A model that thinks the metadata is spoken writes stage directions into it."""
    prompt = render_system_prompt(make_profile())
    assert "只有 text 会被念出来" in prompt


def test_prompt_lists_only_styles_the_character_has():
    prompt = render_system_prompt(make_profile())
    assert "soft_teasing" in prompt
    assert "excited" not in prompt


def test_prompt_states_the_length_limit():
    profile = make_profile()
    prompt = render_system_prompt(profile)
    assert str(profile.llm.max_chars_per_reply) in prompt


def test_history_is_trimmed_from_the_front():
    history = [
        LLMMessage(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"m{i}")
        for i in range(20)
    ]
    trimmed = trim_history(history, max_turns=3)
    assert len(trimmed) == 6
    assert trimmed[-1].content == "m19"
    assert trim_history(history, 0) == []


def test_schema_is_exposed_for_constrained_decoding():
    schema = speech_plan_json_schema()
    assert "text" in schema["properties"]
    assert "emotion" in schema["properties"]
    assert "speaking_rate" in schema["properties"]


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_unavailable_style_is_mapped_onto_one_the_pack_can_perform():
    profile = make_profile()
    plan = CharacterSpeechPlan(text="随你便。", emotion="furious")
    guarded, report = apply_guards(plan, profile)

    assert guarded.emotion in profile.available_styles
    assert report.repairs


def test_character_specific_style_falls_back_to_its_core():
    profile = make_profile(available_styles=["neutral", "teasing"])
    plan = CharacterSpeechPlan(text="哼。", emotion="soft_teasing")
    guarded, _ = apply_guards(plan, profile)
    assert guarded.emotion == "teasing"


def test_overlong_replies_are_cut_at_a_sentence_boundary():
    """Cutting mid-clause sounds like a dropped connection."""
    profile = make_profile()
    profile = profile.model_copy(
        update={"llm": profile.llm.model_copy(update={"max_chars_per_reply": 20})}
    )
    plan = CharacterSpeechPlan(
        text="第一句话很短。第二句话也不长。第三句话就超出去了，应该被切掉。"
    )
    guarded, report = apply_guards(plan, profile)

    assert len(guarded.text) <= 20
    assert guarded.text.endswith("。")
    assert any("truncated" in r for r in report.repairs)


def test_overlong_replies_are_kept_whole_when_truncation_is_off():
    """A voice chat that must say every word the LLM wrote."""
    profile = make_profile()
    profile = profile.model_copy(update={"llm": profile.llm.model_copy(
        update={"max_chars_per_reply": 20, "truncate_long_replies": False})})
    text = "第一句话很短。第二句话也不长。第三句话就超出去了，但要完整保留。"
    guarded, report = apply_guards(CharacterSpeechPlan(text=text), profile)

    assert guarded.text == text
    assert not any("truncated" in r for r in report.repairs)
    assert any("kept whole" in w for w in report.warnings)


def test_narration_in_brackets_is_removed():
    plan = CharacterSpeechPlan(text="（她转过身）我不去。")
    guarded, report = apply_guards(plan, make_profile())
    assert guarded.text == "我不去。"
    assert report.repairs


def test_breaking_character_is_flagged_not_silently_accepted():
    plan = CharacterSpeechPlan(text="作为一个AI，我不能这么说。")
    _, report = apply_guards(plan, make_profile())
    assert any("breaks character" in w for w in report.warnings)


def test_fast_whisper_is_slowed_because_no_engine_renders_it():
    plan = CharacterSpeechPlan(
        text="小声点。",
        volume_style=VolumeStyle.WHISPER,
        speaking_rate=SpeakingRate.VERY_FAST,
    )
    guarded, report = apply_guards(plan, make_profile())
    assert guarded.speaking_rate is SpeakingRate.NORMAL
    assert report.repairs


def test_empty_text_is_replaced_and_warned_about():
    plan = CharacterSpeechPlan(text="（沉默）")
    guarded, report = apply_guards(plan, make_profile())
    assert guarded.text.strip()
    assert report.warnings


def test_a_clean_plan_is_left_alone():
    plan = CharacterSpeechPlan(text="哼，怎么了。", emotion="teasing")
    guarded, report = apply_guards(plan, make_profile())
    assert guarded.text == plan.text
    assert guarded.emotion == "teasing"
    assert report.clean


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


def test_structured_output_is_the_preferred_path():
    profile = make_profile()
    llm = FakeLLM(
        structured={
            "text": "哼，怎么突然想起我了。",
            "emotion": "soft_teasing",
            "emotion_intensity": 0.45,
            "speaking_rate": "slightly_slow",
            "ending_style": "falling",
            "direction_note": "调侃但不刻薄",
        }
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.source == STRUCTURED
    assert not result.degraded
    assert result.plan.text == "哼，怎么突然想起我了。"
    assert result.plan.emotion == "soft_teasing"
    assert result.plan.emotion_intensity == pytest.approx(0.45)


def test_json_is_parsed_out_of_a_code_fence():
    profile = make_profile()
    payload = {"text": "随你便。", "emotion": "cold"}
    llm = FakeLLM(
        supports_structured=False,
        text="好的，这是我的回答：\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.source == PARSED
    assert result.plan.text == "随你便。"
    assert result.plan.emotion == "cold"


def test_plain_text_still_produces_a_speakable_plan():
    """A conversation turn must not fail because the provider ignored the format."""
    profile = make_profile()
    llm = FakeLLM(supports_structured=False, text="哼，我才没等你。")
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.source == FALLBACK
    assert result.degraded
    assert result.plan.text == "哼，我才没等你。"
    assert result.plan.emotion == profile.voice.default_reference_style


def test_structured_failure_falls_through_to_parsing():
    profile = make_profile()
    llm = FakeLLM(
        structured_raises=True,
        text=json.dumps({"text": "算了。", "emotion": "cold"}, ensure_ascii=False),
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.source == PARSED
    assert result.plan.text == "算了。"


def test_unknown_fields_do_not_lose_a_good_plan():
    """Models add keys. Rejecting the line over one is a bad trade."""
    profile = make_profile()
    llm = FakeLLM(
        structured={"text": "嗯。", "emotion": "neutral", "confidence": 0.91, "notes": "x"}
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))
    assert result.plan.text == "嗯。"


def test_a_bad_enum_value_costs_the_field_not_the_line():
    profile = make_profile()
    llm = FakeLLM(
        structured={"text": "我知道了。", "emotion": "cold", "speaking_rate": "sprinting"}
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.plan.text == "我知道了。"
    assert result.plan.emotion == "cold"
    assert result.plan.speaking_rate is SpeakingRate.NORMAL


def test_guards_run_on_planner_output():
    profile = make_profile()
    llm = FakeLLM(structured={"text": "（叹气）随你便。", "emotion": "furious"})
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "在吗"))

    assert result.plan.text == "随你便。"
    assert result.plan.emotion in profile.available_styles
    assert result.guards.repairs


def test_dialogue_examples_reach_the_prompt():
    profile = make_profile()
    llm = FakeLLM(structured={"text": "嗯。"})
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    run(planner.plan("denia_cn", "在吗"))

    system = llm.seen_messages[0]
    assert system.role is Role.SYSTEM
    assert "哼，怎么了。" in system.content


def test_history_reaches_the_prompt_in_order():
    profile = make_profile()
    llm = FakeLLM(structured={"text": "嗯。"})
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    history = [
        LLMMessage(role=Role.USER, content="昨天那件事"),
        LLMMessage(role=Role.ASSISTANT, content="我记得。"),
    ]
    run(planner.plan("denia_cn", "后来呢", history=history))

    contents = [m.content for m in llm.seen_messages]
    assert contents[1:3] == ["昨天那件事", "我记得。"]
    assert contents[-1] == "后来呢"


def test_only_the_text_goes_back_into_history():
    """Feeding the stage directions back teaches the model to talk about them."""
    from cvai_speech_planner import plan_to_messages

    plan = CharacterSpeechPlan(text="随你便。", emotion="cold", direction_note="冷淡")
    messages = plan_to_messages(plan)
    assert len(messages) == 1
    assert messages[0].content == "随你便。"
    assert "冷淡" not in messages[0].content


def test_plan_projects_onto_engine_neutral_controls():
    """The handoff to the voice stack: plan → StyleControls → TTSRequest."""
    profile = make_profile()
    llm = FakeLLM(
        structured={
            "text": "小声一点。",
            "emotion": "soft",
            "speaking_rate": "slow",
            "volume_style": "whisper",
            "reference_style": "soft",
        }
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "他们睡了吗"))

    controls = result.plan.to_style_controls()
    assert controls.effective_reference_style == "soft"
    assert controls.speed_factor() == pytest.approx(0.85)
    assert controls.volume_style is VolumeStyle.WHISPER


def test_planner_and_normalizer_compose_into_synthesizable_chunks():
    """End to end for Milestone 8: plan → normalize → chunk."""
    from cvai_text_normalizer import ChineseTextNormalizer, SpeechChunker

    profile = make_profile()
    llm = FakeLLM(
        structured={
            "text": "会议定在2026年3月17日，别迟到。",
            "emotion": "cold",
            "speaking_rate": "normal",
        }
    )
    planner = CharacterSpeechPlanner(llm, FakeCharacters(profile))
    result = run(planner.plan("denia_cn", "什么时候开会"))

    normalizer = ChineseTextNormalizer.from_character(profile)
    normalized = normalizer.normalize(result.plan.text)
    chunks = SpeechChunker(controls=result.plan.to_style_controls()).split(normalized.text)

    assert chunks
    assert not any(ch.isdigit() for chunk in chunks for ch in chunk.text)
    assert all(c.controls.emotion == "cold" for c in chunks)
    assert chunks[-1].is_final
