"""The scripted LLM, and the offline path it makes possible.

Every other part of this project could already be exercised with no key and no network —
the benchmark has a mock engine, the preprocessing stages report their own absence, the
DSP is real. The LLM was the last thing that turned "try it" into "get an API key first",
which is a poor first five minutes for a project whose claim is that the machinery works
before any model does.

What matters about a development provider is not that it answers well but that it answers
*honestly* and on the system's **primary** path: one that silently pushes the planner onto
its parse-and-repair fallback teaches you that the fallback is normal, and then the day a
real provider degrades, nothing looks wrong.
"""

from __future__ import annotations

import asyncio

from cvai_core.registry import build_llm_provider
from cvai_llm_providers import MockLLMProvider
from cvai_speech_planner import CharacterSpeechPlanner, render_system_prompt
from cvai_speech_planner.planner import STRUCTURED
from cvai_types import LLMMessage, Role


def run(coro):
    return asyncio.run(coro)


def messages(system: str = "", user: str = "在吗") -> list[LLMMessage]:
    out = []
    if system:
        out.append(LLMMessage(role=Role.SYSTEM, content=system))
    out.append(LLMMessage(role=Role.USER, content=user))
    return out


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------


def test_it_answers_without_a_key_or_a_network():
    reply = run(MockLLMProvider().complete(messages()))
    assert reply.content.strip()
    assert reply.provider == "mock"


def test_the_same_turn_always_produces_the_same_reply():
    """A difference between two runs must never be this provider's doing."""
    provider = MockLLMProvider()
    first = run(provider.complete(messages(user="今天天气怎么样"))).content
    for _ in range(5):
        assert run(provider.complete(messages(user="今天天气怎么样"))).content == first


def test_different_turns_can_produce_different_replies():
    provider = MockLLMProvider()
    system = render_prompt()
    replies = {
        run(provider.complete(messages(system, user=text))).content
        for text in ("在吗", "今天天气怎么样", "你在干嘛", "我先走了", "你猜呢")
    }
    assert len(replies) > 1


def test_it_borrows_the_characters_own_lines():
    """The most a stand-in can honestly do: her register, nothing invented."""
    provider = MockLLMProvider()
    reply = run(provider.complete(messages(render_prompt()))).content
    assert reply in {e.character for e in _profile().dialogue_examples}


def test_a_prompt_without_examples_falls_back_to_bland_defaults():
    """A memorable scripted line gets mistaken for the model having said something."""
    reply = run(MockLLMProvider().complete(messages("你是一个角色。"))).content
    assert reply.strip()


def test_a_pinned_reply_wins():
    provider = MockLLMProvider(reply="哼，随你便。")
    assert run(provider.complete(messages(render_prompt()))).content == "哼，随你便。"


def test_streaming_reassembles_into_the_same_reply():
    provider = MockLLMProvider(chunk_size=2)

    async def collect():
        deltas = []
        async for chunk in provider.stream(messages(render_prompt())):
            deltas.append(chunk.delta)
        return "".join(deltas)

    assert run(collect()) == run(provider.complete(messages(render_prompt()))).content


def test_streaming_ends_with_a_final_chunk():
    provider = MockLLMProvider()

    async def last():
        chunk = None
        async for chunk in provider.stream(messages()):
            pass
        return chunk

    final = run(last())
    assert final.is_final and final.finish_reason == "stop"


# --------------------------------------------------------------------------------------
# Structured output — the planner's primary path
# --------------------------------------------------------------------------------------


def test_it_supports_structured_output():
    assert MockLLMProvider().capabilities().supports_structured_output


def test_the_structured_plan_is_complete_enough_to_perform():
    plan = run(MockLLMProvider().complete_structured(messages(render_prompt()), {}))
    assert plan["text"].strip()
    assert plan["emotion"]
    assert plan["speaking_rate"] == "normal"


def test_the_style_comes_from_the_line_it_borrowed():
    """Carrying it through is what makes `--audition` vary instead of performing
    everything neutral, which would make the audition useless for its one purpose."""
    provider = MockLLMProvider()
    prompt = render_prompt()
    styles = {
        run(provider.complete_structured(messages(prompt, user=text), {}))["emotion"]
        for text in ("在吗", "别急", "你猜呢", "我先走了", "今天天气怎么样", "我今天有点累")
    }
    assert len(styles) > 1


def test_a_style_the_character_cannot_perform_is_not_requested():
    """The profile's available_styles is the contract with the voice pack; a plan that
    breaks it silently degrades to neutral at retrieval time."""
    provider = MockLLMProvider()
    profile = _profile()
    plan = run(provider.complete_structured(messages(render_prompt()), {}))
    assert plan["emotion"] in profile.available_styles


def test_the_direction_note_says_it_is_not_a_model():
    """Anyone reading a run record has to be able to see that no model was involved."""
    plan = run(MockLLMProvider().complete_structured(messages(), {}))
    assert "mock" in plan["direction_note"].lower()


def test_the_planner_uses_its_primary_path_with_it(app_config):
    """A development provider that pushes the system onto its degraded path teaches you
    the degraded path is normal."""
    from cvai_core.interfaces.character import CharacterProvider

    profile = _profile()

    class Characters(CharacterProvider):
        def get(self, character_id):
            return profile

        def list_ids(self):
            return [profile.character_id]

    planner = CharacterSpeechPlanner(MockLLMProvider(), Characters())
    result = run(planner.plan(profile.character_id, "今天天气怎么样"))
    assert result.source == STRUCTURED
    assert not result.degraded
    assert result.plan.text.strip()


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def test_it_is_reachable_by_configuration(app_config):
    """`CVAI_LLM=mock` has to actually select it — the whole point is one env var."""
    provider = build_llm_provider(app_config, "mock")
    assert provider.provider == "mock"


def test_the_demo_character_exists_and_is_playable(repo_root_path):
    """`make demo` builds the pack; without a profile there is nothing to talk to, and
    the offline path stops one step short of working."""
    from cvai_core.loaders import FilesystemCharacterProvider

    provider = FilesystemCharacterProvider(repo_root_path / "characters" / "profiles")
    profile = provider.get("demo_zh")
    assert profile.voice.voicepack_id == "demo_zh"
    assert profile.voice.preferred_engine == "mock"
    assert len(profile.dialogue_examples) >= 6


def test_the_demo_character_passes_its_own_linter(repo_root_path):
    from cvai_core.character_lint import lint_profile
    from cvai_core.loaders import FilesystemCharacterProvider

    provider = FilesystemCharacterProvider(repo_root_path / "characters" / "profiles")
    report = lint_profile(provider.get("demo_zh"))
    assert report.ok, [issue.message for issue in report.errors()]


def test_every_declared_demo_style_has_examples(repo_root_path):
    """The audition walks available_styles; a style with no examples is a silent gap."""
    from collections import Counter

    from cvai_core.loaders import FilesystemCharacterProvider

    provider = FilesystemCharacterProvider(repo_root_path / "characters" / "profiles")
    profile = provider.get("demo_zh")
    counts = Counter(example.style for example in profile.dialogue_examples)
    for style in profile.available_styles:
        assert counts[style] >= 2, style


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _profile():
    from cvai_core.loaders import load_character_profile
    from cvai_core.paths import repo_root

    return load_character_profile(
        repo_root() / "characters" / "profiles" / "demo_zh.yaml"
    )


def render_prompt() -> str:
    profile = _profile()
    return render_system_prompt(profile, examples=profile.dialogue_examples)
