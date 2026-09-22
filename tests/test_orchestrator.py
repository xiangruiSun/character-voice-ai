"""Conversation orchestrator and barge-in (Milestones 9-13, spec §15 and §16).

Everything here runs against fakes — a fake LLM, the mock TTS engine, a tiny reference
bank — because what is being tested is the *sequencing*: which states happen in which
order, what reaches the engine, and above all what happens when the user cuts the
character off mid-sentence.

Barge-in is the most heavily covered part. Spec §16 lists four things it must do (stop
playback, clear the queue, cancel pending TTS, optionally cancel the LLM) and each has a
test, because an interruption path that half-works is worse than none: the character
gets one more word in after being cut off, and the illusion goes with it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from cvai_conversation import (
    PLAN_MODE_STREAMING,
    ConversationOrchestrator,
    OrchestratorConfig,
)
from cvai_core.interfaces.character import CharacterProvider
from cvai_core.interfaces.llm import LLMProvider
from cvai_reference_retrieval import RuleBasedReferenceRetriever
from cvai_speech_planner import CharacterSpeechPlanner
from cvai_tts_providers import MockTTSProvider
from cvai_types import (
    AudioProperties,
    CharacterProfile,
    ConversationState,
    CoreStyle,
    InterruptReason,
    InterruptSignal,
    LLMCapabilities,
    LLMResponse,
    LLMStreamChunk,
    ReferenceBank,
    ReferenceSample,
    SessionConfig,
    SpeakingHabits,
    TurnEventType,
    VoiceBinding,
)


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


def make_profile() -> CharacterProfile:
    return CharacterProfile(
        character_id="denia_cn",
        character_name="迪尼娅",
        personality=["表面疏离"],
        speaking_habits=SpeakingHabits(typical_sentence_length="short"),
        voice=VoiceBinding(voicepack_id="denia_cn", default_reference_style="neutral"),
        available_styles=["neutral", "soft", "teasing"],
    )


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
        plan: dict | None = None,
        deltas: list[str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.plan = plan or {"text": "今天外面风有点大。你出门记得多穿一件。", "emotion": "soft"}
        self.deltas = deltas or ["今天外面", "风有点大。", "你出门记得", "多穿一件。"]
        self.delay = delay
        self.stream_cancelled = False

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(provider=self.provider, model="fake", supports_structured_output=True)

    async def complete(self, messages, *, temperature=None, max_output_tokens=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        return LLMResponse(content=self.plan["text"], model="fake", provider=self.provider)

    async def stream(self, messages, *, temperature=None, max_output_tokens=None):
        try:
            for delta in self.deltas:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield LLMStreamChunk(delta=delta)
            yield LLMStreamChunk(is_final=True, finish_reason="stop")
        except asyncio.CancelledError:
            self.stream_cancelled = True
            raise

    async def complete_structured(self, messages, schema, *, temperature=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        return dict(self.plan)


def make_retriever() -> RuleBasedReferenceRetriever:
    samples = [
        ReferenceSample(
            reference_id=f"{style}_{index:02d}",
            audio_path=f"references/{style}/{style}_{index:02d}.wav",
            transcript="示例台词。",
            style=style,
            core_style=CoreStyle(style),
            audio=AudioProperties(sample_rate=24000, channels=1, duration_s=4.0),
        )
        for style in ("neutral", "soft", "teasing")
        for index in (1, 2)
    ]
    return RuleBasedReferenceRetriever(
        ReferenceBank(voicepack_id="denia_cn", voicepack_version="0.1.0", samples=samples)
    )


def make_orchestrator(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    tts: Any | None = None,
    config: OrchestratorConfig | None = None,
    enable_barge_in: bool = True,
) -> ConversationOrchestrator:
    profile = make_profile()
    characters = FakeCharacters(profile)
    return ConversationOrchestrator(
        SessionConfig(session_id="s1", character_id=profile.character_id,
                      enable_barge_in=enable_barge_in),
        profile,
        planner=CharacterSpeechPlanner(llm or FakeLLM(), characters),
        tts=tts or MockTTSProvider(sample_rate=16000),
        retriever=make_retriever(),
        config=config,
        audio_root=tmp_path / "sessions",
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------------------
# A normal turn
# --------------------------------------------------------------------------------------


def test_a_turn_runs_the_whole_pipeline(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    events = run(orchestrator.collect("在吗"))

    kinds = [event.type for event in events]
    assert TurnEventType.PLAN in kinds
    assert TurnEventType.CHUNK in kinds
    assert TurnEventType.AUDIO in kinds
    assert kinds[-1] is TurnEventType.TURN_END

    audio = [e for e in events if e.type is TurnEventType.AUDIO]
    assert audio
    for event in audio:
        assert Path(event.audio_path).is_file()
        assert event.duration_s and event.duration_s > 0
        # Spec §23 applies to live conversation too: every rendered chunk says which
        # reference clip and which engine produced it.
        assert event.reference_id
        assert event.engine == "mock"


def test_states_follow_the_documented_order(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    events = run(orchestrator.collect("在吗"))

    states = [e.state for e in events if e.type is TurnEventType.STATE]
    assert states[0] is ConversationState.THINKING
    assert ConversationState.PLANNING in states
    assert ConversationState.SYNTHESIZING in states
    assert ConversationState.SPEAKING in states
    assert orchestrator.state is ConversationState.IDLE


def test_the_caption_arrives_before_any_audio(tmp_path: Path):
    """Text first is what makes the wait for speech feel short."""
    orchestrator = make_orchestrator(tmp_path)
    events = run(orchestrator.collect("在吗"))

    first_text = next(i for i, e in enumerate(events) if e.type is TurnEventType.TEXT_DELTA)
    first_audio = next(i for i, e in enumerate(events) if e.type is TurnEventType.AUDIO)
    assert first_text < first_audio


def test_long_replies_are_chunked_not_sent_whole(tmp_path: Path):
    llm = FakeLLM(
        plan={
            "text": "今天外面风有点大。你出门记得多穿一件。别又像上次那样感冒了。",
            "emotion": "neutral",
        }
    )
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(orchestrator.collect("说点什么"))

    chunks = [e for e in events if e.type is TurnEventType.CHUNK]
    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[-1].is_final


def test_short_sentences_are_merged_rather_than_each_becoming_a_request(tmp_path: Path):
    """Four-character sentences are not worth a TTS round trip each (spec §14)."""
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(orchestrator.collect("说点什么"))

    chunks = [e for e in events if e.type is TurnEventType.CHUNK]
    assert len(chunks) < 3
    assert "".join(c.text for c in chunks) == "第一句。第二句。第三句。"


def test_text_reaching_the_engine_is_normalized(tmp_path: Path):
    """Spec §13: raw LLM text is not appropriate TTS input."""
    llm = FakeLLM(plan={"text": "会议定在2026年3月17日，有78.5%的把握。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(orchestrator.collect("什么时候"))

    chunks = [e.text for e in events if e.type is TurnEventType.CHUNK]
    assert chunks
    assert not any(ch.isdigit() for text in chunks for ch in text)
    assert any("二零二六年" in text for text in chunks)


def test_history_accumulates_across_turns(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    run(orchestrator.collect("在吗"))
    run(orchestrator.collect("后来呢"))

    contents = [m.content for m in orchestrator.history]
    assert contents[0] == "在吗"
    assert contents[2] == "后来呢"
    assert len(orchestrator.turns) == 2


def test_the_plan_style_reaches_the_reference_choice(tmp_path: Path):
    llm = FakeLLM(plan={"text": "你猜呢？", "emotion": "teasing", "reference_style": "teasing"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(orchestrator.collect("在想什么"))

    audio = [e for e in events if e.type is TurnEventType.AUDIO]
    assert audio
    assert all(e.reference_id.startswith("teasing") for e in audio)


def test_submit_text_conforms_to_the_session_interface(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)

    async def drain():
        return [chunk async for chunk in orchestrator.submit_text("在吗")]

    chunks = run(drain())
    assert chunks
    assert all(chunk.normalized_text for chunk in chunks)
    assert chunks[-1].is_final


# --------------------------------------------------------------------------------------
# Streaming mode
# --------------------------------------------------------------------------------------


def test_streaming_mode_speaks_before_the_reply_is_finished(tmp_path: Path):
    """Spec §14: do not wait for the whole LLM response before TTS begins."""
    orchestrator = make_orchestrator(
        tmp_path, config=OrchestratorConfig(plan_mode=PLAN_MODE_STREAMING)
    )
    events = run(orchestrator.collect("在吗"))

    kinds = [e.type for e in events]
    # Audio for the first sentence happens before the last text delta arrives.
    first_audio = kinds.index(TurnEventType.AUDIO)
    last_delta = len(kinds) - 1 - kinds[::-1].index(TurnEventType.TEXT_DELTA)
    assert first_audio < last_delta

    assert [e for e in events if e.type is TurnEventType.AUDIO]
    # No plan in streaming mode — that is the trade it makes.
    assert TurnEventType.PLAN not in kinds


def test_streaming_mode_still_records_the_reply_in_history(tmp_path: Path):
    orchestrator = make_orchestrator(
        tmp_path, config=OrchestratorConfig(plan_mode=PLAN_MODE_STREAMING)
    )
    run(orchestrator.collect("在吗"))
    assert orchestrator.history[-1].content == "今天外面风有点大。你出门记得多穿一件。"


# --------------------------------------------------------------------------------------
# Barge-in (spec §16)
# --------------------------------------------------------------------------------------


async def _interrupt_after_first_audio(orchestrator, text: str):
    """Run a turn and interrupt as soon as the first audio chunk lands."""
    events = []
    async for event in orchestrator.run_turn(text):
        events.append(event)
        if event.type is TurnEventType.AUDIO and len(
            [e for e in events if e.type is TurnEventType.AUDIO]
        ) == 1:
            await orchestrator.interrupt(
                InterruptSignal(reason=InterruptReason.USER_SPEECH)
            )
    return events


def test_barge_in_stops_further_audio(tmp_path: Path):
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。第四句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(_interrupt_after_first_audio(orchestrator, "说点什么"))

    audio = [e for e in events if e.type is TurnEventType.AUDIO]
    assert len(audio) == 1, "the character kept talking after being cut off"


def test_barge_in_emits_interrupted_then_listening(tmp_path: Path):
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    events = run(_interrupt_after_first_audio(orchestrator, "说点什么"))

    states = [e.state for e in events if e.type is TurnEventType.STATE]
    assert ConversationState.INTERRUPTED in states
    assert orchestrator.state is ConversationState.LISTENING
    assert events[-1].type is TurnEventType.TURN_END
    assert events[-1].detail == "interrupted"


def test_the_interrupted_turn_is_marked_as_such(tmp_path: Path):
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)
    run(_interrupt_after_first_audio(orchestrator, "说点什么"))
    assert orchestrator.turns[-1].was_interrupted


def test_a_late_interrupt_does_not_kill_the_next_turn(tmp_path: Path):
    """The difference between barge-in working and the system randomly going mute."""
    orchestrator = make_orchestrator(tmp_path)

    async def scenario():
        first = await orchestrator.collect("在吗")
        stale_turn_id = first[0].turn_id
        # The signal arrives after the turn it was aimed at has already ended.
        handled = await orchestrator.interrupt(turn_id=stale_turn_id)
        second = await orchestrator.collect("后来呢")
        return handled, second

    handled, second = run(scenario())
    assert handled is False
    assert [e for e in second if e.type is TurnEventType.AUDIO]
    assert not orchestrator.turns[-1].was_interrupted


def test_interrupting_when_nothing_is_running_is_a_no_op(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    assert run(orchestrator.interrupt()) is False


def test_barge_in_can_be_disabled_for_a_session(tmp_path: Path):
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm, enable_barge_in=False)
    events = run(_interrupt_after_first_audio(orchestrator, "说点什么"))

    audio = [e for e in events if e.type is TurnEventType.AUDIO]
    assert len(audio) > 1
    assert not orchestrator.turns[-1].was_interrupted


def test_audio_that_finishes_after_the_interrupt_is_dropped(tmp_path: Path):
    """Otherwise the character gets one more word in after being cut off."""

    class SlowMock(MockTTSProvider):
        async def synthesize(self, request, output_path):
            await asyncio.sleep(0.05)
            return await super().synthesize(request, output_path)

    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm, tts=SlowMock(sample_rate=16000))

    async def scenario():
        events = []
        agen = orchestrator.run_turn("说点什么")
        async for event in agen:
            events.append(event)
            if event.type is TurnEventType.CHUNK and event.chunk_index == 0:
                # Interrupt while the first chunk is still being synthesized.
                await orchestrator.interrupt()
        return events

    events = run(scenario())
    assert not [e for e in events if e.type is TurnEventType.AUDIO]
    assert orchestrator.turns[-1].was_interrupted


def test_streaming_mode_can_also_be_interrupted(tmp_path: Path):
    orchestrator = make_orchestrator(
        tmp_path,
        llm=FakeLLM(deltas=["第一句。", "第二句。", "第三句。", "第四句。"]),
        config=OrchestratorConfig(plan_mode=PLAN_MODE_STREAMING),
    )
    events = run(_interrupt_after_first_audio(orchestrator, "说点什么"))
    assert len([e for e in events if e.type is TurnEventType.AUDIO]) == 1
    assert orchestrator.turns[-1].was_interrupted


def test_the_session_recovers_and_can_run_another_turn(tmp_path: Path):
    llm = FakeLLM(plan={"text": "第一句。第二句。第三句。", "emotion": "neutral"})
    orchestrator = make_orchestrator(tmp_path, llm=llm)

    async def scenario():
        await _interrupt_after_first_audio(orchestrator, "说点什么")
        return await orchestrator.collect("那继续")

    events = run(scenario())
    assert [e for e in events if e.type is TurnEventType.AUDIO]
    assert orchestrator.state is ConversationState.IDLE


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------


def test_a_synthesis_failure_ends_the_turn_with_an_error_not_a_crash(tmp_path: Path):
    from cvai_core.errors import SynthesisError

    class BrokenTTS(MockTTSProvider):
        async def synthesize(self, request, output_path):
            raise SynthesisError("engine exploded")

    orchestrator = make_orchestrator(tmp_path, tts=BrokenTTS())
    events = run(orchestrator.collect("在吗"))

    errors = [e for e in events if e.type is TurnEventType.ERROR]
    assert errors and "engine exploded" in errors[0].error
    # And the session is usable again afterwards.
    assert orchestrator.state is ConversationState.IDLE


def test_a_planner_failure_is_reported_and_recovered(tmp_path: Path):
    class BrokenLLM(FakeLLM):
        async def complete_structured(self, messages, schema, *, temperature=None):
            raise RuntimeError("provider down")

        async def complete(self, messages, *, temperature=None, max_output_tokens=None):
            raise RuntimeError("provider down")

    orchestrator = make_orchestrator(tmp_path, llm=BrokenLLM())
    events = run(orchestrator.collect("在吗"))

    assert [e for e in events if e.type is TurnEventType.ERROR]
    assert orchestrator.state is ConversationState.IDLE


def test_audio_is_written_under_the_session_and_turn(tmp_path: Path):
    orchestrator = make_orchestrator(tmp_path)
    events = run(orchestrator.collect("在吗"))
    audio = [e for e in events if e.type is TurnEventType.AUDIO][0]

    path = Path(audio.audio_path)
    assert "s1" in path.parts
    assert audio.turn_id in path.parts
