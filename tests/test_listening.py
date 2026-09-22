"""Microphone endpointing, barge-in and the voice loop (Milestone 11, spec §14-§16).

Everything here is synthetic audio: a sine burst for speech, low-level noise for a room,
silence for a pause. That is enough, because what is being tested is the *decision* layer
— when an utterance starts, when it has ended, when the user is talking over the
character — and those decisions are made from frame energy, not from anything linguistic.

Two behaviours get the most attention, both because they are the ones a user notices:

* **Not cutting the user off.** Ending an utterance early means the character answers
  half a sentence, which reads as not listening.
* **Barge-in.** Being talked over by a machine is the single fastest way to destroy the
  illusion spec §18 is aiming at, so a false negative here costs more than a false
  positive and the tests are written around that asymmetry.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
from cvai_conversation import (
    ListenerConfig,
    ListenerEvent,
    UtteranceDetector,
    VoiceLoop,
    floats_to_pcm16,
    pcm16_to_floats,
)
from cvai_core.interfaces.stt import SpeechToTextProvider
from cvai_types import ConversationState, STTCapabilities, Transcript, TurnEventType
from test_orchestrator import make_orchestrator, run

CONFIG = ListenerConfig()


# --------------------------------------------------------------------------------------
# Synthetic audio
# --------------------------------------------------------------------------------------


def frames(count: int, config: ListenerConfig = CONFIG) -> int:
    return count * int(config.sample_rate * config.frame_ms / 1000)


def tone(
    ms: float, *, amplitude: float = 0.3, config: ListenerConfig = CONFIG
) -> list[float]:
    """A 220 Hz burst. Stands in for a voiced Mandarin syllable."""
    count = int(config.sample_rate * ms / 1000)
    step = 2.0 * math.pi * 220.0 / config.sample_rate
    return [amplitude * math.sin(step * n) for n in range(count)]


def silence(ms: float, *, config: ListenerConfig = CONFIG) -> list[float]:
    return [0.0] * int(config.sample_rate * ms / 1000)


def room_noise(
    ms: float, *, amplitude: float = 0.002, config: ListenerConfig = CONFIG
) -> list[float]:
    """Deterministic low-level hiss — a quiet room, not a silent file."""
    count = int(config.sample_rate * ms / 1000)
    return [amplitude * ((n * 7919 % 2000) / 1000.0 - 1.0) for n in range(count)]


def feed(detector: UtteranceDetector, samples, **kwargs) -> list[ListenerEvent]:
    return detector.feed(samples, **kwargs)


# --------------------------------------------------------------------------------------
# Endpointing
# --------------------------------------------------------------------------------------


def test_one_loud_frame_is_not_speech():
    """A door closing is one frame. Requiring a run is what makes that not a turn."""
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    events = feed(detector, tone(CONFIG.frame_ms))
    assert events == []
    assert not detector.in_speech


def test_a_run_of_voiced_frames_starts_an_utterance():
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    events = feed(detector, tone(CONFIG.frame_ms * CONFIG.speech_start_frames))
    assert ListenerEvent.SPEECH_START in events
    assert detector.in_speech


def test_the_utterance_keeps_the_frames_before_the_decision():
    """Speech is declared three frames late; the audio must not start three frames late.

    Losing the first 60 ms of a Mandarin syllable is enough to turn 四 into 是 in the
    transcript, so the detector holds a short pre-roll and flushes it on start.
    """
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(CONFIG.frame_ms * CONFIG.speech_start_frames))
    # Every frame that triggered the start is in the buffer, not just the last one.
    assert detector.buffered_seconds >= (
        CONFIG.frame_ms * CONFIG.speech_start_frames / 1000.0
    )


def test_a_pause_for_breath_does_not_end_the_utterance():
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(400))
    events = feed(detector, silence(CONFIG.silence_end_ms - 200))
    assert ListenerEvent.SPEECH_END not in events
    assert detector.in_speech


def test_trailing_silence_ends_the_utterance():
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(600))
    events = feed(detector, silence(CONFIG.silence_end_ms + 100))
    assert ListenerEvent.SPEECH_END in events
    assert not detector.in_speech
    captured = detector.take_utterance()
    assert len(captured) > frames(10)


def test_a_cough_is_not_a_turn():
    """Shorter than `min_utterance_ms` of speech is discarded rather than transcribed."""
    config = ListenerConfig(min_utterance_ms=400)
    detector = UtteranceDetector(config)
    feed(detector, room_noise(400, config=config))
    feed(detector, tone(120, config=config))
    events = feed(detector, silence(config.silence_end_ms + 100, config=config))
    assert ListenerEvent.SPEECH_END not in events
    assert detector.take_utterance() == []


def test_a_monologue_is_cut_at_the_length_cap():
    """Better a transcribed 4 s than an unbounded buffer nothing ever transcribes."""
    config = ListenerConfig(max_utterance_s=1.5)
    detector = UtteranceDetector(config)
    feed(detector, room_noise(400, config=config))
    events = feed(detector, tone(3000, config=config))
    assert ListenerEvent.TOO_LONG in events
    assert ListenerEvent.SPEECH_END in events
    assert detector.take_utterance()


def test_a_second_utterance_works_after_the_first():
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(500))
    assert ListenerEvent.SPEECH_END in feed(detector, silence(CONFIG.silence_end_ms + 60))
    detector.take_utterance()

    feed(detector, tone(500))
    assert ListenerEvent.SPEECH_END in feed(detector, silence(CONFIG.silence_end_ms + 60))
    assert detector.take_utterance()


def test_flush_ends_a_running_utterance_without_waiting():
    """Push-to-talk: a released button is a fact, the silence timer is a guess."""
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(600))
    assert detector.in_speech
    assert detector.flush() == [ListenerEvent.SPEECH_END]
    assert not detector.in_speech
    assert detector.take_utterance()


def test_flush_with_nothing_captured_is_not_a_turn():
    detector = UtteranceDetector()
    feed(detector, room_noise(200))
    assert detector.flush() == []


# --------------------------------------------------------------------------------------
# Noise floor
# --------------------------------------------------------------------------------------


def test_a_noisy_room_does_not_read_as_continuous_speech():
    """The gate is relative to the room, not to an absolute level."""
    detector = UtteranceDetector()
    events = feed(detector, room_noise(2000, amplitude=0.02))
    assert ListenerEvent.SPEECH_START not in events


def test_speech_still_registers_over_that_noise():
    detector = UtteranceDetector()
    feed(detector, room_noise(1000, amplitude=0.02))
    events = feed(detector, tone(400, amplitude=0.4))
    assert ListenerEvent.SPEECH_START in events


def test_a_sustained_loud_passage_cannot_raise_the_floor_above_the_voice():
    """The floor rises slowly on purpose.

    If it tracked loudness symmetrically, a long sentence would raise the floor above
    the speaker's own level and the end of the sentence would read as silence — the
    character would then start answering halfway through.
    """
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(3000, amplitude=0.4))
    assert detector.in_speech
    assert detector.noise_floor_dbfs is not None
    # Still well below the tone's own level (~-11 dBFS for amplitude 0.4).
    assert detector.noise_floor_dbfs < -25.0


def test_silence_alone_is_never_speech():
    """Digital silence has no dynamic range to calibrate against; the absolute floor
    is what stops the gate from promoting its own hiss."""
    detector = UtteranceDetector()
    assert feed(detector, silence(3000)) == []


# --------------------------------------------------------------------------------------
# Barge-in
# --------------------------------------------------------------------------------------


def test_speech_during_playback_is_a_barge_in():
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    events = feed(
        detector,
        tone(CONFIG.frame_ms * CONFIG.barge_in_frames),
        character_is_speaking=True,
    )
    assert ListenerEvent.BARGE_IN in events
    assert detector.in_speech


def test_a_short_noise_during_playback_is_not_a_barge_in():
    """A false positive costs a cut-off reply; the threshold is a run, not a frame."""
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    events = feed(detector, tone(CONFIG.frame_ms), character_is_speaking=True)
    assert events == []


def test_the_interrupting_words_are_kept():
    """The user should not have to repeat themselves after interrupting."""
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    feed(detector, tone(CONFIG.frame_ms * CONFIG.barge_in_frames), character_is_speaking=True)
    captured_so_far = detector.buffered_seconds
    assert captured_so_far > 0.0

    # Playback has stopped by now, so the rest of the sentence arrives normally.
    feed(detector, tone(400))
    events = feed(detector, silence(CONFIG.silence_end_ms + 60))
    assert ListenerEvent.SPEECH_END in events
    assert detector.buffered_seconds > captured_so_far


def test_stray_clicks_during_a_long_reply_are_not_prepended_to_the_next_utterance():
    """The pre-roll is a ring, not a growing buffer.

    Buffering every voiced frame heard during playback would glue a cough from the start
    of a 20 s reply onto the front of whatever the user eventually says.
    """
    detector = UtteranceDetector()
    feed(detector, room_noise(400))
    for _ in range(10):
        feed(detector, tone(CONFIG.frame_ms), character_is_speaking=True)
        feed(detector, silence(200), character_is_speaking=True)
    assert not detector.in_speech
    assert detector.buffered_seconds == 0.0


# --------------------------------------------------------------------------------------
# PCM conversion
# --------------------------------------------------------------------------------------


def test_pcm_round_trips_within_quantisation_error():
    original = tone(100)
    restored = pcm16_to_floats(floats_to_pcm16(original))
    assert len(restored) == len(original)
    assert max(abs(a - b) for a, b in zip(original, restored)) < 1e-4


def test_pcm_clamps_rather_than_wrapping():
    """A sample above full scale must clip, not wrap around to full negative."""
    assert pcm16_to_floats(floats_to_pcm16([2.0, -2.0])) == pytest.approx(
        [0.99997, -0.99997], abs=1e-4
    )


def test_an_odd_trailing_byte_is_ignored():
    """Frames arrive off a socket; a half sample must not raise."""
    assert pcm16_to_floats(b"\x00\x01\x02") == pytest.approx([256 / 32768.0])


def test_feed_pcm16_matches_feed():
    a, b = UtteranceDetector(), UtteranceDetector()
    samples = room_noise(400) + tone(500)
    assert feed(a, samples) == b.feed_pcm16(floats_to_pcm16(samples))


# --------------------------------------------------------------------------------------
# The voice loop
# --------------------------------------------------------------------------------------


class FakeSTT(SpeechToTextProvider):
    provider = "fake"

    def __init__(self, text: str = "你在吗") -> None:
        self.text = text
        self.calls: list[tuple[float, tuple[str, ...]]] = []

    def capabilities(self) -> STTCapabilities:
        return STTCapabilities(provider=self.provider, model="fake", supports_hotwords=True)

    async def transcribe(self, audio_path: Path, *, language="zh-CN", hotwords=None):
        # The file must exist and be readable: the loop writes a real WAV, and a
        # provider that is handed a path it cannot open is the bug this catches.
        assert Path(audio_path).is_file()
        assert Path(audio_path).stat().st_size > 44
        self.calls.append((language, tuple(hotwords or ())))
        return Transcript(text=self.text, provider=self.provider, model="fake")


def make_loop(tmp_path: Path, *, stt: FakeSTT | None = None) -> VoiceLoop:
    return VoiceLoop(make_orchestrator(tmp_path), stt or FakeSTT())


async def drain(loop: VoiceLoop, samples) -> list:
    events = []
    async for event in loop.feed(floats_to_pcm16(samples)):
        events.append(event)
    return events


def test_a_spoken_utterance_runs_a_whole_turn(tmp_path: Path):
    stt = FakeSTT("今天天气怎么样")
    loop = make_loop(tmp_path, stt=stt)

    async def scenario():
        assert await drain(loop, room_noise(400) + tone(600)) == []
        return await drain(loop, silence(CONFIG.silence_end_ms + 60))

    events = run(scenario())
    kinds = [event.type for event in events]
    assert kinds[0] is TurnEventType.TRANSCRIPT
    assert events[0].text == "今天天气怎么样"
    assert TurnEventType.AUDIO in kinds
    assert kinds[-1] is TurnEventType.TURN_END
    assert stt.calls and stt.calls[0][0] == "zh-CN"


def test_the_character_name_is_passed_to_the_recogniser(tmp_path: Path):
    """Generic Chinese ASR mangles proper nouns, and the LLM then answers the wrong
    question. The profile already knows the name, so there is no excuse not to."""
    stt = FakeSTT()
    loop = make_loop(tmp_path, stt=stt)
    assert "迪尼娅" in loop.hotwords

    run(drain(loop, room_noise(400) + tone(600)))
    run(drain(loop, silence(CONFIG.silence_end_ms + 60)))
    assert "迪尼娅" in stt.calls[0][1]


def test_an_empty_transcript_is_not_a_turn(tmp_path: Path):
    """Noise that endpoints as speech but transcribes as nothing must not make the
    character say something. An unprompted reply is worse than a missed one."""
    loop = make_loop(tmp_path, stt=FakeSTT(""))
    run(drain(loop, room_noise(400) + tone(600)))
    events = run(drain(loop, silence(CONFIG.silence_end_ms + 60)))
    assert events == []
    assert loop.orchestrator.state is ConversationState.IDLE


def test_observe_returns_the_utterance_without_running_the_turn(tmp_path: Path):
    """The transport needs these separable: intake is ordered and fast, the turn is
    slow, and running them together means not listening during the reply."""
    loop = make_loop(tmp_path)

    async def scenario():
        assert await loop.observe(floats_to_pcm16(room_noise(400) + tone(600))) == []
        return await loop.observe(floats_to_pcm16(silence(CONFIG.silence_end_ms + 60)))

    utterances = run(scenario())
    assert len(utterances) == 1
    assert len(utterances[0]) > frames(10)


def test_end_utterance_flushes_without_the_silence_timer(tmp_path: Path):
    loop = make_loop(tmp_path)

    async def scenario():
        await loop.observe(floats_to_pcm16(room_noise(400) + tone(600)))
        utterances = loop.end_utterance()
        assert len(utterances) == 1
        return [event async for event in loop.respond(utterances[0])]

    events = run(scenario())
    assert events[0].type is TurnEventType.TRANSCRIPT
    assert events[-1].type is TurnEventType.TURN_END


def test_speaking_over_the_character_interrupts_her(tmp_path: Path):
    """Spec §16, over the microphone rather than a button."""
    loop = make_loop(tmp_path)
    orchestrator = loop.orchestrator

    async def scenario():
        # Get her talking, then cut in while she is mid-reply.
        turn = asyncio.create_task(_consume(orchestrator.run_turn("在吗")))
        await _wait_for_state(
            orchestrator, {ConversationState.SPEAKING, ConversationState.SYNTHESIZING}
        )
        await loop.observe(floats_to_pcm16(room_noise(200)))
        await loop.observe(
            floats_to_pcm16(tone(CONFIG.frame_ms * CONFIG.barge_in_frames * 2))
        )
        events = await turn
        return events

    events = run(scenario())
    assert any(
        e.type is TurnEventType.TURN_END and e.detail == "interrupted" for e in events
    )


def test_the_loop_ignores_audio_that_never_becomes_speech(tmp_path: Path):
    stt = FakeSTT()
    loop = make_loop(tmp_path, stt=stt)
    assert run(drain(loop, room_noise(2000))) == []
    assert stt.calls == []


async def _consume(stream) -> list:
    return [event async for event in stream]


async def _wait_for_state(orchestrator, wanted: set, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while orchestrator.state not in wanted:
        if asyncio.get_running_loop().time() > deadline:  # pragma: no cover
            raise AssertionError(f"state stayed {orchestrator.state}")
        await asyncio.sleep(0.005)
