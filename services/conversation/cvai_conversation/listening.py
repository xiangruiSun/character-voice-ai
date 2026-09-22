"""Listening: utterance endpointing and barge-in detection (Milestone 11).

Turning a microphone stream into turns needs three decisions, and all three are here
rather than in the browser, because spec §15 puts pipeline state on the server:

* **When did the user start speaking?** A few consecutive voiced frames, not one — a
  door closing is one frame.
* **When did they stop?** Trailing silence long enough to be a finished thought rather
  than a pause for breath. Too short and the character interrupts *them*; too long and
  the conversation feels sluggish. 700 ms is the usual compromise for Mandarin.
* **Are they talking over the character?** The same voiced-frame test with a shorter
  trigger, because during playback a false positive costs only a cut-off reply, while a
  miss means the user is talked over — and being talked over by a machine is the thing
  that most reliably breaks the illusion.

The detector is an adaptive energy gate, deliberately the same family as the one in
``cvai_core.dsp`` so the runtime and the Voice Pack pipeline agree about what counts as
speech. Silero VAD is better in noise and is used when installed; the energy gate is the
fallback and the reference implementation, and it is what the tests pin.

Echo cancellation is **not** here. Without it, open speakers will trigger barge-in on the
character's own voice. The browser's `echoCancellation` constraint handles the common
case; headphones handle the rest. Anything more belongs in WebRTC, not in this loop.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import AsyncIterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING

from cvai_core.dsp import rms, to_dbfs
from cvai_core.interfaces.stt import SpeechToTextProvider
from cvai_core.logging_setup import get_logger
from cvai_types import (
    ConversationState,
    CVAIModel,
    InterruptReason,
    InterruptSignal,
    TurnEvent,
    TurnEventType,
)
from pydantic import Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .orchestrator import ConversationOrchestrator

log = get_logger(__name__)


class ListenerEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    #: The user started speaking while the character was still talking.
    BARGE_IN = "barge_in"
    #: The utterance hit the length cap and was cut so it could be transcribed at all.
    TOO_LONG = "too_long"


class ListenerConfig(CVAIModel):
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    frame_ms: int = Field(default=20, ge=10, le=60)

    #: Consecutive voiced frames before speech is declared. One frame is a door closing.
    speech_start_frames: int = Field(default=3, ge=1, le=20)
    #: Trailing silence that ends an utterance. Shorter and the character interrupts the
    #: user mid-thought; longer and the conversation drags.
    silence_end_ms: int = Field(default=700, ge=100, le=3000)
    min_utterance_ms: int = Field(default=300, ge=50, le=5000)
    max_utterance_s: float = Field(default=20.0, gt=1.0, le=120.0)

    #: Speech needed to trigger barge-in while the character is talking. Shorter than a
    #: normal start: a miss means the user is talked over, which is worse than a
    #: needlessly cut-off reply.
    barge_in_frames: int = Field(default=5, ge=1, le=50)

    #: dB above the running noise floor for a frame to count as speech.
    margin_db: float = Field(default=10.0, ge=3.0, le=30.0)
    #: Absolute level below which nothing is speech, whatever the floor says. Stops a
    #: silent room from having its own hiss promoted to speech.
    absolute_floor_dbfs: float = Field(default=-45.0, le=0.0)
    #: How fast the noise floor tracks the room. Slow to rise, quick to fall, so a
    #: sustained loud passage does not raise the floor above the user's voice.
    floor_attack: float = Field(default=0.35, gt=0.0, le=1.0)
    floor_release: float = Field(default=0.02, gt=0.0, le=1.0)


class UtteranceDetector:
    """Frame-by-frame endpointing over a live microphone stream."""

    def __init__(self, config: ListenerConfig | None = None) -> None:
        self.config = config or ListenerConfig()
        self.frame_samples = max(
            1, int(self.config.sample_rate * self.config.frame_ms / 1000)
        )
        self._pending: list[float] = []
        self._buffer: list[float] = []
        # One frame longer than the longest trigger, so the frame that first looked
        # like speech is still in the ring when the decision is finally made.
        self._preroll: deque[list[float]] = deque(
            maxlen=max(self.config.speech_start_frames, self.config.barge_in_frames) + 1
        )
        self._noise_floor: float | None = None
        self._voiced_run = 0
        self._silent_run = 0
        self._in_speech = False
        self._speaking_run = 0

    # -- state ----------------------------------------------------------------------

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def buffered_seconds(self) -> float:
        return len(self._buffer) / self.config.sample_rate

    @property
    def noise_floor_dbfs(self) -> float | None:
        return self._noise_floor

    def reset(self) -> None:
        self._pending.clear()
        self._buffer.clear()
        self._preroll.clear()
        self._voiced_run = 0
        self._silent_run = 0
        self._speaking_run = 0
        self._in_speech = False

    def take_utterance(self) -> list[float]:
        """Remove and return the captured utterance."""
        captured, self._buffer = self._buffer, []
        return captured

    # -- feeding ---------------------------------------------------------------------

    def feed(
        self, samples: Sequence[float], *, character_is_speaking: bool = False
    ) -> list[ListenerEvent]:
        """Add audio; return whatever it made true."""
        events: list[ListenerEvent] = []
        self._pending.extend(samples)

        while len(self._pending) >= self.frame_samples:
            frame = self._pending[: self.frame_samples]
            del self._pending[: self.frame_samples]
            events.extend(self._process(frame, character_is_speaking))
        return events

    def feed_pcm16(
        self, payload: bytes, *, character_is_speaking: bool = False
    ) -> list[ListenerEvent]:
        """Convenience for the wire format: little-endian signed 16-bit PCM."""
        return self.feed(
            pcm16_to_floats(payload), character_is_speaking=character_is_speaking
        )

    # -- internals -------------------------------------------------------------------

    def _process(
        self, frame: list[float], character_is_speaking: bool
    ) -> list[ListenerEvent]:
        level = to_dbfs(rms(frame))
        voiced = self._is_voiced(level)
        self._track_floor(level, voiced=voiced)

        events: list[ListenerEvent] = []

        if self._in_speech:
            self._buffer.extend(frame)
        else:
            # Everything heard before speech is declared is held in a short ring, so
            # that when it *is* declared the utterance starts at the first syllable
            # rather than three frames into it. Mandarin初声 are short; losing 60 ms of
            # one is enough to turn 四 into 是 in the transcript.
            self._preroll.append(list(frame))

        if character_is_speaking:
            # While the character talks, the only question is whether to cut her off.
            self._speaking_run = self._speaking_run + 1 if voiced else 0
            if self._speaking_run >= self.config.barge_in_frames:
                self._speaking_run = 0
                self._start_speech()
                events.append(ListenerEvent.BARGE_IN)
            return events

        self._speaking_run = 0

        if voiced:
            self._voiced_run += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
            self._voiced_run = 0

        if self._in_speech:
            if self.buffered_seconds >= self.config.max_utterance_s:
                self._in_speech = False
                events.append(ListenerEvent.TOO_LONG)
                events.append(ListenerEvent.SPEECH_END)
                return events
            if self._silent_run * self.config.frame_ms >= self.config.silence_end_ms:
                self._in_speech = False
                if self._long_enough():
                    events.append(ListenerEvent.SPEECH_END)
                else:
                    # A cough is not a turn.
                    self._buffer.clear()
            return events

        if self._voiced_run >= self.config.speech_start_frames:
            self._start_speech()
            events.append(ListenerEvent.SPEECH_START)
        return events

    def _start_speech(self) -> None:
        self._in_speech = True
        self._silent_run = 0
        self._voiced_run = self.config.speech_start_frames
        for held in self._preroll:
            self._buffer.extend(held)
        self._preroll.clear()

    def flush(self) -> list[ListenerEvent]:
        """End the current utterance now, however it was going.

        Used for push-to-talk and for ``user_audio_end``: the client says the user has
        stopped, so there is no point waiting out the silence timer. Returns a
        ``SPEECH_END`` only if there is enough audio to be worth transcribing.
        """
        if not self._in_speech:
            return []
        self._in_speech = False
        self._silent_run = 0
        self._voiced_run = 0
        self._preroll.clear()
        if self.buffered_seconds * 1000.0 >= self.config.min_utterance_ms:
            return [ListenerEvent.SPEECH_END]
        self._buffer.clear()
        return []

    def _long_enough(self) -> bool:
        trailing = self.config.silence_end_ms / 1000.0
        speech = max(0.0, self.buffered_seconds - trailing)
        return speech * 1000.0 >= self.config.min_utterance_ms

    def _is_voiced(self, level: float) -> bool:
        if level <= self.config.absolute_floor_dbfs:
            return False
        floor = self._noise_floor if self._noise_floor is not None else level
        return level >= floor + self.config.margin_db

    def _track_floor(self, level: float, *, voiced: bool) -> None:
        if self._noise_floor is None:
            self._noise_floor = level
            return
        if voiced:
            # The floor is a model of the *room*, so speech must not feed it. An
            # earlier version updated on every frame, and a long uninterrupted
            # sentence then pulled the floor up to within the margin of the speaker's
            # own voice: after roughly a second and a half, speech read as silence and
            # the character started answering halfway through. Held here instead.
            return
        # Falls quickly toward a quieter room, rises slowly, so a burst of background
        # noise does not immediately desensitise the gate.
        rate = (
            self.config.floor_attack
            if level < self._noise_floor
            else self.config.floor_release
        )
        self._noise_floor += (level - self._noise_floor) * rate


def pcm16_to_floats(payload: bytes) -> list[float]:
    """Little-endian signed 16-bit PCM to floats in [-1, 1]."""
    import struct

    count = len(payload) // 2
    if count == 0:
        return []
    return [v / 32768.0 for v in struct.unpack(f"<{count}h", payload[: count * 2])]


def floats_to_pcm16(samples: Sequence[float]) -> bytes:
    import struct

    out = bytearray()
    for value in samples:
        clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else value)
        out += struct.pack("<h", int(round(clamped * 32767)))
    return bytes(out)


# --------------------------------------------------------------------------------------
# Voice loop
# --------------------------------------------------------------------------------------


class VoiceLoop:
    """Microphone audio in, turn events out (Milestone 11).

    Ties the detector to STT and the orchestrator. Deliberately thin: the decisions live
    in the detector, the conversation lives in the orchestrator, and this only routes
    between them — which is what keeps the microphone path and the typed path identical
    from the orchestrator's point of view.

    Intake is split from response on purpose. :meth:`observe` is fast and must stay
    ordered — it is the frame-by-frame path, and audio arriving out of order would be
    audio the detector cannot endpoint. :meth:`respond` is slow: ASR, an LLM and a TTS
    engine. A transport that ran them together would stop reading the microphone for as
    long as the character takes to answer, which is exactly the window barge-in needs.
    :meth:`feed` does both, for callers (tests, a CLI) where that does not matter.
    """

    def __init__(
        self,
        orchestrator: "ConversationOrchestrator",
        stt: SpeechToTextProvider,
        *,
        config: ListenerConfig | None = None,
        hotwords: Sequence[str] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.stt = stt
        self.detector = UtteranceDetector(config)
        # Character and world proper nouns. Generic Chinese ASR mangles them, and a
        # mangled name means the LLM answers a question the user did not ask.
        self.hotwords = list(hotwords or _profile_hotwords(orchestrator))

    async def observe(self, payload: bytes) -> list[list[float]]:
        """Take one chunk of microphone PCM; return any finished utterances.

        Cheap and ordered. Barge-in is handled here rather than by the caller, because
        the whole value of detecting it is that it happens *now* — a cut-off reply one
        turn late is worse than none.
        """
        speaking = self.orchestrator.state in (
            ConversationState.SPEAKING,
            ConversationState.SYNTHESIZING,
        )
        events = self.detector.feed_pcm16(payload, character_is_speaking=speaking)

        utterances: list[list[float]] = []
        for event in events:
            if event is ListenerEvent.BARGE_IN:
                await self.orchestrator.interrupt(
                    InterruptSignal(reason=InterruptReason.USER_SPEECH)
                )
            elif event is ListenerEvent.TOO_LONG:
                log.warning("utterance hit the length cap; transcribing what we have")
            elif event is ListenerEvent.SPEECH_END:
                captured = self.detector.take_utterance()
                if captured:
                    utterances.append(captured)
        return utterances

    async def respond(self, samples: Sequence[float]) -> AsyncIterator[TurnEvent]:
        """Transcribe one captured utterance and run the turn it asks for."""
        transcript = await self._transcribe(samples)
        text = (transcript or "").strip()
        if not text:
            # Silence, noise, or ASR having a bad time. Not a turn; say so and wait.
            log.info("utterance produced no transcript; ignoring")
            return

        yield TurnEvent(
            type=TurnEventType.TRANSCRIPT,
            turn_id=self.orchestrator.current_turn_id or "pending",
            text=text,
            is_final=True,
        )
        async for event in self.orchestrator.run_turn(text):
            yield event

    async def feed(self, payload: bytes) -> AsyncIterator[TurnEvent]:
        """Observe and respond in one pass, for callers that can block on a turn."""
        for utterance in await self.observe(payload):
            async for event in self.respond(utterance):
                yield event

    def end_utterance(self) -> list[list[float]]:
        """The client says the user stopped — don't wait out the silence timer.

        Push-to-talk and ``user_audio_end`` both land here. Endpointing by timer is a
        guess; a released button is not, so it wins. Returns what to
        :meth:`respond` to, in the same shape as :meth:`observe`.
        """
        utterances: list[list[float]] = []
        for event in self.detector.flush():
            if event is ListenerEvent.SPEECH_END:
                captured = self.detector.take_utterance()
                if captured:
                    utterances.append(captured)
        return utterances

    def reset(self) -> None:
        """Forget any part-captured utterance (a new turn, or listening stopped)."""
        self.detector.reset()

    async def _transcribe(self, samples: Sequence[float]) -> str:
        import tempfile
        from pathlib import Path

        from cvai_core.audio import write_wav

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utterance.wav"
            write_wav(path, samples, self.detector.config.sample_rate)
            result = await self.stt.transcribe(
                path, language="zh-CN", hotwords=self.hotwords
            )
        return result.text


def _profile_hotwords(orchestrator: "ConversationOrchestrator") -> list[str]:
    """Proper nouns worth telling the ASR about, taken from the character profile."""
    profile = getattr(orchestrator, "profile", None)
    if profile is None:
        return []
    words = [profile.character_name]
    words += list(profile.speaking_habits.frequent_expressions)
    return [word for word in words if word and len(word) <= 12]


def estimate_frames(seconds: float, config: ListenerConfig) -> int:
    return max(1, int(math.ceil(seconds * 1000.0 / config.frame_ms)))
