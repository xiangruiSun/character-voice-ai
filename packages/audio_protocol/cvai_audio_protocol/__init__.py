"""Browser ↔ server message protocol for the voice conversation (Milestones 9-13).

Defined in Milestone 1, used later, for the reason spec §15 gives: the orchestrator owns
pipeline state and frontend components must not invent their own. A shared, typed
protocol is how that rule is enforced rather than merely stated — the frontend can only
*observe* ``state`` messages, and the only thing it can *assert* is user intent
(``start_listening``, ``user_audio``, ``interrupt``).

Two channels over one WebSocket:

* **control** — JSON text frames, typed here;
* **audio** — binary frames, raw PCM, described by a preceding ``audio_begin`` control
  message. Audio never travels as base64 inside JSON: a 20 ms frame of 24 kHz PCM is
  960 bytes, and base64-encoding it inside a JSON envelope roughly doubles that for no
  benefit at conversational rates.

Barge-in is part of the protocol from the start (spec §16): ``interrupt`` flows up,
``audio_cancel`` flows down, and both carry the turn id so a late audio frame from a
cancelled turn can be discarded rather than played over the next one.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from cvai_types import ConversationState, CVAIModel, InterruptReason, utcnow
from pydantic import Field

PROTOCOL_VERSION = "1"

#: Server → client audio frames are raw little-endian 16-bit PCM at this rate unless an
#: ``audio_begin`` says otherwise. 24 kHz is the common denominator across the candidate
#: engines; 48 kHz-native engines are resampled at the edge, not in the player.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FRAME_MS = 20


class ClientMessageType(str, Enum):
    HELLO = "hello"
    START_LISTENING = "start_listening"
    STOP_LISTENING = "stop_listening"
    USER_TEXT = "user_text"
    USER_AUDIO_BEGIN = "user_audio_begin"
    USER_AUDIO_END = "user_audio_end"
    INTERRUPT = "interrupt"
    PLAYBACK_PROGRESS = "playback_progress"
    BYE = "bye"


class ServerMessageType(str, Enum):
    READY = "ready"
    STATE = "state"
    TRANSCRIPT = "transcript"
    CHARACTER_TEXT = "character_text"
    AUDIO_BEGIN = "audio_begin"
    AUDIO_END = "audio_end"
    AUDIO_CANCEL = "audio_cancel"
    TURN_END = "turn_end"
    ERROR = "error"


class _Message(CVAIModel):
    protocol: str = PROTOCOL_VERSION
    at: str = Field(default_factory=lambda: utcnow().isoformat())


# --------------------------------------------------------------------------------------
# Client → server
# --------------------------------------------------------------------------------------


class Hello(_Message):
    type: Literal[ClientMessageType.HELLO] = ClientMessageType.HELLO
    character_id: str
    session_id: str | None = None
    #: What the browser can actually play, so the server resamples once, server-side,
    #: instead of the page discovering mid-conversation that it cannot decode 48 kHz.
    playback_sample_rate: int = DEFAULT_SAMPLE_RATE


class StartListening(_Message):
    type: Literal[ClientMessageType.START_LISTENING] = ClientMessageType.START_LISTENING


class StopListening(_Message):
    type: Literal[ClientMessageType.STOP_LISTENING] = ClientMessageType.STOP_LISTENING


class UserText(_Message):
    type: Literal[ClientMessageType.USER_TEXT] = ClientMessageType.USER_TEXT
    text: str


class UserAudioBegin(_Message):
    type: Literal[ClientMessageType.USER_AUDIO_BEGIN] = ClientMessageType.USER_AUDIO_BEGIN
    sample_rate: int = 16000
    channels: int = 1


class UserAudioEnd(_Message):
    type: Literal[ClientMessageType.USER_AUDIO_END] = ClientMessageType.USER_AUDIO_END


class Interrupt(_Message):
    """Barge-in from the client (spec §16)."""

    type: Literal[ClientMessageType.INTERRUPT] = ClientMessageType.INTERRUPT
    reason: InterruptReason = InterruptReason.USER_SPEECH
    #: The turn being interrupted. The server ignores an interrupt for a turn that has
    #: already finished, which is what stops a late signal from killing the next reply.
    turn_id: str | None = None


class PlaybackProgress(_Message):
    """How much audio the client has actually played.

    The server needs this to know when speech really ended: audio handed to the browser
    is not audio heard, and the difference is exactly the window in which barge-in
    happens.
    """

    type: Literal[ClientMessageType.PLAYBACK_PROGRESS] = ClientMessageType.PLAYBACK_PROGRESS
    turn_id: str
    played_ms: float = Field(ge=0.0)
    buffered_ms: float = Field(default=0.0, ge=0.0)


class Bye(_Message):
    type: Literal[ClientMessageType.BYE] = ClientMessageType.BYE


ClientMessage = Union[
    Hello,
    StartListening,
    StopListening,
    UserText,
    UserAudioBegin,
    UserAudioEnd,
    Interrupt,
    PlaybackProgress,
    Bye,
]


# --------------------------------------------------------------------------------------
# Server → client
# --------------------------------------------------------------------------------------


class Ready(_Message):
    type: Literal[ServerMessageType.READY] = ServerMessageType.READY
    session_id: str
    character_id: str
    character_name: str
    engine: str
    sample_rate: int = DEFAULT_SAMPLE_RATE


class StateChanged(_Message):
    """The single source of truth for pipeline state (spec §15)."""

    type: Literal[ServerMessageType.STATE] = ServerMessageType.STATE
    state: ConversationState
    turn_id: str | None = None
    detail: str = ""


class TranscriptMessage(_Message):
    type: Literal[ServerMessageType.TRANSCRIPT] = ServerMessageType.TRANSCRIPT
    text: str
    is_final: bool = False


class CharacterText(_Message):
    """What the character says. The user sees this; the performance metadata stays server-side."""

    type: Literal[ServerMessageType.CHARACTER_TEXT] = ServerMessageType.CHARACTER_TEXT
    turn_id: str
    text: str
    is_final: bool = False


class AudioBegin(_Message):
    """Binary frames that follow belong to this chunk until ``audio_end``."""

    type: Literal[ServerMessageType.AUDIO_BEGIN] = ServerMessageType.AUDIO_BEGIN
    turn_id: str
    chunk_index: int = Field(ge=0)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    #: The text this chunk speaks, for caption display and debugging.
    text: str = ""


class AudioEnd(_Message):
    type: Literal[ServerMessageType.AUDIO_END] = ServerMessageType.AUDIO_END
    turn_id: str
    chunk_index: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)


class AudioCancel(_Message):
    """Drop everything buffered for this turn, now (spec §16)."""

    type: Literal[ServerMessageType.AUDIO_CANCEL] = ServerMessageType.AUDIO_CANCEL
    turn_id: str
    reason: InterruptReason = InterruptReason.USER_SPEECH


class TurnEnd(_Message):
    type: Literal[ServerMessageType.TURN_END] = ServerMessageType.TURN_END
    turn_id: str
    was_interrupted: bool = False


class ErrorMessage(_Message):
    type: Literal[ServerMessageType.ERROR] = ServerMessageType.ERROR
    code: str
    message: str
    recoverable: bool = True


ServerMessage = Union[
    Ready,
    StateChanged,
    TranscriptMessage,
    CharacterText,
    AudioBegin,
    AudioEnd,
    AudioCancel,
    TurnEnd,
    ErrorMessage,
]


_CLIENT_TYPES: dict[str, type] = {
    ClientMessageType.HELLO.value: Hello,
    ClientMessageType.START_LISTENING.value: StartListening,
    ClientMessageType.STOP_LISTENING.value: StopListening,
    ClientMessageType.USER_TEXT.value: UserText,
    ClientMessageType.USER_AUDIO_BEGIN.value: UserAudioBegin,
    ClientMessageType.USER_AUDIO_END.value: UserAudioEnd,
    ClientMessageType.INTERRUPT.value: Interrupt,
    ClientMessageType.PLAYBACK_PROGRESS.value: PlaybackProgress,
    ClientMessageType.BYE.value: Bye,
}

_SERVER_TYPES: dict[str, type] = {
    ServerMessageType.READY.value: Ready,
    ServerMessageType.STATE.value: StateChanged,
    ServerMessageType.TRANSCRIPT.value: TranscriptMessage,
    ServerMessageType.CHARACTER_TEXT.value: CharacterText,
    ServerMessageType.AUDIO_BEGIN.value: AudioBegin,
    ServerMessageType.AUDIO_END.value: AudioEnd,
    ServerMessageType.AUDIO_CANCEL.value: AudioCancel,
    ServerMessageType.TURN_END.value: TurnEnd,
    ServerMessageType.ERROR.value: ErrorMessage,
}


def parse_client_message(payload: dict) -> ClientMessage:
    kind = payload.get("type")
    model = _CLIENT_TYPES.get(kind)
    if model is None:
        raise ValueError(f"unknown client message type {kind!r}")
    return model.model_validate(payload)


def parse_server_message(payload: dict) -> ServerMessage:
    kind = payload.get("type")
    model = _SERVER_TYPES.get(kind)
    if model is None:
        raise ValueError(f"unknown server message type {kind!r}")
    return model.model_validate(payload)


__all__ = [
    "DEFAULT_FRAME_MS",
    "DEFAULT_SAMPLE_RATE",
    "PROTOCOL_VERSION",
    "AudioBegin",
    "AudioCancel",
    "AudioEnd",
    "Bye",
    "CharacterText",
    "ClientMessage",
    "ClientMessageType",
    "ErrorMessage",
    "Hello",
    "Interrupt",
    "PlaybackProgress",
    "Ready",
    "ServerMessage",
    "ServerMessageType",
    "StartListening",
    "StateChanged",
    "StopListening",
    "TranscriptMessage",
    "TurnEnd",
    "UserAudioBegin",
    "UserAudioEnd",
    "UserText",
    "parse_client_message",
    "parse_server_message",
]
