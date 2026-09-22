"""Conversation state machine (spec §15) and barge-in signals (spec §16).

Defined in Milestone 1 even though the orchestrator arrives in Milestone 9, because spec
§16 requires that barge-in can be added *without restructuring the system*. The way to
guarantee that is to make interruption part of the state vocabulary from the start, so
every later component is written against a machine that already has an ``INTERRUPTED``
state and a cancellation token.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .base import CVAIModel, Slug, utcnow


class ConversationState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    PLANNING = "planning"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


#: Legal transitions. The orchestrator validates against this table so an invalid
#: transition is a loud bug rather than a subtly stuck UI.
ALLOWED_TRANSITIONS: dict[ConversationState, frozenset[ConversationState]] = {
    ConversationState.IDLE: frozenset(
        {ConversationState.LISTENING, ConversationState.THINKING, ConversationState.ERROR}
    ),
    ConversationState.LISTENING: frozenset(
        {
            ConversationState.TRANSCRIBING,
            ConversationState.IDLE,
            ConversationState.ERROR,
        }
    ),
    ConversationState.TRANSCRIBING: frozenset(
        {ConversationState.THINKING, ConversationState.IDLE, ConversationState.ERROR}
    ),
    ConversationState.THINKING: frozenset(
        {
            ConversationState.PLANNING,
            ConversationState.SYNTHESIZING,
            ConversationState.INTERRUPTED,
            ConversationState.ERROR,
        }
    ),
    ConversationState.PLANNING: frozenset(
        {
            ConversationState.SYNTHESIZING,
            ConversationState.INTERRUPTED,
            ConversationState.ERROR,
        }
    ),
    ConversationState.SYNTHESIZING: frozenset(
        {
            ConversationState.SPEAKING,
            ConversationState.INTERRUPTED,
            ConversationState.ERROR,
        }
    ),
    ConversationState.SPEAKING: frozenset(
        {
            ConversationState.IDLE,
            ConversationState.LISTENING,
            ConversationState.SYNTHESIZING,  # next chunk in a streamed reply
            ConversationState.INTERRUPTED,
            ConversationState.ERROR,
        }
    ),
    ConversationState.INTERRUPTED: frozenset(
        {ConversationState.LISTENING, ConversationState.IDLE, ConversationState.ERROR}
    ),
    ConversationState.ERROR: frozenset({ConversationState.IDLE}),
}


def can_transition(source: ConversationState, target: ConversationState) -> bool:
    return target in ALLOWED_TRANSITIONS.get(source, frozenset())


class StateTransition(CVAIModel):
    from_state: ConversationState
    to_state: ConversationState
    reason: str = Field(default="", max_length=200)
    at: str = Field(default_factory=lambda: utcnow().isoformat())


class InterruptReason(str, Enum):
    USER_SPEECH = "user_speech"
    USER_CANCEL = "user_cancel"
    NEW_TURN = "new_turn"
    ERROR = "error"


class InterruptSignal(CVAIModel):
    """Everything barge-in must do, in one object (spec §16).

    Each flag maps to one required behaviour so a partial implementation is visible in
    the type rather than hidden in a handler.
    """

    reason: InterruptReason
    stop_playback: bool = True
    clear_audio_queue: bool = True
    cancel_pending_tts: bool = True
    cancel_llm_stream: bool = False
    at: str = Field(default_factory=lambda: utcnow().isoformat())


class TurnEventType(str, Enum):
    """What the orchestrator emits while running one turn.

    A single event stream rather than separate callbacks: the API layer, the CLI and
    the tests all consume the same sequence, so "what happens during a turn" is defined
    in exactly one place and cannot drift between transports.
    """

    STATE = "state"
    TRANSCRIPT = "transcript"
    #: Raw text as the LLM produces it. For a caption track, never for TTS.
    TEXT_DELTA = "text_delta"
    PLAN = "plan"
    #: A chunk of text that has been normalized and is about to be synthesized.
    CHUNK = "chunk"
    #: Synthesized audio for a chunk, ready to play.
    AUDIO = "audio"
    TURN_END = "turn_end"
    ERROR = "error"


class TurnEvent(CVAIModel):
    """One thing that happened during a turn."""

    type: TurnEventType
    turn_id: str
    at: str = Field(default_factory=lambda: utcnow().isoformat())

    state: ConversationState | None = None
    text: str = ""
    chunk_index: int | None = Field(default=None, ge=0)
    is_final: bool = False
    #: Path to a rendered audio file, for the AUDIO event.
    audio_path: str | None = None
    sample_rate: int | None = None
    duration_s: float | None = None
    #: Which reference clip conditioned this chunk. Kept on the event so a live
    #: conversation is as traceable as a benchmark run (spec §23).
    reference_id: str | None = None
    engine: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    detail: str = ""


class Turn(CVAIModel):
    turn_index: int = Field(ge=0)
    user_text: str = ""
    character_text: str = ""
    was_interrupted: bool = False
    started_at: str = Field(default_factory=lambda: utcnow().isoformat())
    ended_at: str | None = None


class SessionConfig(CVAIModel):
    session_id: str = Field(min_length=1, max_length=120)
    character_id: Slug
    #: ``None`` uses the character's ``preferred_engine``, then the config default.
    tts_engine: str | None = None
    enable_barge_in: bool = True
    #: Conservative sentence boundaries first; spec §14 says naturalness beats speed.
    max_chunk_chars: int = Field(default=60, ge=10, le=300)
    min_chunk_chars: int = Field(default=8, ge=1, le=100)
