"""Translating orchestrator events into browser protocol messages.

Two vocabularies exist on purpose. :class:`~cvai_types.conversation.TurnEvent` is what
the orchestrator emits — internal, complete, including things the browser has no business
with (which reference clip, which engine, synthesis latency). The
:mod:`cvai_audio_protocol` messages are what goes over the wire.

Keeping the mapping in its own pure function means the WebSocket handler has no logic in
it, and means this — the part where a field can quietly go missing — is unit-testable.

**What is deliberately not forwarded:** the performance metadata. Spec §12 says the user
only sees or hears the text; emotion, intensity and reference style are instructions to
the voice stack. Sending them to the browser would invite a frontend to start displaying
or, worse, deciding them, which spec §15 forbids.
"""

from __future__ import annotations

from typing import Iterable

from cvai_audio_protocol import (
    AudioBegin,
    AudioEnd,
    CharacterText,
    ErrorMessage,
    ServerMessage,
    StateChanged,
    TranscriptMessage,
    TurnEnd,
)
from cvai_types import TurnEvent, TurnEventType


def to_server_messages(event: TurnEvent) -> list[ServerMessage]:
    """Map one orchestrator event onto zero or more protocol messages."""
    if event.type is TurnEventType.STATE and event.state is not None:
        return [
            StateChanged(state=event.state, turn_id=event.turn_id, detail=event.detail)
        ]

    if event.type is TurnEventType.TRANSCRIPT:
        # What the server heard, echoed back. Not a nicety: when ASR mishears a name,
        # the character's reply looks unhinged until you can see *why*, and the user
        # can correct it by saying it again instead of assuming the character is broken.
        return [TranscriptMessage(text=event.text, is_final=event.is_final)]

    if event.type is TurnEventType.TEXT_DELTA:
        return [
            CharacterText(
                turn_id=event.turn_id, text=event.text, is_final=event.is_final
            )
        ]

    if event.type is TurnEventType.AUDIO:
        # Begin/end bracket the binary frames the transport sends between them, so the
        # client always knows the format and duration of what it is buffering.
        return [
            AudioBegin(
                turn_id=event.turn_id,
                chunk_index=event.chunk_index or 0,
                sample_rate=event.sample_rate or 24000,
                text=event.text,
            ),
            AudioEnd(
                turn_id=event.turn_id,
                chunk_index=event.chunk_index or 0,
                duration_ms=round((event.duration_s or 0.0) * 1000.0, 2),
            ),
        ]

    if event.type is TurnEventType.TURN_END:
        return [
            TurnEnd(
                turn_id=event.turn_id,
                was_interrupted=event.detail == "interrupted",
            )
        ]

    if event.type is TurnEventType.ERROR:
        return [
            ErrorMessage(
                code="turn_failed",
                message=event.error or "unknown error",
                recoverable=True,
            )
        ]

    # PLAN and CHUNK are internal. The plan carries performance direction the user must
    # not see (spec §12); the chunk is superseded by the AUDIO event that follows it.
    return []


def stream_to_messages(events: Iterable[TurnEvent]) -> list[ServerMessage]:
    out: list[ServerMessage] = []
    for event in events:
        out.extend(to_server_messages(event))
    return out


def audio_events(events: Iterable[TurnEvent]) -> list[TurnEvent]:
    """The events that carry a playable file, in order."""
    return [e for e in events if e.type is TurnEventType.AUDIO and e.audio_path]
