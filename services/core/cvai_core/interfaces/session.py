"""``ConversationSession`` — one live conversation (spec §15, §16).

Declared in Milestone 1 and implemented in Milestone 9. It is here now because spec §16
requires barge-in to be addable "without restructuring the entire system", and the way to
honour that promise is to make interruption part of the interface before anything is
built against it. Note also spec §15: the *orchestrator* owns pipeline state; frontend
components observe it and never manage it.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from cvai_types import (
    ConversationState,
    InterruptSignal,
    SessionConfig,
    SpeechChunk,
    Turn,
)


class ConversationSession(abc.ABC):
    """Stateful conversation loop."""

    config: SessionConfig

    @property
    @abc.abstractmethod
    def state(self) -> ConversationState:
        ...

    @abc.abstractmethod
    async def submit_text(self, text: str) -> AsyncIterator[SpeechChunk]:
        """Run one turn from typed input, yielding speech chunks as they are planned."""

    @abc.abstractmethod
    async def submit_audio(self, audio: bytes, *, sample_rate: int) -> None:
        """Feed captured microphone audio (Milestone 11)."""

    @abc.abstractmethod
    async def interrupt(self, signal: InterruptSignal) -> None:
        """Barge-in (spec §16): stop playback, drain the queue, cancel pending work."""

    @abc.abstractmethod
    def history(self) -> list[Turn]:
        ...

    async def aclose(self) -> None:
        return None
