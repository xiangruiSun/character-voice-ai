"""Provider interfaces (spec §2).

The abstractions the spec asks for, in one place so that no service imports another
service's concrete classes:

* :class:`~cvai_core.interfaces.tts.TTSProvider`
* :class:`~cvai_core.interfaces.stt.SpeechToTextProvider`
* :class:`~cvai_core.interfaces.llm.LLMProvider`
* :class:`~cvai_core.interfaces.character.CharacterProvider`
* :class:`~cvai_core.interfaces.reference.ReferenceRetriever`
* :class:`~cvai_core.interfaces.session.ConversationSession`

**Sync vs async.** The three model-backed providers (TTS, STT, LLM) are async: they all
end up as network calls to a sidecar or a hosted API, and Milestones 10-13 need streaming
and cancellation. Making them async now avoids rewriting every adapter at Milestone 10.
``CharacterProvider`` and ``ReferenceRetriever`` are sync: they are local lookups over
data already in memory, and making them async would buy nothing but colour.
"""

from __future__ import annotations

from .character import CharacterProvider
from .llm import LLMProvider
from .reference import ReferenceRetriever
from .session import ConversationSession
from .stt import SpeechToTextProvider
from .tts import TTSProvider

__all__ = [
    "CharacterProvider",
    "ConversationSession",
    "LLMProvider",
    "ReferenceRetriever",
    "SpeechToTextProvider",
    "TTSProvider",
]
