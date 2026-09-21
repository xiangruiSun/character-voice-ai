"""``SpeechToTextProvider``.

OpenAI transcription first, with the interface shaped so ``faster-whisper`` or FunASR's
streaming Paraformer can be dropped in later (spec §3). ``transcribe_stream`` is separate
from ``transcribe`` rather than a flag, because the two have genuinely different
semantics: one returns a final transcript, the other emits partials that the orchestrator
uses for endpointing and barge-in.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from pathlib import Path

from cvai_types import STTCapabilities, Transcript


class SpeechToTextProvider(abc.ABC):
    provider: str = "unknown"

    @abc.abstractmethod
    def capabilities(self) -> STTCapabilities:
        ...

    @abc.abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "zh-CN",
        hotwords: list[str] | None = None,
    ) -> Transcript:
        """Transcribe a complete audio file.

        ``hotwords`` carries character and world proper nouns; providers that do not
        support them ignore the argument rather than failing, and say so in capabilities.
        """

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        language: str = "zh-CN",
        sample_rate: int = 16000,
    ) -> AsyncIterator[Transcript]:
        """Emit partial then final transcripts (Milestone 11)."""
        raise NotImplementedError(f"{self.provider} does not support streaming STT")
        yield Transcript(text="")  # pragma: no cover

    async def aclose(self) -> None:
        return None
