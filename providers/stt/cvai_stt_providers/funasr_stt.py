"""FunASR speech-to-text adapter (Milestone 11).

The local, Mandarin-first alternative to the OpenAI adapter, and the better default for
this project once the weights are on the machine:

* **Paraformer-zh is trained for Mandarin**, where Whisper-family models are trained for
  everything and are measurably worse on short Chinese utterances — which is all a
  conversation turn ever is.
* **It takes real hotwords.** The OpenAI adapter can only hint proper nouns through a
  prompt; the contextual Paraformer takes a weighted vocabulary, so a character's name
  and her world's nouns are recognised rather than guessed at.
* **It runs offline**, so a conversation does not put the user's microphone on someone
  else's server, and there is no per-turn network latency in the listening path.

The model loading and hotword formatting are deliberately *not* reimplemented here. They
live in the Voice Pack preprocessing backends, and the runtime borrows them so that the
ASR which transcribed the training data is the same ASR that hears the user — the same
reason the DSP lives in ``cvai_core.dsp`` rather than in the pipeline that first needed
it. The import is lazy: the runtime does not depend on the preprocessing extras being
installed, it only uses them when they are.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from cvai_core.errors import ProviderUnavailableError
from cvai_core.interfaces.stt import SpeechToTextProvider
from cvai_core.registry import STT_PROVIDERS
from cvai_types import STTCapabilities, Transcript, TranscriptSegment


@STT_PROVIDERS.register("funasr")
class FunASRSTTProvider(SpeechToTextProvider):
    provider = "funasr"

    def __init__(
        self,
        *,
        model: str = "paraformer-zh",
        punctuation_model: str | None = "ct-punc",
        device: str = "cpu",
        hotword_weight: int = 20,
    ) -> None:
        self.model = model
        self.punctuation_model = punctuation_model
        self.device = device
        self.hotword_weight = hotword_weight
        self._backend: Any = None

    def capabilities(self) -> STTCapabilities:
        return STTCapabilities(
            provider=self.provider,
            model=self.model,
            # Paraformer has a streaming variant; this adapter uses the offline one,
            # because endpointing already happens in the listener and a final
            # transcript per utterance is what the orchestrator consumes.
            supports_streaming=False,
            supports_word_timestamps=True,
            supports_hotwords=True,
            languages=["zh-CN"],
        )

    def _backend_or_raise(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            from cvai_voice_preprocessing.backends.funasr_backends import (  # noqa: PLC0415
                FunASRTranscriber,
            )
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError(
                "the FunASR STT adapter needs the preprocessing extras: "
                "pip install -e '.[preprocess]'"
            ) from exc

        backend = FunASRTranscriber(
            model=self.model,
            punc_model=self.punctuation_model,
            device=self.device,
            hotword_weight=self.hotword_weight,
        )
        if not backend.available():  # pragma: no cover - depends on extras
            raise ProviderUnavailableError(backend.unavailable_reason())
        self._backend = backend
        return backend

    async def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "zh-CN",
        hotwords: list[str] | None = None,
    ) -> Transcript:
        backend = self._backend_or_raise()
        started = time.perf_counter()
        # FunASR is synchronous and holds the GIL through a C++ runtime call. Off the
        # event loop it goes, or one transcription stalls every other session's audio.
        result = await asyncio.to_thread(
            backend.transcribe, Path(audio_path), hotwords=tuple(hotwords or ())
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        segments: list[TranscriptSegment] = []
        timings = getattr(result, "char_timings", None) or []
        if timings:
            segments.append(
                TranscriptSegment(
                    text=result.text,
                    start_s=min(t.start_s for t in timings),
                    end_s=max(t.end_s for t in timings),
                    confidence=getattr(result, "confidence", None),
                )
            )
        return Transcript(
            text=result.text,
            language="zh-CN",
            segments=segments,
            provider=self.provider,
            model=self.model,
            latency_ms=elapsed_ms,
        )
