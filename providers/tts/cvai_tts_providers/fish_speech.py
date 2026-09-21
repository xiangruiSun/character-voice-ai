"""Fish Speech / OpenAudio adapter (Milestone 5).

Cloning with this engine is a two-step affair: VQ tokens are extracted from the reference
audio (``tools/vqgan/extract_vq.py`` → a ``.npy``), then generation is conditioned on
those tokens plus the reference text. Extracting per request would be slow and wasteful,
so the tokens are **pre-extracted for the whole Reference Bank at pack build time** and
recorded in each :class:`~cvai_types.voicepack.ReferenceSample`'s ``precomputed`` map
under the key ``fish_speech_prompt_tokens``. This adapter simply passes them through.

Fine-tuning is LoRA-only, and the project documentation warns against fine-tuning on an
RL-trained checkpoint — a constraint recorded in the candidate config so Milestone 5 does
not rediscover it the hard way.

**Licence caution:** the code is open but the weight licence must be confirmed before any
commercial use. ``commercial_use`` is left ``None`` rather than guessed, and the
Milestone 7 decision record has to resolve it if this engine wins.

**Status:** written against the documented API; validated in Milestone 5.
"""

from __future__ import annotations

from typing import Any

from cvai_core.errors import SynthesisError
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import AdaptationMode, ProviderCapabilities, TTSRequest

from .sidecar import GenericSidecarProvider

#: Key used in ``ReferenceSample.precomputed`` for this engine's prompt tokens.
PROMPT_TOKEN_KEY = "fish_speech_prompt_tokens"


@TTS_PROVIDERS.register("fish_speech")
class FishSpeechProvider(GenericSidecarProvider):
    engine = "fish_speech"
    call_name = "tts"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888",
        *,
        engine_version: str = "openaudio-s1-mini",
        native_sample_rate: int = 44100,
        chunk_length: int = 200,
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        require_precomputed_tokens: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, engine_version=engine_version, **kwargs)
        self.native_sample_rate = native_sample_rate
        self.defaults: dict[str, Any] = {
            "chunk_length": chunk_length,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "format": "wav",
        }
        #: Fail loudly when tokens are missing rather than silently falling back to
        #: on-the-fly extraction: the silent path is 10-30x slower and the cause is
        #: invisible in a benchmark's timings.
        self.require_precomputed_tokens = require_precomputed_tokens

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[AdaptationMode.ZERO_SHOT, AdaptationMode.LORA],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            requires_reference_text=True,  # --prompt-text
            supports_multiple_references=True,
            supports_instruct=False,
            supports_emotion_vector=False,
            supports_duration_control=False,
            supports_speed_factor=False,
            supports_seed=True,
            supports_streaming=True,
            supports_hot_checkpoint_swap=False,
            license="see upstream; weights licence must be confirmed",
            commercial_use=None,
        )

    def native_kwargs(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        reference = request.reference
        if reference is None:
            raise SynthesisError("fish_speech needs a reference clip")

        tokens = reference.precomputed.get(PROMPT_TOKEN_KEY)
        if tokens is None and self.require_precomputed_tokens:
            raise SynthesisError(
                f"reference {reference.reference_id!r} has no {PROMPT_TOKEN_KEY}; "
                "run the reference-bank token extraction step before benchmarking "
                "fish_speech, or set require_precomputed_tokens=false to allow the "
                "much slower per-request extraction"
            )

        kwargs: dict[str, Any] = dict(self.defaults)
        kwargs.update(
            {
                "text": request.text,
                "prompt_text": reference.transcript,
                "seed": seed,
            }
        )
        if tokens is not None:
            kwargs["prompt_tokens"] = tokens
        else:
            kwargs["reference_audio"] = reference.audio_path
        return kwargs
