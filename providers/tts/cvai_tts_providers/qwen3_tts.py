"""Qwen3-TTS adapter (Milestone 4).

Drives ``Qwen3TTSModel.generate_voice_clone(text, language, ref_audio, ref_text)`` from
the ``qwen-tts`` package through a sidecar. Notes that shaped this adapter:

* ``ref_text`` is part of the call, which is why the Reference Bank stores transcripts
  (decision D3).
* ``create_voice_clone_prompt(ref_audio, ref_text)`` builds a reusable prompt object.
  The sidecar caches one per reference id — for this engine that object *is* the "keep
  the model warm" optimisation, and re-deriving it per sentence is the equivalent of
  reloading from disk (spec §27). The adapter signals which clip to cache under via
  ``voice_clone_prompt_key``.
* The engine takes a natural-language ``instruct`` string, so character style survives as
  more than a choice of reference clip. The instruction is assembled in Chinese from the
  neutral controls plus the pack's per-style override.
* Licence: Apache-2.0, commercial use permitted — which, with first-class fine-tuning
  support, makes this the most operationally comfortable of the strong candidates.

**Status:** written against the documented API; validated against a live sidecar in
Milestone 4.
"""

from __future__ import annotations

from typing import Any

from cvai_core.registry import TTS_PROVIDERS
from cvai_types import (
    AdaptationMode,
    ProviderCapabilities,
    SpeakingRate,
    TTSRequest,
    VolumeStyle,
)

from .sidecar import GenericSidecarProvider

#: Qwen3-TTS names languages in words, not BCP-47 tags.
_LANGUAGE = "Chinese"

_RATE_WORDS: dict[SpeakingRate, str] = {
    SpeakingRate.VERY_SLOW: "语速很慢",
    SpeakingRate.SLOW: "语速慢",
    SpeakingRate.SLIGHTLY_SLOW: "语速略慢",
    SpeakingRate.NORMAL: "语速正常",
    SpeakingRate.SLIGHTLY_FAST: "语速略快",
    SpeakingRate.FAST: "语速快",
    SpeakingRate.VERY_FAST: "语速很快",
}

_VOLUME_WORDS: dict[VolumeStyle, str] = {
    VolumeStyle.WHISPER: "用气声轻声说",
    VolumeStyle.SOFT: "音量轻柔",
    VolumeStyle.NORMAL: "音量正常",
    VolumeStyle.LOUD: "音量偏大",
    VolumeStyle.SHOUT: "大声喊",
}


@TTS_PROVIDERS.register("qwen3_tts")
class QwenTTSProvider(GenericSidecarProvider):
    engine = "qwen3_tts"
    call_name = "generate_voice_clone"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9881",
        *,
        engine_version: str = "Qwen3-TTS-12Hz-1.7B-Base",
        native_sample_rate: int = 24000,
        x_vector_only_mode: bool = False,
        auto_instruct: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, engine_version=engine_version, **kwargs)
        self.native_sample_rate = native_sample_rate
        self.x_vector_only_mode = x_vector_only_mode
        #: When true, build an ``instruct`` string from the neutral controls if the
        #: planner did not supply one. Turn it off to benchmark the engine with reference
        #: conditioning alone, which is the like-for-like comparison against GPT-SoVITS.
        self.auto_instruct = auto_instruct

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[
                AdaptationMode.ZERO_SHOT,
                AdaptationMode.FINETUNED,
                AdaptationMode.LORA,
            ],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            requires_reference_text=not self.x_vector_only_mode,
            supports_multiple_references=False,
            supports_instruct=True,
            supports_emotion_vector=False,
            supports_duration_control=False,
            supports_speed_factor=False,
            supports_seed=True,
            supports_streaming=True,
            supports_hot_checkpoint_swap=True,
            license="Apache-2.0",
            commercial_use=True,
        )

    def native_kwargs(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": request.text,
            "language": _LANGUAGE,
        }
        if request.reference is not None:
            kwargs["ref_audio"] = request.reference.audio_path
            kwargs["ref_text"] = request.reference.transcript
            # Tells the sidecar which cached voice_clone_prompt to reuse.
            kwargs["voice_clone_prompt_key"] = request.reference.reference_id
        if self.x_vector_only_mode:
            kwargs["x_vector_only_mode"] = True

        instruct = request.controls.instruct
        if instruct is None and self.auto_instruct:
            instruct = build_instruct(request)
        if instruct:
            kwargs["instruct"] = instruct
        return kwargs


def build_instruct(request: TTSRequest) -> str:
    """Compose a Chinese performance instruction from the neutral controls.

    Kept short and concrete. Long prompts make instruct-capable engines drift away from
    the reference timbre, which trades the thing we care about most (character identity)
    for the thing we care about less (a slightly more emphatic reading).
    """
    controls = request.controls
    parts: list[str] = []
    style = controls.emotion.replace("_", " ")
    intensity = controls.emotion_intensity
    degree = "略带" if intensity < 0.35 else ("明显" if intensity > 0.7 else "")
    parts.append(f"{degree}{style}的语气".strip())
    parts.append(_RATE_WORDS[controls.speaking_rate])
    if controls.volume_style is not VolumeStyle.NORMAL:
        parts.append(_VOLUME_WORDS[controls.volume_style])
    return "，".join(parts)
