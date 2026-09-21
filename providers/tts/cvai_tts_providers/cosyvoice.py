"""CosyVoice / Fun-CosyVoice3 adapter (optional candidate, Milestone 5c).

Added after the technology survey. Its argument for inclusion is not zero-shot quality
(several candidates match it) but **streaming**: bidirectional text-in/audio-out with
~150 ms latency, plus Apache-2.0 weights and in-repo training code. If two engines tie on
character fidelity in Milestone 7, the one that streams natively wins Milestones 12-13
outright.

Style reaches it as a natural-language ``instruct`` string, same as Qwen3-TTS, so the two
share the instruction builder.
"""

from __future__ import annotations

from typing import Any

from cvai_core.errors import SynthesisError
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import AdaptationMode, ProviderCapabilities, TTSRequest

from .qwen3_tts import build_instruct
from .sidecar import GenericSidecarProvider


@TTS_PROVIDERS.register("cosyvoice")
class CosyVoiceProvider(GenericSidecarProvider):
    engine = "cosyvoice"
    call_name = "inference_zero_shot"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9883",
        *,
        engine_version: str = "Fun-CosyVoice3-0.5B",
        native_sample_rate: int = 24000,
        auto_instruct: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, engine_version=engine_version, **kwargs)
        self.native_sample_rate = native_sample_rate
        self.auto_instruct = auto_instruct

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[
                AdaptationMode.ZERO_SHOT,
                AdaptationMode.FINETUNED,
            ],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            requires_reference_text=True,  # prompt_text accompanies prompt_wav
            supports_multiple_references=False,
            supports_instruct=True,
            supports_emotion_vector=False,
            supports_duration_control=False,
            supports_speed_factor=True,
            supports_seed=True,
            supports_streaming=True,
            supports_hot_checkpoint_swap=True,
            license="Apache-2.0",
            commercial_use=True,
        )

    def native_kwargs(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        reference = request.reference
        if reference is None:
            raise SynthesisError("cosyvoice needs a prompt clip")

        kwargs: dict[str, Any] = {
            "tts_text": request.text,
            "prompt_text": reference.transcript,
            "prompt_wav": reference.audio_path,
            "stream": bool(request.stream),
            "speed": round(request.controls.speed_factor(), 4),
        }
        instruct = request.controls.instruct
        if instruct is None and self.auto_instruct:
            instruct = build_instruct(request)
        if instruct:
            # The sidecar routes to inference_instruct2 when an instruction is present.
            kwargs["instruct_text"] = instruct
        return kwargs
