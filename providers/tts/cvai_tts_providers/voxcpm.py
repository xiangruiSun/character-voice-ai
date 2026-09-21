"""VoxCPM2 adapter (optional candidate, Milestone 5d).

Added after the survey for three practical reasons: 48 kHz native output (no upsampling
artefacts in the listening test), LoRA fine-tuning from 5-10 minutes of audio (a real
possibility if the Denia pack comes in under the 20-minute target), and Apache-2.0
weights. It also runs on CPU/MPS, which makes it the only candidate that can be smoke-
tested without a GPU.
"""

from __future__ import annotations

from typing import Any

from cvai_core.errors import SynthesisError
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import AdaptationMode, ProviderCapabilities, TTSRequest

from .qwen3_tts import build_instruct
from .sidecar import GenericSidecarProvider


@TTS_PROVIDERS.register("voxcpm")
class VoxCPMProvider(GenericSidecarProvider):
    engine = "voxcpm"
    call_name = "generate"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9884",
        *,
        engine_version: str = "VoxCPM2",
        native_sample_rate: int = 48000,
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
                AdaptationMode.LORA,
            ],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            # The transcript unlocks its highest-fidelity cloning path, so treat it as
            # required rather than optional.
            requires_reference_text=True,
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
        reference = request.reference
        if reference is None:
            raise SynthesisError("voxcpm needs a reference clip")

        kwargs: dict[str, Any] = {
            "text": request.text,
            "prompt_wav_path": reference.audio_path,
            "prompt_text": reference.transcript,
        }
        instruct = request.controls.instruct
        if instruct is None and self.auto_instruct:
            instruct = build_instruct(request)
        if instruct:
            kwargs["style_prompt"] = instruct
        return kwargs
