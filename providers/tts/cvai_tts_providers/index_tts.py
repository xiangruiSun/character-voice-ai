"""IndexTTS-2.5 adapter (optional candidate, Milestone 5b).

Not in the spec's original three, added after the technology survey because it is the
only surveyed engine that **decouples emotion from speaker identity**: an 8-dimensional
emotion vector with an ``emo_alpha`` strength, plus ``duration_factor`` for pacing. That
is a direct match for spec §5's insistence that a character is prosody and performance,
not just timbre — and it is the one axis GPT-SoVITS structurally cannot express.

Two caveats, both recorded rather than assumed away:

* No documented fine-tuning path. It competes as zero-shot only, which under decision D4
  means it is reported in a separate row and never compared head-to-head with a
  fine-tuned engine as if the comparison were fair.
* Its licence is the bilibili Model Use License, not an OSI licence, so
  ``commercial_use`` is ``None`` until someone reads it.
"""

from __future__ import annotations

from typing import Any

from cvai_core.errors import SynthesisError
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import (
    AdaptationMode,
    CoreStyle,
    EmotionVector,
    ProviderCapabilities,
    TTSRequest,
)

from .sidecar import GenericSidecarProvider


@TTS_PROVIDERS.register("index_tts")
class IndexTTSProvider(GenericSidecarProvider):
    engine = "index_tts"
    call_name = "infer"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9882",
        *,
        engine_version: str = "IndexTTS-2.5",
        native_sample_rate: int = 24000,
        emo_alpha: float = 0.8,
        use_emo_text: bool = False,
        derive_emotion_vector: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, engine_version=engine_version, **kwargs)
        self.native_sample_rate = native_sample_rate
        self.emo_alpha = emo_alpha
        #: Let the engine infer emotion from the script. Off by default: the planner has
        #: already decided the performance, and letting the engine re-decide it produces
        #: readings that contradict the character's intent.
        self.use_emo_text = use_emo_text
        #: Derive a vector from the neutral controls when the plan has none. The mapping
        #: is coarse; per-style overrides in the voice pack are the accurate source.
        self.derive_emotion_vector = derive_emotion_vector

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[AdaptationMode.ZERO_SHOT],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            requires_reference_text=False,
            supports_multiple_references=False,
            supports_instruct=False,
            supports_emotion_vector=True,
            supports_duration_control=True,
            supports_speed_factor=False,
            supports_seed=True,
            supports_streaming=False,
            supports_hot_checkpoint_swap=False,
            license="bilibili Model Use License (not OSI)",
            commercial_use=None,
        )

    def native_kwargs(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        reference = request.reference
        if reference is None:
            raise SynthesisError("index_tts needs a speaker prompt clip")

        controls = request.controls
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": reference.audio_path,
            "text": request.text,
            "lang": "ZH",
            # 0.5-2.0 in the engine; our speed factor is the inverse sense (higher =
            # faster = shorter), so invert it here rather than in the sidecar.
            "duration_factor": round(
                min(2.0, max(0.5, 1.0 / max(0.5, controls.speed_factor()))), 4
            ),
            "emo_alpha": self.emo_alpha * max(0.2, controls.emotion_intensity),
            "use_emo_text": self.use_emo_text,
        }

        vector = controls.emotion_vector
        if vector is None and self.derive_emotion_vector:
            vector = EmotionVector.from_core_style(
                _core_style_of(reference.core_style), controls.emotion_intensity
            )
        if vector is not None and not vector.is_empty():
            kwargs["emo_vector"] = vector.as_list()
        return kwargs


def _core_style_of(value: CoreStyle | str) -> CoreStyle:
    if isinstance(value, CoreStyle):
        return value
    try:
        return CoreStyle(value)
    except ValueError:
        return CoreStyle.NEUTRAL
