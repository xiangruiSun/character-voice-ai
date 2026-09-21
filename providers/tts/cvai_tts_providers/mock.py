"""Mock TTS engine — dependency-free, deterministic, and useful.

Its job is not to sound good. Its job is to make the whole pipeline — reference
retrieval, the benchmark runner, blind-test assembly, rating aggregation, the report —
runnable and testable on a laptop with no GPU, no model weights and no network, so that
when a real engine arrives in Milestone 3 the only new variable is the engine.

To be useful it has to behave like a real engine in the ways the harness cares about:

* duration tracks the text length, the speaking rate and the speed factor, so
  duration-based checks are exercised;
* output varies with the seed and with the chosen reference clip, so
  reference-rotation and determinism are both observable;
* pitch and loudness follow the style controls, so a mis-wired control path shows up as
  audibly identical files rather than passing silently;
* its capabilities are configurable, so the "engine cannot honour this control" path is
  covered by tests instead of only appearing once a real engine is plugged in.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from cvai_core.audio import estimate_speech_duration, write_wav
from cvai_core.errors import SynthesisError
from cvai_core.interfaces.tts import TTSProvider
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import (
    AdaptationMode,
    EndingStyle,
    PitchProfile,
    ProviderCapabilities,
    TTSHealth,
    TTSRequest,
    TTSResult,
    VolumeStyle,
)

from .base import TimedCall, derive_seed, finalize_result

#: Rough F0 centres in Hz. A female character voice sits in the upper half of this.
_PITCH_HZ: dict[PitchProfile, float] = {
    PitchProfile.LOW: 140.0,
    PitchProfile.MID_LOW: 180.0,
    PitchProfile.MID: 220.0,
    PitchProfile.MID_HIGH: 265.0,
    PitchProfile.HIGH: 320.0,
}

_VOLUME_GAIN: dict[VolumeStyle, float] = {
    VolumeStyle.WHISPER: 0.12,
    VolumeStyle.SOFT: 0.3,
    VolumeStyle.NORMAL: 0.6,
    VolumeStyle.LOUD: 0.82,
    VolumeStyle.SHOUT: 0.95,
}

#: Final-syllable F0 multiplier, so ending style is audible.
_ENDING_CONTOUR: dict[EndingStyle, float] = {
    EndingStyle.FALLING: 0.82,
    EndingStyle.FLAT: 1.0,
    EndingStyle.RISING: 1.25,
    EndingStyle.TRAILING_OFF: 0.9,
    EndingStyle.CUT_OFF: 1.05,
}

_PUNCTUATION = "，。！？；：、…,.!?;:"


@TTS_PROVIDERS.register("mock")
class MockTTSProvider(TTSProvider):
    """Deterministic synthetic speech for development and testing."""

    engine = "mock"

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        engine_version: str = "mock-1",
        supports_instruct: bool = False,
        supports_emotion_vector: bool = False,
        supports_duration_control: bool = True,
        supports_hot_checkpoint_swap: bool = True,
        simulated_latency_ms: float = 0.0,
        failure_rate: float = 0.0,
        harmonics: int = 3,
    ) -> None:
        self.sample_rate = sample_rate
        self.engine_version = engine_version
        self._supports_instruct = supports_instruct
        self._supports_emotion_vector = supports_emotion_vector
        self._supports_duration_control = supports_duration_control
        self._supports_hot_checkpoint_swap = supports_hot_checkpoint_swap
        self.simulated_latency_ms = simulated_latency_ms
        #: Deterministic-per-request failure injection, for exercising the runner's
        #: error path without needing a genuinely broken engine.
        self.failure_rate = failure_rate
        self.harmonics = max(1, harmonics)
        self.loaded_checkpoint: str | None = None

    # -- interface ----------------------------------------------------------------

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[
                AdaptationMode.ZERO_SHOT,
                AdaptationMode.FINETUNED,
                AdaptationMode.LORA,
            ],
            native_sample_rate=self.sample_rate,
            supports_reference_audio=True,
            requires_reference_text=False,
            supports_instruct=self._supports_instruct,
            supports_emotion_vector=self._supports_emotion_vector,
            supports_duration_control=self._supports_duration_control,
            supports_speed_factor=True,
            supports_seed=True,
            supports_streaming=False,
            supports_hot_checkpoint_swap=self._supports_hot_checkpoint_swap,
            license="MIT",
            commercial_use=True,
        )

    async def health(self) -> TTSHealth:
        return TTSHealth(engine=self.engine, available=True, detail="in-process mock")

    async def load_checkpoint(self, checkpoint_id: str) -> None:
        self.loaded_checkpoint = checkpoint_id

    async def synthesize(self, request: TTSRequest, output_path: Path) -> TTSResult:
        seed = derive_seed(request, salt=self.engine_version)

        if self.failure_rate > 0.0:
            # Deterministic: the same request always fails or always succeeds.
            if random.Random(seed ^ 0x5EED).random() < self.failure_rate:
                raise SynthesisError(
                    f"mock engine injected a failure for request {request.request_id!r}"
                )

        controls = request.controls
        speed = controls.speed_factor()
        duration_s = estimate_speech_duration(request.text, speed_factor=speed)
        sample_rate = request.output_sample_rate or self.sample_rate

        with TimedCall() as timer:
            samples = self._render(request, duration_s, sample_rate, seed)
            write_wav(output_path, samples, sample_rate)
        latency_ms = timer.elapsed_ms + self.simulated_latency_ms

        resolved = {
            "sample_rate": sample_rate,
            "speed_factor": round(speed, 4),
            "pitch_hz": round(self._base_f0(request, seed), 2),
            "volume_style": controls.volume_style.value,
            "ending_style": controls.ending_style.value,
            "reference_id": request.reference.reference_id if request.reference else None,
            "checkpoint_id": request.checkpoint_id,
            "loaded_checkpoint": self.loaded_checkpoint,
            "seed": seed,
            **request.engine_params,
        }

        return finalize_result(
            request,
            self.capabilities(),
            output_path,
            latency_ms=latency_ms,
            seed=seed,
            resolved_params=resolved,
            sample_rate=sample_rate,
            duration_s=duration_s,
        )

    # -- rendering ----------------------------------------------------------------

    def _base_f0(self, request: TTSRequest, seed: int) -> float:
        # Pitch follows the style; the reference clip shifts it, so two different
        # reference clips produce audibly different output. That is what makes
        # reference rotation visible in a listening test of the mock.
        controls = request.controls
        profile = PitchProfile.MID
        if controls.volume_style is VolumeStyle.WHISPER:
            profile = PitchProfile.MID_LOW
        elif controls.emotion_intensity > 0.7:
            profile = PitchProfile.MID_HIGH
        base = _PITCH_HZ[profile]
        offset = (random.Random(seed).random() - 0.5) * 24.0
        return max(80.0, base + offset)

    def _render(
        self, request: TTSRequest, duration_s: float, sample_rate: int, seed: int
    ) -> list[float]:
        rng = random.Random(seed)
        controls = request.controls
        total = int(duration_s * sample_rate)
        base_f0 = self._base_f0(request, seed)
        gain = _VOLUME_GAIN[controls.volume_style]
        breath = 0.05 + (0.25 if controls.volume_style is VolumeStyle.WHISPER else 0.0)

        # Syllable grid, so the envelope has speech-like rhythm rather than a flat tone.
        syllables = max(1, sum(1 for ch in request.text if not ch.isspace()))
        syllable_len = total / syllables
        pause_positions = {
            i for i, ch in enumerate(request.text) if ch in _PUNCTUATION
        }

        ending_mult = _ENDING_CONTOUR[controls.ending_style]
        vibrato_depth = 0.01 + 0.03 * controls.emotion_intensity

        samples: list[float] = []
        two_pi = 2.0 * math.pi
        for n in range(total):
            position = n / total if total else 0.0
            syllable_index = int(n / syllable_len) if syllable_len else 0
            within = (n % syllable_len) / syllable_len if syllable_len else 0.0

            # Envelope: attack-decay per syllable, silence where punctuation falls.
            if syllable_index in pause_positions:
                envelope = 0.0
            else:
                envelope = math.sin(math.pi * min(1.0, max(0.0, within))) ** 0.7

            # Pitch contour: slight declination across the line, plus the ending shape
            # applied over the last 15%.
            contour = 1.0 - 0.12 * position
            if position > 0.85:
                blend = (position - 0.85) / 0.15
                contour *= (1.0 - blend) + blend * ending_mult
            vibrato = 1.0 + vibrato_depth * math.sin(two_pi * 5.2 * n / sample_rate)
            f0 = base_f0 * contour * vibrato

            value = 0.0
            for harmonic in range(1, self.harmonics + 1):
                value += (1.0 / harmonic) * math.sin(
                    two_pi * f0 * harmonic * n / sample_rate
                )
            value /= self.harmonics
            value += breath * (rng.random() - 0.5)
            samples.append(value * envelope * gain)

        return samples
