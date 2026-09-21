"""TTS request/result types and the provider capability model.

The capability model (decision D2) is what lets one interface serve six engines whose
style control mechanisms have almost nothing in common. An adapter declares what it can
honour; the router and the benchmark read that declaration instead of special-casing
engine names.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from .base import CVAIModel, Language, RelPath, StyleTag, UnitFloat, utcnow
from .style import CoreStyle, StyleControls


class AdaptationMode(str, Enum):
    """How a candidate engine was adapted to the character (decision D4).

    Benchmark results are keyed by ``(engine, adaptation_mode)``. Comparing a fine-tuned
    engine against a zero-shot one and declaring a winner is the fastest way to pick the
    wrong model.
    """

    ZERO_SHOT = "zero_shot"
    FINETUNED = "finetuned"
    LORA = "lora"


class ReferenceSelection(CVAIModel):
    """The clip the retriever picked, plus why."""

    reference_id: str
    #: Absolute or pack-relative path, resolved by the caller before it reaches an engine.
    audio_path: str
    #: Required by Qwen3-TTS / Fish Speech / VoxCPM2 and strongly beneficial for
    #: GPT-SoVITS. Carried here so adapters never have to go back to the bank.
    transcript: str
    language: Language = "zh-CN"
    style: StyleTag
    core_style: CoreStyle
    #: True when the requested style had no clips and the chain fell back. Logged, and
    #: surfaced in evaluation: a benchmark where half the lines silently fell back to
    #: neutral is measuring the wrong thing.
    was_fallback: bool = False
    requested_style: StyleTag | None = None
    #: Engine-specific precomputed artifacts for this clip (e.g. Fish Speech VQ tokens).
    precomputed: dict[str, RelPath] = Field(default_factory=dict)


class ProviderCapabilities(CVAIModel):
    """What a TTS adapter can actually honour."""

    engine: str = Field(min_length=1, max_length=60)
    engine_version: str = Field(default="unknown", max_length=60)
    supported_adaptation_modes: list[AdaptationMode] = Field(
        default_factory=lambda: [AdaptationMode.ZERO_SHOT]
    )
    native_sample_rate: int = Field(default=24000, ge=8000, le=48000)

    supports_reference_audio: bool = True
    #: Engine requires the reference transcript, not merely benefits from it.
    requires_reference_text: bool = False
    supports_multiple_references: bool = False
    supports_instruct: bool = False
    supports_emotion_vector: bool = False
    supports_duration_control: bool = False
    supports_speed_factor: bool = False
    supports_seed: bool = False
    supports_streaming: bool = False
    #: Engine can hot-swap character checkpoints without a process restart (e.g.
    #: GPT-SoVITS ``/set_gpt_weights``). Drives the "never load from disk per sentence"
    #: requirement in spec §27.
    supports_hot_checkpoint_swap: bool = False

    #: Licence of the model weights, and whether commercial use is permitted. Checked in
    #: the Milestone 7 decision so a licence-incompatible engine cannot win by accident.
    license: str = Field(default="unknown", max_length=120)
    commercial_use: bool | None = None

    def unsupported_controls(self, controls: StyleControls) -> list[str]:
        """Which parts of a request this engine will silently ignore.

        Callers log this. A silently dropped emotion vector is indistinguishable from a
        bad model at listening-test time, which makes the benchmark unreadable.
        """
        dropped: list[str] = []
        if controls.instruct and not self.supports_instruct:
            dropped.append("instruct")
        if controls.emotion_vector is not None and not self.supports_emotion_vector:
            dropped.append("emotion_vector")
        if controls.duration_factor is not None and not (
            self.supports_duration_control or self.supports_speed_factor
        ):
            dropped.append("duration_factor")
        return dropped


class TTSRequest(CVAIModel):
    """One synthesis request."""

    request_id: str = Field(min_length=1, max_length=120)
    #: TTS-ready text: already normalized by the Chinese text normalizer. Adapters must
    #: not normalize again — double normalization mangles numbers and Latin tokens.
    text: str = Field(min_length=1, max_length=1000)
    language: Language = "zh-CN"
    controls: StyleControls = Field(default_factory=StyleControls)
    reference: ReferenceSelection | None = None
    #: Which character checkpoint to use, when the engine supports adaptation.
    checkpoint_id: str | None = None
    adaptation_mode: AdaptationMode = AdaptationMode.ZERO_SHOT
    #: ``None`` means "engine default"; the benchmark always sets one (decision D5).
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    output_sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    stream: bool = False
    #: Escape hatch for engine-specific knobs during the benchmark. Recorded verbatim in
    #: the generation record, so an experiment stays reproducible even when it used a
    #: parameter the neutral schema does not model.
    engine_params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reference_needed_for_zero_shot(self) -> "TTSRequest":
        if self.adaptation_mode is AdaptationMode.ZERO_SHOT and self.reference is None:
            raise ValueError(
                "zero-shot synthesis needs a reference selection; "
                "for a fine-tuned character voice set adaptation_mode accordingly"
            )
        return self


class TTSResult(CVAIModel):
    """The audio plus everything needed to reproduce it (decision D5, spec §23)."""

    request_id: str
    audio_path: str
    sample_rate: int = Field(ge=8000, le=48000)
    duration_s: float = Field(gt=0.0)
    engine: str
    engine_version: str = "unknown"
    adaptation_mode: AdaptationMode = AdaptationMode.ZERO_SHOT
    checkpoint_id: str | None = None
    reference_id: str | None = None
    seed: int | None = None
    #: Exactly the parameters sent to the engine, after adapter mapping. This is the
    #: field that makes a result reproducible rather than merely described.
    resolved_params: dict[str, object] = Field(default_factory=dict)
    #: Controls the engine could not honour, from ``ProviderCapabilities``.
    dropped_controls: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())

    #: Optional real-time factor, latency_ms / (duration_s * 1000).
    @property
    def rtf(self) -> float:
        return (self.latency_ms / 1000.0) / self.duration_s if self.duration_s else 0.0


class TTSHealth(CVAIModel):
    """Result of an adapter readiness probe.

    Used by the benchmark runner to skip an engine cleanly instead of failing a whole run,
    and by the API to report which voices are live.
    """

    engine: str
    available: bool
    detail: str = ""
    checked_at: str = Field(default_factory=lambda: utcnow().isoformat())


class VoiceQualityHint(CVAIModel):
    """Optional per-engine tuning hints stored alongside a checkpoint.

    Populated during Milestones 3-5 as each engine is tuned for the character, so the
    winning configuration is carried forward into the application rather than living in
    someone's shell history.
    """

    engine: str
    recommended_params: dict[str, object] = Field(default_factory=dict)
    avoid_params: dict[str, object] = Field(default_factory=dict)
    min_reference_seconds: float = Field(default=3.0, gt=0.0)
    max_reference_seconds: float = Field(default=10.0, gt=0.0)
    notes: str = Field(default="", max_length=1000)
    confidence: UnitFloat = 0.5
