"""Backend selection.

The pipeline asks for a capability; this module supplies the best implementation that is
actually installed, and says which one it picked. Selection is reported rather than
silent, because "the emotion labels are all `unknown`" has two very different causes —
the model said so, or the model was never installed — and the run report has to
distinguish them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cvai_core.logging_setup import get_logger

from .base import (
    ASRBackend,
    ASRResult,
    Backend,
    BackendInfo,
    CharTiming,
    DenoiseBackend,
    EmotionBackend,
    EmotionResult,
    SeparationBackend,
    SpeakerBackend,
    UnavailableBackend,
    VADBackend,
)
from .builtin import (
    STUB_TRANSCRIPT_SOURCE,
    EnergyVADBackend,
    NullEmotionBackend,
    SpectralSpeakerBackend,
    StubASRBackend,
)
from .cleaning import (
    DEFAULT_VOCAL_MODEL,
    CopyOnlyCleaningBackend,
    DeepFilterNetBackend,
    UVRSeparationBackend,
)
from .funasr_backends import (
    EMOTION2VEC_TO_CORE_STYLE,
    CamPlusPlusSpeakerBackend,
    Emotion2VecBackend,
    FunASRTranscriber,
    normalize_emotion_label,
)
from .vad import SileroVADBackend

log = get_logger(__name__)


@dataclass
class BackendBundle:
    """Every backend one pipeline run will use."""

    vad: VADBackend
    asr: ASRBackend
    speaker: SpeakerBackend
    emotion: EmotionBackend
    separation: SeparationBackend | None = None
    denoise: DenoiseBackend | None = None
    #: Set when ASR fell back to the stub. The dataset builder refuses to ship these.
    asr_is_stub: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict[str, BackendInfo]:
        chosen: dict[str, BackendInfo] = {
            "vad": self.vad.info(),
            "asr": self.asr.info(),
            "speaker": self.speaker.info(),
            "emotion": self.emotion.info(),
        }
        if self.separation is not None:
            chosen["separation"] = self.separation.info()
        if self.denoise is not None:
            chosen["denoise"] = self.denoise.info()
        return chosen

    def render_report(self) -> str:
        lines = ["Backends"]
        for role, info in self.report().items():
            mark = "✓" if info.available else "·"
            extra = f"  ({info.detail})" if info.detail else ""
            lines.append(f"  {mark} {role:<11} {info.name} {info.version}{extra}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def _pick(candidates: list[Backend], fallback: Backend) -> tuple[Backend, str | None]:
    for candidate in candidates:
        if candidate.available():
            return candidate, None
    return fallback, candidates[0].unavailable_reason() if candidates else None


def resolve_backends(
    *,
    prefer_device: str = "cpu",
    enable_separation: bool = False,
    enable_denoise: bool = False,
    allow_stub_asr: bool = True,
    asr_model: str = "paraformer-zh",
    emotion_model: str = "iic/emotion2vec_plus_large",
    separation_model: str = DEFAULT_VOCAL_MODEL,
    speech_pad_ms: int = 180,
    denoise_attenuation_db: float = 12.0,
) -> BackendBundle:
    """Choose the best available backend for each capability."""
    notes: list[str] = []

    vad_candidates: list[Backend] = [SileroVADBackend(speech_pad_ms=speech_pad_ms)]
    vad, vad_reason = _pick(
        vad_candidates, EnergyVADBackend(speech_pad_ms=float(speech_pad_ms))
    )
    if vad_reason:
        notes.append(f"VAD: {vad_reason} — using the built-in energy gate")

    asr_candidates: list[Backend] = [
        FunASRTranscriber(model=asr_model, device=prefer_device)
    ]
    asr_is_stub = False
    asr, asr_reason = _pick(asr_candidates, StubASRBackend())
    if asr_reason:
        asr_is_stub = True
        if not allow_stub_asr:
            raise RuntimeError(
                f"no ASR backend available and stub ASR is disallowed: {asr_reason}"
            )
        notes.append(
            f"ASR: {asr_reason} — using the STUB transcriber. Transcripts will be "
            "placeholders and the dataset builder will refuse to ship them."
        )

    speaker, speaker_reason = _pick(
        [CamPlusPlusSpeakerBackend(device=prefer_device)], SpectralSpeakerBackend()
    )
    if speaker_reason:
        notes.append(
            f"Speaker: {speaker_reason} — using the built-in spectral embedding, "
            "which is advisory only and will not reliably separate similar voices"
        )

    emotion, emotion_reason = _pick(
        [Emotion2VecBackend(model=emotion_model, device=prefer_device)],
        NullEmotionBackend(),
    )
    if emotion_reason:
        notes.append(f"Emotion: {emotion_reason} — styles must be labelled by hand")

    separation: SeparationBackend | None = None
    if enable_separation:
        candidate = UVRSeparationBackend(model_filename=separation_model)
        if candidate.available():
            separation = candidate
        else:
            separation = CopyOnlyCleaningBackend()
            notes.append(
                f"Separation is enabled but {candidate.unavailable_reason()} — "
                "files are copied through unchanged and recorded as such"
            )

    denoise_backend: DenoiseBackend | None = None
    if enable_denoise:
        candidate = DeepFilterNetBackend(attenuation_db=denoise_attenuation_db)
        if candidate.available():
            denoise_backend = candidate
        else:
            denoise_backend = CopyOnlyCleaningBackend()
            notes.append(
                f"Denoising is enabled but {candidate.unavailable_reason()} — "
                "files are copied through unchanged and recorded as such"
            )

    bundle = BackendBundle(
        vad=vad,  # type: ignore[arg-type]
        asr=asr,  # type: ignore[arg-type]
        speaker=speaker,  # type: ignore[arg-type]
        emotion=emotion,  # type: ignore[arg-type]
        separation=separation,
        denoise=denoise_backend,
        asr_is_stub=asr_is_stub,
        notes=notes,
    )
    for note in notes:
        log.warning(note)
    return bundle


__all__ = [
    "ASRBackend",
    "ASRResult",
    "Backend",
    "BackendBundle",
    "BackendInfo",
    "CamPlusPlusSpeakerBackend",
    "CharTiming",
    "CopyOnlyCleaningBackend",
    "DEFAULT_VOCAL_MODEL",
    "DeepFilterNetBackend",
    "DenoiseBackend",
    "EMOTION2VEC_TO_CORE_STYLE",
    "Emotion2VecBackend",
    "EmotionBackend",
    "EmotionResult",
    "EnergyVADBackend",
    "FunASRTranscriber",
    "NullEmotionBackend",
    "STUB_TRANSCRIPT_SOURCE",
    "SeparationBackend",
    "SileroVADBackend",
    "SpeakerBackend",
    "SpectralSpeakerBackend",
    "StubASRBackend",
    "UVRSeparationBackend",
    "UnavailableBackend",
    "VADBackend",
    "normalize_emotion_label",
    "resolve_backends",
]
