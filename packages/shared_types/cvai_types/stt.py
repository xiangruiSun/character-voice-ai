"""Speech-to-text types.

Spec §3: start with OpenAI transcription, keep the interface open so faster-whisper or a
local FunASR streaming model can be substituted without touching the orchestrator.
"""

from __future__ import annotations

from pydantic import Field

from .base import CVAIModel, Language, UnitFloat


class TranscriptSegment(CVAIModel):
    text: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    confidence: UnitFloat | None = None


class Transcript(CVAIModel):
    text: str
    language: Language = "zh-CN"
    segments: list[TranscriptSegment] = Field(default_factory=list)
    #: True while an interim (partial) result is being streamed.
    is_partial: bool = False
    provider: str = "unknown"
    model: str = "unknown"
    latency_ms: float | None = Field(default=None, ge=0.0)
    duration_s: float | None = Field(default=None, ge=0.0)


class STTCapabilities(CVAIModel):
    provider: str
    model: str
    supports_streaming: bool = False
    supports_word_timestamps: bool = False
    #: Whether a domain vocabulary can be injected. Matters for character and world
    #: proper nouns, which generic ASR reliably mangles.
    supports_hotwords: bool = False
    languages: list[Language] = Field(default_factory=lambda: ["zh-CN"])
