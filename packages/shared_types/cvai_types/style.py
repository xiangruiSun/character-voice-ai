"""Style and performance taxonomy.

Spec §5 is explicit that a character voice is not a speaker embedding: it is timbre *plus*
pitch distribution, rate, rhythm, prosody, pauses, endings, breathiness, texture,
pronunciation habits, emotion and volume dynamics. This module is where that list becomes
types.

Two levels exist on purpose (spec §9):

* **Core styles** — a small fixed vocabulary shared by every character, so cross-character
  tooling and the retriever fallback chain always have something to land on.
* **Character styles** — per-pack tags such as ``soft_teasing`` that carry the real
  performance nuance. Each one declares which core style it falls back to.

``StyleControls`` is the engine-neutral payload that travels from the Speech Planner to a
TTS adapter (decision D2). Engines express style very differently — GPT-SoVITS only
through the choice of reference clip, IndexTTS-2.5 through an 8-dimensional emotion
vector, Qwen3-TTS and CosyVoice3 through natural-language ``instruct`` strings — so the
payload is a superset and each adapter maps the parts it supports.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import Field, model_validator

from .base import CVAIModel, StyleTag, UnitFloat


class CoreStyle(str, Enum):
    """The V1 core taxonomy (spec §9). Deliberately small."""

    NEUTRAL = "neutral"
    SOFT = "soft"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    TEASING = "teasing"
    SERIOUS = "serious"
    SURPRISED = "surprised"
    WHISPER = "whisper"
    EXCITED = "excited"


CORE_STYLES: Final[tuple[str, ...]] = tuple(s.value for s in CoreStyle)


class SpeakingRate(str, Enum):
    VERY_SLOW = "very_slow"
    SLOW = "slow"
    SLIGHTLY_SLOW = "slightly_slow"
    NORMAL = "normal"
    SLIGHTLY_FAST = "slightly_fast"
    FAST = "fast"
    VERY_FAST = "very_fast"


#: Numeric speed factor per rate bucket. Engines that take a float (GPT-SoVITS
#: ``speed_factor``, IndexTTS-2.5 ``duration_factor``) read this table instead of each
#: adapter inventing its own mapping.
SPEAKING_RATE_FACTOR: Final[dict[SpeakingRate, float]] = {
    SpeakingRate.VERY_SLOW: 0.75,
    SpeakingRate.SLOW: 0.85,
    SpeakingRate.SLIGHTLY_SLOW: 0.93,
    SpeakingRate.NORMAL: 1.0,
    SpeakingRate.SLIGHTLY_FAST: 1.07,
    SpeakingRate.FAST: 1.15,
    SpeakingRate.VERY_FAST: 1.3,
}


class PitchProfile(str, Enum):
    LOW = "low"
    MID_LOW = "mid_low"
    MID = "mid"
    MID_HIGH = "mid_high"
    HIGH = "high"


class PauseStyle(str, Enum):
    CLIPPED = "clipped"
    NATURAL = "natural"
    DRAWN_OUT = "drawn_out"
    HESITANT = "hesitant"


class VolumeStyle(str, Enum):
    WHISPER = "whisper"
    SOFT = "soft"
    NORMAL = "normal"
    LOUD = "loud"
    SHOUT = "shout"


class EndingStyle(str, Enum):
    """How the last syllable of a line lands. A strong character-identity cue."""

    FALLING = "falling"
    FLAT = "flat"
    RISING = "rising"
    TRAILING_OFF = "trailing_off"
    CUT_OFF = "cut_off"


# --------------------------------------------------------------------------------------
# Emotion vector
# --------------------------------------------------------------------------------------

#: Dimension order matches IndexTTS-2.5's ``emo_vector`` so the adapter is a direct
#: pass-through. Other engines ignore it or derive an ``instruct`` string from it.
EMOTION_VECTOR_DIMS: Final[tuple[str, ...]] = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


class EmotionVector(CVAIModel):
    """Continuous emotion mix, each dimension in [0, 1]."""

    happy: UnitFloat = 0.0
    angry: UnitFloat = 0.0
    sad: UnitFloat = 0.0
    afraid: UnitFloat = 0.0
    disgusted: UnitFloat = 0.0
    melancholic: UnitFloat = 0.0
    surprised: UnitFloat = 0.0
    calm: UnitFloat = 0.0

    def as_list(self) -> list[float]:
        """In ``EMOTION_VECTOR_DIMS`` order, ready for an engine that wants a list."""
        return [getattr(self, dim) for dim in EMOTION_VECTOR_DIMS]

    def is_empty(self) -> bool:
        return all(v == 0.0 for v in self.as_list())

    @classmethod
    def from_core_style(cls, style: CoreStyle, intensity: float = 0.6) -> "EmotionVector":
        """Coarse core-style → vector bootstrap.

        This exists so an engine with explicit emotion control is usable before anyone has
        hand-tuned vectors for a character. It is a starting point for the benchmark, not
        a claim about the character's performance; per-character overrides belong in the
        voice pack's style definitions.
        """
        intensity = max(0.0, min(1.0, intensity))
        mapping: dict[CoreStyle, dict[str, float]] = {
            CoreStyle.NEUTRAL: {"calm": 0.5},
            CoreStyle.SOFT: {"calm": 0.7},
            CoreStyle.HAPPY: {"happy": 1.0},
            CoreStyle.SAD: {"sad": 1.0},
            CoreStyle.ANGRY: {"angry": 1.0},
            CoreStyle.TEASING: {"happy": 0.6, "calm": 0.3},
            CoreStyle.SERIOUS: {"calm": 0.8},
            CoreStyle.SURPRISED: {"surprised": 1.0},
            CoreStyle.WHISPER: {"calm": 0.6},
            CoreStyle.EXCITED: {"happy": 0.8, "surprised": 0.4},
        }
        weights = mapping.get(style, {"calm": 0.5})
        return cls(**{dim: round(w * intensity, 4) for dim, w in weights.items()})


# --------------------------------------------------------------------------------------
# Style definition (voice-pack level) and controls (request level)
# --------------------------------------------------------------------------------------


class StyleDefinition(CVAIModel):
    """A style a specific character can actually perform.

    Declared in the voice pack manifest. ``core_style`` is the fallback used by the
    reference retriever and by cross-character tooling.
    """

    name: StyleTag
    core_style: CoreStyle
    description: str = Field(default="", max_length=500)
    #: A real line from the character that exemplifies this style. Doubles as
    #: documentation for human annotators and as few-shot material for the planner.
    example_line: str = Field(default="", max_length=300)
    #: Per-character emotion vector override for engines that accept one.
    emotion_vector: EmotionVector | None = None
    #: Per-character natural-language instruction for engines that accept one
    #: (Qwen3-TTS / CosyVoice3 ``instruct``). Written in Chinese.
    instruct: str | None = Field(default=None, max_length=300)
    typical_speaking_rate: SpeakingRate = SpeakingRate.NORMAL
    typical_pitch_profile: PitchProfile = PitchProfile.MID


class StyleControls(CVAIModel):
    """Engine-neutral performance directions for one utterance (decision D2).

    Produced by the Character Speech Planner (spec §12), consumed by ``TTSProvider``
    adapters, each of which maps what it can and declares the rest unsupported through
    ``ProviderCapabilities``.
    """

    emotion: StyleTag = "neutral"
    emotion_intensity: UnitFloat = 0.5
    speaking_rate: SpeakingRate = SpeakingRate.NORMAL
    volume_style: VolumeStyle = VolumeStyle.NORMAL
    pause_style: PauseStyle = PauseStyle.NATURAL
    ending_style: EndingStyle = EndingStyle.FALLING
    #: Which Reference Bank style to draw a clip from. Defaults to ``emotion`` but is
    #: separate on purpose: a line can be *written* as embarrassed while being best
    #: *performed* from the ``soft_teasing`` reference clips. Leave unset to follow
    #: ``emotion``; set it explicitly to decouple the two.
    reference_style: StyleTag | None = None
    #: Optional natural-language direction, Chinese, for instruct-capable engines.
    instruct: str | None = Field(default=None, max_length=300)
    #: Optional explicit emotion mix for vector-capable engines.
    emotion_vector: EmotionVector | None = None
    #: Optional explicit duration scaling. When ``None``, adapters derive it from
    #: ``speaking_rate`` via ``SPEAKING_RATE_FACTOR``.
    duration_factor: float | None = Field(default=None, ge=0.5, le=2.0)

    @model_validator(mode="after")
    def _default_reference_style(self) -> "StyleControls":
        if self.reference_style is None:
            # Follow ``emotion`` unless the planner decoupled them explicitly.
            object.__setattr__(self, "reference_style", self.emotion)
        return self

    @property
    def effective_reference_style(self) -> str:
        """Never ``None`` after validation; typed accessor for adapters."""
        return self.reference_style or self.emotion

    def speed_factor(self) -> float:
        """The float an engine wants, honouring an explicit ``duration_factor``."""
        if self.duration_factor is not None:
            return self.duration_factor
        return SPEAKING_RATE_FACTOR[self.speaking_rate]


def resolve_style_chain(
    requested: str,
    style_definitions: dict[str, StyleDefinition],
) -> list[str]:
    """Fallback chain for a requested style: exact → core style → ``neutral``.

    Returned entries are de-duplicated and ordered by preference. The retriever walks this
    chain, which is how a character-specific tag like ``embarrassed_teasing`` still finds
    audio when only ``teasing`` clips exist.
    """
    chain: list[str] = []

    def _add(value: str) -> None:
        if value and value not in chain:
            chain.append(value)

    _add(requested)
    definition = style_definitions.get(requested)
    if definition is not None:
        _add(definition.core_style.value)
    elif requested in CORE_STYLES:
        pass  # already a core style
    _add(CoreStyle.NEUTRAL.value)
    return chain
