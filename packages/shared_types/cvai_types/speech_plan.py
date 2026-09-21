"""Character Speech Planner output (spec §12).

The LLM is a performance director, not just a writer: it returns the line *and* how the
line should be delivered. The user only ever sees ``text``; everything else is routed to
the voice stack.

This schema is also the LLM's response format — it is emitted as a JSON schema for
structured output in Milestone 8, which is why every field is either a small enum or a
bounded scalar. Free-form strings in a structured LLM response are where drift starts.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import CVAIModel, Language, StyleTag, UnitFloat
from .style import (
    EmotionVector,
    EndingStyle,
    PauseStyle,
    SpeakingRate,
    StyleControls,
    VolumeStyle,
)


class CharacterSpeechPlan(CVAIModel):
    """One planned utterance."""

    text: str = Field(min_length=1, max_length=2000)
    language: Language = "zh-CN"

    emotion: StyleTag = "neutral"
    emotion_intensity: UnitFloat = 0.5
    speaking_rate: SpeakingRate = SpeakingRate.NORMAL
    volume_style: VolumeStyle = VolumeStyle.NORMAL
    pause_style: PauseStyle = PauseStyle.NATURAL
    ending_style: EndingStyle = EndingStyle.FALLING
    reference_style: StyleTag | None = None

    #: Optional richer directions for engines that accept them. The planner may leave
    #: these unset; adapters derive sensible values from the fields above.
    instruct: str | None = Field(default=None, max_length=300)
    emotion_vector: EmotionVector | None = None

    #: Planner's own note on why it chose this delivery. Never spoken, never shown to the
    #: user; kept because it is the single most useful signal when a character starts
    #: sounding wrong and you need to know whether the planner or the voice is at fault.
    direction_note: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _fill_reference_style(self) -> "CharacterSpeechPlan":
        if self.reference_style is None:
            object.__setattr__(self, "reference_style", self.emotion)
        return self

    def to_style_controls(self) -> StyleControls:
        """Project the plan onto the engine-neutral control payload (decision D2)."""
        return StyleControls(
            emotion=self.emotion,
            emotion_intensity=self.emotion_intensity,
            speaking_rate=self.speaking_rate,
            volume_style=self.volume_style,
            pause_style=self.pause_style,
            ending_style=self.ending_style,
            reference_style=self.reference_style,
            instruct=self.instruct,
            emotion_vector=self.emotion_vector,
        )


class SpeechChunk(CVAIModel):
    """A TTS-sized piece of a planned utterance (spec §14).

    Produced by the chunker in Milestone 12. Tokens are never sent to TTS individually and
    a whole paragraph is never sent as one request — both are listed as failure modes in
    spec §27.
    """

    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=500)
    #: Normalized, TTS-ready text. Populated by the Chinese text normalizer; ``None``
    #: means normalization has not run yet and the chunk must not be synthesized.
    normalized_text: str | None = Field(default=None, max_length=1000)
    is_final: bool = False
    #: Inherited from the parent plan so each chunk is independently synthesizable.
    controls: StyleControls = Field(default_factory=StyleControls)

    @property
    def synthesis_text(self) -> str:
        if self.normalized_text is None:
            raise ValueError(
                f"chunk {self.chunk_index} has not been normalized; "
                "run the Chinese text normalizer before synthesis"
            )
        return self.normalized_text
