"""Character Profile — the "what the character says" half of the system (spec §11).

Kept strictly separate from the voice: a profile references a voice pack by id and never
embeds audio, checkpoints or engine settings. Swapping the TTS engine must not touch a
character profile, and rewriting a character's personality must not invalidate a trained
voice model.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from .base import CVAIModel, Language, Slug, StyleTag
from .style import CoreStyle


class DialogueExample(CVAIModel):
    """An original line, ideally lifted verbatim from the source material.

    These are the highest-value content in the profile: spec §11 wants prompting plus
    retrieval rather than LLM fine-tuning for V1, and original dialogue is what makes
    prompting work. ``user`` may be empty for a standalone line (a battle cry, an idle
    barks line).
    """

    user: str = Field(default="", max_length=500)
    character: str = Field(min_length=1, max_length=1000)
    style: StyleTag = "neutral"
    #: Where this line came from — quest name, cutscene id, voice file name. Needed to
    #: audit that examples really are original and not paraphrased by an LLM.
    source: str = Field(default="", max_length=200)


class SpeakingHabits(CVAIModel):
    """How the character constructs speech, independent of voice timbre."""

    typical_sentence_length: str = Field(
        default="medium",
        description="short | medium | long, or a free description",
        max_length=120,
    )
    #: Catchphrases and recurring constructions. Used in the system prompt and, later, to
    #: check whether generated lines drift away from the character's register.
    frequent_expressions: list[str] = Field(default_factory=list)
    #: Fillers, laughs, sighs — written as they should appear in text for the TTS front-end.
    verbal_tics: list[str] = Field(default_factory=list)
    #: Character-specific pronunciation, ``{"重": "chong2"}``. Feeds the Chinese text
    #: normalizer's per-character override lexicon in Milestone 8 rather than being
    #: hard-coded anywhere.
    pronunciation_overrides: dict[str, str] = Field(default_factory=dict)
    #: Punctuation and pacing habits, e.g. frequent use of "……".
    punctuation_habits: list[str] = Field(default_factory=list)


class MemorySettings(CVAIModel):
    enabled: bool = True
    #: Verbatim turns kept in the prompt before summarization kicks in.
    max_turns: int = Field(default=12, ge=0, le=200)
    summarize_after_turns: int = Field(default=20, ge=0, le=500)
    #: Long-term facts the character should always know about the user.
    persistent_facts: list[str] = Field(default_factory=list)


class VoiceBinding(CVAIModel):
    """Points a character at a voice, without describing the voice itself."""

    voicepack_id: Slug
    default_reference_style: StyleTag = "neutral"
    #: Filled in after the Milestone 7 decision. ``None`` means "whatever the config's
    #: default TTS provider is", which is how the benchmark drives every engine in turn.
    preferred_engine: str | None = None
    fallback_engine: str | None = None


class LLMSettings(CVAIModel):
    model: str = "gpt-4o"
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=400, ge=16, le=4000)
    #: Hard cap on generated line length. Long lines break the illusion and make TTS
    #: chunking worse; spec §27 lists "extremely long TTS chunks" as a failure mode.
    max_chars_per_reply: int = Field(default=120, ge=10, le=2000)


class CharacterProfile(CVAIModel):
    """Everything the Character Brain needs (spec §11)."""

    character_id: Slug
    character_name: str = Field(min_length=1, max_length=100)
    language: Language = "zh-CN"

    personality: list[str] = Field(
        default_factory=list,
        description="Short traits, each one sentence or less.",
    )
    background: str = Field(default="", max_length=4000)
    world_knowledge: list[str] = Field(
        default_factory=list,
        description="Facts the character knows about her world. Also acts as a boundary: "
        "anything outside this list she should not confidently assert.",
    )
    relationship_style: str = Field(
        default="",
        max_length=1000,
        description="How she treats the user specifically.",
    )
    speaking_habits: SpeakingHabits = Field(default_factory=SpeakingHabits)
    forbidden_behavior: list[str] = Field(
        default_factory=list,
        description="Things she must never do or say, including breaking character.",
    )
    dialogue_examples: list[DialogueExample] = Field(default_factory=list)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    voice: VoiceBinding
    llm: LLMSettings = Field(default_factory=LLMSettings)

    #: Styles this character is written to express, beyond the core taxonomy. Must be a
    #: subset of the styles her voice pack can actually perform — checked by the voice
    #: pack validator, because a planner that asks for a style with no reference audio
    #: silently degrades to neutral, which is exactly the "identical intonation" failure.
    available_styles: list[StyleTag] = Field(
        default_factory=lambda: [CoreStyle.NEUTRAL.value]
    )

    @field_validator("dialogue_examples")
    @classmethod
    def _require_examples_for_quality(
        cls, value: list[DialogueExample]
    ) -> list[DialogueExample]:
        # Not an error — a profile is often created before the lines are collected — but
        # the validator exists as the documented place to tighten this later.
        return value

    @field_validator("available_styles")
    @classmethod
    def _neutral_always_available(cls, value: list[StyleTag]) -> list[StyleTag]:
        if CoreStyle.NEUTRAL.value not in value:
            return [CoreStyle.NEUTRAL.value, *value]
        return value
