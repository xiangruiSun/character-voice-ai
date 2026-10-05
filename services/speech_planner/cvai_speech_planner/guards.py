"""Validating and repairing a plan before it reaches the voice stack.

An LLM asked for structured output mostly complies. "Mostly" is the problem: it
occasionally invents a style the voice pack cannot perform, writes four sentences when
the character says one, or lets a line of narration slip in. None of those fail
loudly — they produce audio that is subtly not the character, which is the exact
outcome the project exists to avoid.

So every plan is checked and, where it can be, repaired. Repairs are recorded rather
than silent: a planner that quietly rewrites a quarter of its own output is a planner
whose prompt needs fixing, and that is only visible if the repairs are counted.
"""

from __future__ import annotations

import re
from typing import Sequence

from cvai_types import (
    CharacterProfile,
    CharacterSpeechPlan,
    CoreStyle,
    CVAIModel,
    SpeakingRate,
)
from pydantic import Field

#: Phrases that mean the model dropped character. Spec §27 lists "LLM responses that do
#: not sound like the character" as a failure mode; these are its most literal form.
BREAKING_CHARACTER: tuple[str, ...] = (
    "作为一个AI",
    "作为AI",
    "作为一个人工智能",
    "人工智能助手",
    "语言模型",
    "我是AI",
    "我无法",
    "As an AI",
)

#: Stage directions and narration an LLM adds when it forgets it is only writing speech.
NARRATION = re.compile(r"[（(\[【][^）)\]】]{0,40}[）)\]】]")

#: Sentence terminators used when truncating a reply that ran long.
TERMINATORS = "。！？…"


class GuardReport(CVAIModel):
    repairs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.repairs and not self.warnings


def apply_guards(
    plan: CharacterSpeechPlan,
    profile: CharacterProfile,
    *,
    available_styles: Sequence[str] | None = None,
    strip_narration: bool = True,
) -> tuple[CharacterSpeechPlan, GuardReport]:
    """Return a plan the voice stack can act on, plus what had to be changed."""
    report = GuardReport()
    updates: dict[str, object] = {}

    styles = set(available_styles or profile.available_styles)

    text = plan.text
    if strip_narration:
        stripped = NARRATION.sub("", text).strip()
        if stripped != text.strip():
            report.repairs.append("removed narration in brackets")
            # Deliberately allowed to empty the text: a reply that is *entirely* stage
            # direction has nothing to speak, and the empty-text check below turns that
            # into a visible warning rather than the engine reading "（沉默）" aloud.
            text = stripped

    for phrase in BREAKING_CHARACTER:
        if phrase.lower() in text.lower():
            report.warnings.append(
                f"the reply contains {phrase!r}, which breaks character — "
                "tighten the forbidden_behavior list in the profile"
            )
            break

    limit = profile.llm.max_chars_per_reply
    if len(text) > limit:
        if profile.llm.truncate_long_replies:
            text = _truncate_at_sentence(text, limit)
            report.repairs.append(f"truncated to {limit} characters at a sentence boundary")
        else:
            report.warnings.append(
                f"reply is {len(text)} characters, over the {limit} asked for; "
                "kept whole because truncate_long_replies is off"
            )

    if not text.strip():
        report.warnings.append("plan had no speakable text")
        text = "……"

    if text != plan.text:
        updates["text"] = text

    # --- style ---------------------------------------------------------------------

    emotion = plan.emotion
    if emotion not in styles:
        replacement = _nearest_style(emotion, styles)
        report.repairs.append(
            f"emotion {emotion!r} is not in the character's repertoire → {replacement!r}"
        )
        updates["emotion"] = replacement
        emotion = replacement

    reference = plan.reference_style or emotion
    if reference not in styles:
        replacement = _nearest_style(reference, styles)
        report.repairs.append(
            f"reference_style {reference!r} is unavailable → {replacement!r}"
        )
        reference = replacement
    updates["reference_style"] = reference

    # --- pacing sanity --------------------------------------------------------------

    if plan.volume_style.value == "whisper" and plan.speaking_rate in (
        SpeakingRate.FAST,
        SpeakingRate.VERY_FAST,
    ):
        # Fast whispering is not a performance any of the engines render convincingly;
        # it comes out as clipped noise.
        report.repairs.append("fast whisper is not renderable → slowed to normal")
        updates["speaking_rate"] = SpeakingRate.NORMAL

    return plan.model_copy(update=updates), report


def _truncate_at_sentence(text: str, limit: int) -> str:
    """Cut at the last sentence end within the limit; fall back to a hard cut.

    A reply cut mid-clause sounds like a dropped connection. Cutting at a full stop
    sounds like the character simply stopped talking, which is in character for almost
    everyone.
    """
    window = text[:limit]
    for index in range(len(window) - 1, 0, -1):
        if window[index] in TERMINATORS:
            return window[: index + 1]
    return window.rstrip() + "……"


def _nearest_style(style: str, available: set[str]) -> str:
    """Map an unavailable style onto something the pack can actually perform."""
    if style in available:
        return style
    # A character-specific tag usually embeds its core style: soft_teasing → teasing.
    for part in reversed(style.split("_")):
        if part in available:
            return part
    try:
        core = CoreStyle(style).value
        if core in available:
            return core
    except ValueError:
        pass
    return "neutral" if "neutral" in available else sorted(available)[0]
