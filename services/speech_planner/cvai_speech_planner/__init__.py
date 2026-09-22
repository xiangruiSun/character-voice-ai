"""Character Brain and Speech Planner (Milestone 8, spec §11 and §12).

Two halves of "what she says and how she says it":

* :mod:`~cvai_speech_planner.prompt` builds the system prompt from a
  :class:`CharacterProfile` — personality, lore, speaking habits, forbidden behaviour,
  and the original dialogue examples that actually carry the voice.
* :mod:`~cvai_speech_planner.planner` asks the LLM for a
  :class:`~cvai_types.speech_plan.CharacterSpeechPlan` — the line plus its delivery —
  and :mod:`~cvai_speech_planner.guards` makes sure what comes back is something the
  character's voice pack can actually perform.

No LLM fine-tuning (spec §11): prompting and retrieval only in V1.
"""

from __future__ import annotations

from cvai_types import CharacterSpeechPlan

from .guards import BREAKING_CHARACTER, GuardReport, apply_guards
from .planner import (
    FALLBACK,
    PARSED,
    STRUCTURED,
    CharacterSpeechPlanner,
    PlanResult,
    plan_to_messages,
)
from .memory import ConversationMemory, extract_facts, render_transcript
from .prompt import UNIVERSAL_RULES, build_messages, render_system_prompt, trim_history


def speech_plan_json_schema() -> dict:
    """The schema the planner constrains the LLM to (spec §12)."""
    return CharacterSpeechPlan.model_json_schema()


__all__ = [
    "BREAKING_CHARACTER",
    "CharacterSpeechPlanner",
    "FALLBACK",
    "GuardReport",
    "PARSED",
    "PlanResult",
    "STRUCTURED",
    "UNIVERSAL_RULES",
    "apply_guards",
    "build_messages",
    "plan_to_messages",
    "render_system_prompt",
    "speech_plan_json_schema",
    "ConversationMemory",
    "extract_facts",
    "render_transcript",
    "trim_history",
]
