"""Character Speech Planner — Milestone 8.

The component that turns "what she says" into "how she says it" (spec §12). It is not
implemented yet, on purpose: the planner's output schema
(:class:`cvai_types.speech_plan.CharacterSpeechPlan`) and the neutral control payload it
produces (:class:`cvai_types.style.StyleControls`) already exist and are already consumed
by the TTS adapters, so the planner can be added without changing anything downstream.

What Milestone 8 adds here:

* ``prompt.py`` — assemble the system prompt from a :class:`CharacterProfile`: personality,
  lore, speaking habits, forbidden behaviour, plus dialogue examples retrieved for the
  current turn.
* ``planner.py`` — call ``LLMProvider.complete_structured`` with the JSON schema of
  ``CharacterSpeechPlan`` and validate the result; fall back to parse-and-repair for
  providers that cannot constrain output.
* ``guards.py`` — reject plans that ask for a style the character's voice pack cannot
  perform, or that exceed ``llm.max_chars_per_reply``.

The schema is available today for anyone who wants to see the contract:

    from cvai_types import CharacterSpeechPlan
    CharacterSpeechPlan.model_json_schema()
"""

from __future__ import annotations

from cvai_types import CharacterSpeechPlan


def speech_plan_json_schema() -> dict:
    """The JSON schema the planner will constrain the LLM to (spec §12).

    Exposed now so the prompt work and the schema cannot drift apart later.
    """
    return CharacterSpeechPlan.model_json_schema()


__all__ = ["speech_plan_json_schema"]
