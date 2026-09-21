"""The Character Speech Planner (spec §12).

The LLM is a performance director, not just a writer: it returns the line *and* how it
should be delivered. The user hears only the text; everything else routes to the voice
stack.

Three ways of getting structured output, in order of preference, because provider
support varies and a conversation turn must not fail over formatting:

1. **Constrained JSON** via ``LLMProvider.complete_structured`` — the good path.
2. **Parse-and-repair** — ask for JSON in the prompt, then extract the first object
   from whatever comes back. Handles code fences and leading chatter.
3. **Plain text** — treat the whole reply as the line and infer delivery from the
   profile's defaults. The character still speaks; she just speaks in her default
   register, which is much better than an error.

Whichever path ran is recorded on the result, because a planner silently spending its
life on path 3 is a planner whose provider needs changing.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from cvai_core.interfaces.character import CharacterProvider
from cvai_core.interfaces.llm import LLMProvider
from cvai_core.logging_setup import get_logger
from cvai_types import (
    CharacterProfile,
    CharacterSpeechPlan,
    CVAIModel,
    LLMMessage,
    Role,
)
from pydantic import Field

from .guards import GuardReport, apply_guards
from .prompt import build_messages, trim_history

log = get_logger(__name__)

#: Finds the first balanced JSON object, tolerating code fences and preamble.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


#: Which path produced a plan. Plain strings rather than an enum: they are written into
#: run records and read by humans more often than by code.
STRUCTURED = "structured"
PARSED = "parsed"
FALLBACK = "fallback"


class PlanResult(CVAIModel):
    plan: CharacterSpeechPlan
    #: Which of the three paths produced it.
    source: str = STRUCTURED
    guards: GuardReport = Field(default_factory=GuardReport)
    raw_response: str = ""

    @property
    def degraded(self) -> bool:
        return self.source != STRUCTURED


class CharacterSpeechPlanner:
    """Turns a user utterance into a planned character response."""

    def __init__(
        self,
        llm: LLMProvider,
        characters: CharacterProvider,
        *,
        example_limit: int = 6,
        strip_narration: bool = True,
    ) -> None:
        self.llm = llm
        self.characters = characters
        self.example_limit = example_limit
        self.strip_narration = strip_narration

    # -- prompt --------------------------------------------------------------------

    def compose(
        self,
        profile: CharacterProfile,
        user_text: str,
        *,
        history: Sequence[LLMMessage] = (),
        style_hint: str | None = None,
    ) -> list[LLMMessage]:
        examples = self.characters.retrieve_examples(
            profile.character_id,
            user_text,
            limit=self.example_limit,
            style=style_hint,
        )
        return build_messages(
            profile,
            user_text,
            history=trim_history(history, profile.memory.max_turns),
            examples=examples,
            memory_facts=profile.memory.persistent_facts,
            style_names=profile.available_styles,
        )

    # -- planning ------------------------------------------------------------------

    async def plan(
        self,
        character_id: str,
        user_text: str,
        *,
        history: Sequence[LLMMessage] = (),
        style_hint: str | None = None,
    ) -> PlanResult:
        profile = self.characters.get(character_id)
        messages = self.compose(
            profile, user_text, history=history, style_hint=style_hint
        )

        raw = ""
        source = STRUCTURED
        plan: CharacterSpeechPlan | None = None

        capabilities = self.llm.capabilities()
        if capabilities.supports_structured_output:
            try:
                payload = await self.llm.complete_structured(
                    messages,
                    CharacterSpeechPlan.model_json_schema(),
                    temperature=profile.llm.temperature,
                )
                raw = json.dumps(payload, ensure_ascii=False)
                plan = self._coerce(payload, profile)
            except Exception as exc:  # noqa: BLE001 - fall through to the next path
                log.warning("structured output failed, falling back to parsing: %s", exc)

        if plan is None:
            response = await self.llm.complete(
                messages,
                temperature=profile.llm.temperature,
                max_output_tokens=profile.llm.max_output_tokens,
            )
            raw = response.content
            plan = self._parse(raw, profile)
            source = PARSED if plan is not None else FALLBACK
            if plan is None:
                plan = self._fallback(raw, profile)

        guarded, report = apply_guards(
            plan, profile, strip_narration=self.strip_narration
        )
        return PlanResult(
            plan=guarded, source=source, guards=report, raw_response=raw
        )

    # -- parsing -------------------------------------------------------------------

    def _parse(self, raw: str, profile: CharacterProfile) -> CharacterSpeechPlan | None:
        text = _CODE_FENCE.sub("", raw or "").strip()
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return self._coerce(payload, profile)

    def _coerce(
        self, payload: dict, profile: CharacterProfile
    ) -> CharacterSpeechPlan | None:
        """Validate a payload, dropping unknown keys rather than failing on them.

        Models add fields. Rejecting an otherwise-good plan because it also contained
        ``"confidence": 0.9`` would trade a usable turn for a schema purist's win.
        """
        known = set(CharacterSpeechPlan.model_fields)
        cleaned = {k: v for k, v in payload.items() if k in known}
        if not str(cleaned.get("text", "")).strip():
            return None
        cleaned.setdefault("language", "zh-CN")
        try:
            return CharacterSpeechPlan.model_validate(cleaned)
        except Exception as exc:  # noqa: BLE001
            log.warning("plan payload failed validation: %s", exc)
            # One retry with only the fields we are sure about: a bad enum value for
            # speaking_rate should not cost us the line itself.
            try:
                return CharacterSpeechPlan(
                    text=str(cleaned["text"]),
                    emotion=str(cleaned.get("emotion", profile.voice.default_reference_style)),
                )
            except Exception:  # noqa: BLE001
                return None

    def _fallback(self, raw: str, profile: CharacterProfile) -> CharacterSpeechPlan:
        """Last resort: speak the reply in the character's default register."""
        text = _CODE_FENCE.sub("", raw or "").strip() or "……"
        log.warning(
            "planner fell back to plain text for %s; the provider is not returning "
            "usable JSON and every line will be delivered in the default style",
            profile.character_id,
        )
        return CharacterSpeechPlan(
            text=text,
            emotion=profile.voice.default_reference_style,
            direction_note="fallback: no structured plan was returned",
        )


def plan_to_messages(plan: CharacterSpeechPlan) -> list[LLMMessage]:
    """Append a plan to conversation history as the assistant's turn.

    Only the text goes into history. The delivery metadata is an instruction to the
    voice stack, not something the character said, and feeding it back would teach the
    model to talk about its own stage directions.
    """
    return [LLMMessage(role=Role.ASSISTANT, content=plan.text)]
