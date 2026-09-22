"""A scripted LLM, so the whole conversation runs with no key and no network.

Every other part of this project can be exercised offline — the benchmark has a mock
engine, the preprocessing stages report their own absence, the DSP is real — and the LLM
was the one remaining thing that turned "try it" into "get an API key first". That is a
poor first five minutes for a project whose claim is that the machinery works before any
model does.

So this provider answers. It is **not** pretending to be intelligent, and the code goes
out of its way to make that obvious: the replies are drawn from the character's own
dialogue examples when the prompt contains them, and are otherwise fixed. Nothing here
is a language model, and nothing measured with it says anything about a character's
writing. What it *does* prove is that the pipeline behind it is wired correctly —
planner → guards → normalizer → chunker → reference retrieval → engine → playback — and
that is a real thing to be able to check on a laptop on a plane.

It supports structured output, so the planner's primary path (``complete_structured``) is
the one exercised rather than its fallback. A development provider that silently pushes
the system onto its degraded path teaches you that the degraded path is normal.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

from cvai_core.interfaces.llm import LLMProvider
from cvai_core.registry import LLM_PROVIDERS
from cvai_types import LLMCapabilities, LLMMessage, LLMResponse, LLMStreamChunk, Role

#: Used when the prompt carries no dialogue examples to borrow from. Deliberately bland:
#: a memorable scripted line gets mistaken for the model having said something.
_DEFAULT_REPLIES: tuple[str, ...] = (
    "嗯，我在。",
    "知道了。",
    "让我想想……好吧。",
    "随你。",
    "这个我不太清楚。",
)

#: Lines in the rendered system prompt look like "她：…" or "她（teasing）：…", so the
#: style each borrowed line was written in comes back with it.
_EXAMPLE_LINE = re.compile(r"^她(?:（([^）]*)）)?：(.+)$", re.MULTILINE)
_STYLE_IN_PROMPT = re.compile(r"从这些里选一个：([^\n]+)")


@LLM_PROVIDERS.register("mock")
class MockLLMProvider(LLMProvider):
    """Deterministic replies for development, tests and the offline demo."""

    provider = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-1",
        reply: str | None = None,
        use_examples: bool = True,
        chunk_size: int = 6,
    ) -> None:
        self.model = model
        #: Pin every reply to one string, for tests that assert on it.
        self.fixed_reply = reply
        self.use_examples = use_examples
        self.chunk_size = max(1, chunk_size)

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            provider=self.provider,
            model=self.model,
            supports_structured_output=True,
            supports_streaming=True,
        )

    # -- the reply --------------------------------------------------------------------

    def _pick(self, messages: list[LLMMessage]) -> tuple[str, str]:
        """The reply and the style to perform it in."""
        if self.fixed_reply is not None:
            return self.fixed_reply, self._default_style(messages)

        user = next(
            (m.content for m in reversed(messages) if m.role is Role.USER), ""
        )
        system = next((m.content for m in messages if m.role is Role.SYSTEM), "")

        pool: list[tuple[str, str]] = []
        if self.use_examples:
            # Her own lines, lifted back out of the prompt they were put into, each with
            # the style it was written in. It keeps the register plausible without
            # inventing anything, which is the most an offline stand-in can honestly do
            # — and carrying the style through means the audition actually varies
            # instead of performing everything neutral.
            pool = [
                (line.strip(), (style or "neutral").strip())
                for style, line in _EXAMPLE_LINE.findall(system)
            ]
        pool = pool or [(reply, "neutral") for reply in _DEFAULT_REPLIES]

        # Deterministic: the same turn always produces the same reply, so a difference
        # between two runs is never this provider's doing.
        index = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16) % len(pool)
        text, style = pool[index]
        return text, style if style in self._allowed(messages) else self._default_style(messages)

    def _reply(self, messages: list[LLMMessage]) -> str:
        return self._pick(messages)[0]

    def _allowed(self, messages: list[LLMMessage]) -> list[str]:
        system = next((m.content for m in messages if m.role is Role.SYSTEM), "")
        match = _STYLE_IN_PROMPT.search(system)
        if not match:
            return ["neutral"]
        return [s.strip() for s in match.group(1).split("、") if s.strip()] or ["neutral"]

    def _default_style(self, messages: list[LLMMessage]) -> str:
        return self._allowed(messages)[0]

    # -- interface ---------------------------------------------------------------------

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self._reply(messages), model=self.model, provider=self.provider
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        text = self._reply(messages)
        for start in range(0, len(text), self.chunk_size):
            yield LLMStreamChunk(delta=text[start : start + self.chunk_size])
        yield LLMStreamChunk(is_final=True, finish_reason="stop")

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        text, style = self._pick(messages)
        return {
            "text": text,
            "emotion": style,
            "emotion_intensity": 0.5,
            "speaking_rate": "normal",
            "direction_note": "mock provider: scripted reply, no model involved",
        }
