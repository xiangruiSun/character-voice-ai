"""``LLMProvider``.

Text and structured metadata only — never audio (spec §3). ``complete_structured`` is a
first-class method rather than a parameter because the Character Speech Planner depends
on it, and a provider that cannot constrain output to a schema needs a visibly different
code path (parse-and-repair) rather than a silently unreliable one.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from cvai_types import LLMCapabilities, LLMMessage, LLMResponse, LLMStreamChunk


class LLMProvider(abc.ABC):
    provider: str = "unknown"

    @abc.abstractmethod
    def capabilities(self) -> LLMCapabilities:
        ...

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Token deltas. The speech chunker consumes these, not the orchestrator."""

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Return JSON conforming to ``schema`` (the Character Speech Plan schema)."""
        raise NotImplementedError(
            f"{self.provider} does not support structured output; "
            "the planner needs a parse-and-repair fallback for this provider"
        )

    async def check(self) -> dict[str, Any]:
        """Readiness for a health endpoint: ``{"ok": bool, ...}``, never raises.

        Providers with a backend to probe (a local server, a model to download)
        override this so a setup problem is reported before the first chat fails.
        """
        return {"ok": True, "provider": self.provider}

    async def aclose(self) -> None:
        return None
