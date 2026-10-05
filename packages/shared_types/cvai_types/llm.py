"""LLM types.

Spec §3 is firm that the LLM layer returns text and structured performance metadata only
— never audio. That constraint is expressed here: an ``LLMResponse`` has no audio field
and never will.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from .base import CVAIModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(CVAIModel):
    role: Role
    content: str


class LLMUsage(CVAIModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class LLMResponse(CVAIModel):
    """A complete (non-streamed) response."""

    content: str
    model: str = "unknown"
    provider: str = "unknown"
    finish_reason: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float | None = Field(default=None, ge=0.0)


class LLMStreamChunk(CVAIModel):
    """One delta from a streaming response.

    The speech chunker consumes these, not the orchestrator directly (spec §14).
    """

    delta: str = ""
    is_final: bool = False
    finish_reason: str | None = None
    #: Only on the final chunk, and only where the provider reports it.
    usage: LLMUsage | None = None
    #: Provider timing on the final chunk, e.g. ``tokens_per_s``, ``load_s``.
    stats: dict[str, float] = Field(default_factory=dict)


class LLMCapabilities(CVAIModel):
    provider: str
    model: str
    supports_streaming: bool = True
    #: Whether the provider can be constrained to a JSON schema. The Character Speech
    #: Planner depends on this; without it the planner needs a parse-and-repair fallback.
    supports_structured_output: bool = False
    max_context_tokens: int = Field(default=8192, gt=0)
