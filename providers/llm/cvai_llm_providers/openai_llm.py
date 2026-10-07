"""OpenAI LLM adapter (Milestones 8-9).

Text and structured metadata only — the built-in voice is explicitly not used as the
character voice (spec §3). ``complete_structured`` is what the Character Speech Planner
calls: it constrains the model to the Character Speech Plan schema so the planner gets a
delivery plan rather than prose it has to parse.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from cvai_core.errors import GenerationError, ProviderUnavailableError
from cvai_core.interfaces.llm import LLMProvider
from cvai_core.registry import LLM_PROVIDERS
from cvai_types import LLMCapabilities, LLMMessage, LLMResponse, LLMStreamChunk, LLMUsage


#: Model families that only accept the default temperature.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


@LLM_PROVIDERS.register("openai")
class OpenAILLMProvider(LLMProvider):
    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.8,
        max_output_tokens: int = 400,
        timeout_s: float = 60.0,
        max_context_tokens: int = 128000,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        #: For thinking models behind OpenAI-compatible APIs (e.g. Gemini 3, which
        #: cannot switch thinking off): "low" keeps replies quick.
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_context_tokens = max_context_tokens
        self._client: Any = None

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            provider=self.provider,
            model=self.model,
            supports_streaming=True,
            supports_structured_output=True,
            max_context_tokens=self.max_context_tokens,
        )

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415 - lazy
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError(
                "the openai package is not installed; pip install -e '.[runtime]'"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": self.timeout_s}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _sampling(
        self, temperature: float | None, max_output_tokens: int | None
    ) -> dict[str, Any]:
        """Sampling arguments the configured model accepts.

        Reasoning models (gpt-5, o-series) reject ``temperature`` and count hidden
        reasoning against the token budget, so they get no temperature and a budget
        large enough that the visible reply is not starved.
        """
        budget = max_output_tokens or self.max_output_tokens
        if self.model.startswith(_REASONING_PREFIXES):
            return {"max_completion_tokens": max(budget, 4000)}
        if self.reasoning_effort:
            return {
                "temperature": self.temperature if temperature is None else temperature,
                "max_completion_tokens": max(budget, 4000),
                "reasoning_effort": self.reasoning_effort,
            }
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "max_completion_tokens": budget,
        }

    async def test_connection(self) -> dict[str, Any]:
        """One attempt with a short timeout: a person is waiting on a "Testing…" button.

        The client's automatic retries with backoff suit a conversation, but here they
        turned a provider's 503 into a minute of silence before any answer.
        """
        started = time.perf_counter()
        try:
            client = self._client_or_raise().with_options(
                max_retries=0, timeout=min(self.timeout_s, 30.0))
            await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                **self._sampling(None, 16),
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return {"ok": False, "latency_ms": None, "error": str(exc)}
        return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": None}

    async def list_models(self) -> list[str]:
        client = self._client_or_raise()
        page = await client.models.list()
        return sorted(model.id for model in page.data)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    def _dump(messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._client_or_raise()
        started = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.model,
            messages=self._dump(messages),
            **self._sampling(temperature, max_output_tokens),
        )
        choice = response.choices[0]
        if not (choice.message.content or "").strip() and choice.finish_reason == "length":
            # A thinking model spent the whole budget reasoning; an empty "success"
            # would be a silent failure.
            raise GenerationError(
                f"{self.model!r} used its whole token budget and wrote no answer; "
                "raise max tokens for this connection"
            )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            finish_reason=choice.finish_reason,
            usage=LLMUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = self._client_or_raise()
        stream = await client.chat.completions.create(
            model=self.model,
            messages=self._dump(messages),
            **self._sampling(temperature, max_output_tokens),
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = getattr(choice.delta, "content", None) or ""
            finished = choice.finish_reason is not None
            if delta or finished:
                yield LLMStreamChunk(
                    delta=delta, is_final=finished, finish_reason=choice.finish_reason
                )

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        client = self._client_or_raise()
        response = await client.chat.completions.create(
            model=self.model,
            messages=self._dump(messages),
            **self._sampling(temperature, None),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "character_speech_plan",
                    "schema": schema,
                    "strict": False,
                },
            },
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
