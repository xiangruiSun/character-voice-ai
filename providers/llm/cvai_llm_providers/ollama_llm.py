"""Ollama LLM adapter — local models (default ``qwen3:4b``) with no key and no cost.

Speaks Ollama's native ``/api/chat`` rather than its OpenAI-compatible endpoint, because
only the native API separates a thinking model's reasoning from its answer (``think``)
and accepts a full JSON schema in ``format`` for the Character Speech Planner.

Streaming is newline-delimited JSON. Cancellation is the HTTP connection: when the
consumer stops iterating, the ``async with`` closes the response and Ollama stops
generating for that request.
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


def _as_bool(value: bool | str) -> bool:
    # Config values interpolated from env vars arrive as strings.
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@LLM_PROVIDERS.register("ollama")
class OllamaLLMProvider(LLMProvider):
    provider = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        max_output_tokens: int = 400,
        think: bool | str = True,
        thinking_token_budget: int = 4096,
        num_ctx: int | str = 8192,
        keep_alive: str = "30m",
        connect_timeout_s: float = 5.0,
        #: Time allowed between streamed lines. Generous because the first line waits
        #: for the model to load (~20 s cold) and for all of its thinking.
        read_timeout_s: float = 180.0,
        transport: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.think = _as_bool(think)
        self.thinking_token_budget = int(thinking_token_budget)
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self._transport = transport  # tests inject httpx.MockTransport
        self._client: Any = None

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            provider=self.provider,
            model=self.model,
            supports_streaming=True,
            supports_structured_output=True,
            max_context_tokens=self.num_ctx,
        )

    # -- transport ------------------------------------------------------------------

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415 - lazy, like the other adapters
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderUnavailableError(
                    "the Ollama adapter needs httpx: pip install -e '.[runtime]'"
                ) from exc
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.read_timeout_s, connect=self.connect_timeout_s
                ),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _payload(
        self,
        messages: list[LLMMessage],
        *,
        stream: bool,
        temperature: float | None,
        max_output_tokens: int | None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        answer_budget = max_output_tokens or self.max_output_tokens
        # Ollama counts thinking tokens against num_predict; without the extra budget a
        # thinking model spends the whole allowance reasoning and returns no answer.
        num_predict = answer_budget + (self.thinking_token_budget if self.think else 0)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": stream,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        if schema is not None:
            payload["format"] = schema
        return payload

    def _translate(self, exc: Exception) -> Exception:
        """Turn transport failures into errors that say what to do next."""
        import httpx  # noqa: PLC0415

        if isinstance(exc, httpx.ConnectError):
            return ProviderUnavailableError(
                f"Ollama is not running at {self.base_url}. Start the Ollama app (or run "
                "`ollama serve`). Not installed? `winget install Ollama.Ollama`."
            )
        if isinstance(exc, httpx.TimeoutException):
            return GenerationError(
                f"Ollama did not respond within {self.read_timeout_s:.0f}s for model "
                f"{self.model!r}. The first request loads the model (~20 s); if it keeps "
                "timing out, check `ollama ps` and GPU memory."
            )
        return GenerationError(f"Ollama request failed: {exc}")

    def _http_error(self, status: int, body: str) -> Exception:
        try:
            message = json.loads(body).get("error", body)
        except (ValueError, AttributeError):
            message = body
        message = str(message).strip() or f"HTTP {status}"
        lowered = message.lower()
        if status == 404 and "not found" in lowered:
            return ProviderUnavailableError(
                f"Model {self.model!r} is not downloaded. Run: ollama pull {self.model}"
            )
        if "out of memory" in lowered or "cudamalloc" in lowered or "oom" in lowered:
            return GenerationError(
                f"The GPU ran out of memory running {self.model!r}. Close other GPU "
                "programs, lower OLLAMA_NUM_CTX, or choose a smaller model. "
                f"(Ollama: {message})"
            )
        return GenerationError(f"Ollama could not run {self.model!r}: {message}")

    async def _lines(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """POST /api/chat and yield each decoded JSON line (one, when not streaming)."""
        import httpx  # noqa: PLC0415

        client = self._http()
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise self._http_error(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        # Errors after the 200 header (e.g. OOM while loading) arrive
                        # in-band on the stream.
                        raise self._http_error(500, line)
                    yield data
        except httpx.HTTPError as exc:
            raise self._translate(exc) from exc

    @staticmethod
    def _final(data: dict[str, Any]) -> tuple[LLMUsage, dict[str, float]]:
        usage = LLMUsage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
        stats: dict[str, float] = {}
        if data.get("eval_duration"):
            stats["tokens_per_s"] = round(
                usage.completion_tokens / (data["eval_duration"] / 1e9), 1
            )
        for key in ("load_duration", "prompt_eval_duration", "total_duration"):
            if data.get(key):
                stats[key.replace("_duration", "_s")] = round(data[key] / 1e9, 3)
        return usage, stats

    # -- interface ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        payload = self._payload(
            messages, stream=True, temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        answered = False
        async for data in self._lines(payload):
            # Only the answer is streamed; `message.thinking` is reasoning the user
            # never sees.
            delta = (data.get("message") or {}).get("content") or ""
            answered = answered or bool(delta.strip())
            if data.get("done"):
                if not answered and data.get("done_reason") == "length":
                    # A thinking model can spend the entire num_predict reasoning;
                    # an empty "successful" reply would be a silent failure.
                    raise GenerationError(
                        f"{self.model!r} used its whole token budget "
                        f"({data.get('eval_count', '?')} tokens) thinking and wrote no "
                        "answer. Raise thinking_token_budget in configs/app.yaml, or use "
                        "a non-thinking model (e.g. LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M)."
                    )
                usage, stats = self._final(data)
                yield LLMStreamChunk(
                    delta=delta, is_final=True,
                    finish_reason=data.get("done_reason") or "stop",
                    usage=usage, stats=stats,
                )
                return
            if delta:
                yield LLMStreamChunk(delta=delta)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        parts: list[str] = []
        final: LLMStreamChunk | None = None
        async for chunk in self.stream(
            messages, temperature=temperature, max_output_tokens=max_output_tokens
        ):
            parts.append(chunk.delta)
            if chunk.is_final:
                final = chunk
        return LLMResponse(
            content="".join(parts).strip(),
            model=self.model,
            provider=self.provider,
            finish_reason=final.finish_reason if final else None,
            usage=final.usage if final and final.usage else LLMUsage(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload = self._payload(
            messages, stream=False, temperature=temperature,
            max_output_tokens=None, schema=schema,
        )
        content = ""
        async for data in self._lines(payload):
            content += (data.get("message") or {}).get("content") or ""
        try:
            return json.loads(content or "{}")
        except ValueError as exc:
            raise GenerationError(
                f"{self.model!r} returned invalid JSON for a structured request: "
                f"{content[:200]!r}"
            ) from exc

    async def list_models(self) -> list[str]:
        response = await self._http().get("/api/tags")
        response.raise_for_status()
        return sorted(m.get("name", "") for m in response.json().get("models", []) if m.get("name"))

    async def test_connection(self) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self.check()
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1) if result["ok"] else None
        result.setdefault("error", None)
        return result

    async def check(self) -> dict[str, Any]:
        """Is Ollama reachable, and is the configured model downloaded?"""
        import httpx  # noqa: PLC0415

        try:
            response = await self._http().get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            error = self._translate(exc)
            return {"ok": False, "provider": self.provider, "model": self.model,
                    "error": str(error)}
        names = {m.get("name") for m in response.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            return {"ok": False, "provider": self.provider, "model": self.model,
                    "error": f"Model {self.model!r} is not downloaded. "
                             f"Run: ollama pull {self.model}"}
        return {"ok": True, "provider": self.provider, "model": self.model}
