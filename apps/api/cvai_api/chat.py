"""Plain chat over the configured ``LLMProvider`` — no voice, no character pipeline.

    POST /api/chat          {"messages": [...]} → {"content": ...}
    POST /api/chat/stream   same body → Server-Sent Events: start, delta…, done | error
    GET  /api/llm/health    is the provider ready (Ollama running, model downloaded)?

The browser only ever talks to these routes. Which model answers — local Qwen through
Ollama today, something else later — is ``providers.llm`` in configs/app.yaml, so
swapping it never touches the frontend.

Deliberately no ``from __future__ import annotations``: FastAPI reads the parameter
annotations below at runtime (see the note in app.py).
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Optional

from cvai_core.errors import CVAIError, GenerationError, ProviderUnavailableError
from cvai_core.interfaces.llm import LLMProvider
from cvai_types import LLMMessage
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger("cvai-api.chat")


class ChatRequest(BaseModel):
    """A whole conversation, oldest first. A character prompt is the system message;
    history is the earlier user/assistant turns. The server keeps no chat state."""

    messages: list[LLMMessage] = Field(min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


def _status(exc: CVAIError) -> int:
    # 503: fix your setup (Ollama down, model missing). 502: the backend failed a request.
    return 503 if isinstance(exc, ProviderUnavailableError) else 502


def _kind(exc: CVAIError) -> str:
    if isinstance(exc, ProviderUnavailableError):
        return "provider_unavailable"
    if isinstance(exc, GenerationError):
        return "generation_failed"
    return "error"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_chat_router(get_llm: Callable[[], LLMProvider]) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/llm/health")
    async def llm_health() -> dict[str, Any]:
        try:
            llm = get_llm()
        except CVAIError as exc:
            return {"ok": False, "error": str(exc)}
        return await llm.check()

    @router.post("/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        llm = get_llm()
        try:
            response = await llm.complete(
                request.messages,
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )
        except CVAIError as exc:
            raise HTTPException(
                status_code=_status(exc), detail={"kind": _kind(exc), "error": str(exc)}
            ) from exc
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "finish_reason": response.finish_reason,
            "usage": response.usage.model_dump(),
            "latency_ms": round(response.latency_ms or 0.0, 1),
        }

    @router.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        llm = get_llm()
        started = time.perf_counter()
        chunks = llm.stream(
            request.messages,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        ).__aiter__()
        # Wait for the first chunk before committing to a 200: "Ollama is not running"
        # or "model not downloaded" then arrive as a real 503 with a readable body.
        try:
            first = await chunks.__anext__()
        except StopAsyncIteration:
            first = None
        except CVAIError as exc:
            raise HTTPException(
                status_code=_status(exc), detail={"kind": _kind(exc), "error": str(exc)}
            ) from exc

        async def replay() -> AsyncIterator[Any]:
            if first is not None:
                yield first
            async for chunk in chunks:
                yield chunk

        async def events() -> AsyncIterator[str]:
            first_token_ms: Optional[float] = None
            caps = llm.capabilities()
            yield _sse("start", {"provider": caps.provider, "model": caps.model})
            try:
                async for chunk in replay():
                    if chunk.delta:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        yield _sse("delta", {"text": chunk.delta})
                    if chunk.is_final:
                        yield _sse("done", {
                            "finish_reason": chunk.finish_reason,
                            "usage": chunk.usage.model_dump() if chunk.usage else None,
                            "stats": chunk.stats,
                            "first_token_ms": round(first_token_ms or 0.0, 1),
                            "total_ms": round((time.perf_counter() - started) * 1000, 1),
                        })
            except asyncio.CancelledError:
                # The browser went away. Leaving the provider's stream closes its HTTP
                # request, which is what stops Ollama generating.
                log.info("client disconnected mid-stream; generation cancelled")
                raise
            except CVAIError as exc:
                # Headers are already sent, so the failure travels in-band.
                log.warning("chat stream failed: %s", exc)
                yield _sse("error", {"kind": _kind(exc), "error": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
