"""``/api/chat`` routes over a fake ``LLMProvider`` — the transport, not the model."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from cvai_api.chat import build_chat_router  # noqa: E402
from cvai_core.errors import GenerationError, ProviderUnavailableError  # noqa: E402
from cvai_core.interfaces.llm import LLMProvider  # noqa: E402
from cvai_types import (  # noqa: E402
    LLMCapabilities,
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class FakeLLM(LLMProvider):
    provider = "fake"

    def __init__(self, *, fail_before: Exception | None = None,
                 fail_after: Exception | None = None) -> None:
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.seen: list[LLMMessage] = []

    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(provider="fake", model="fake-1")

    async def complete(self, messages, *, temperature=None, max_output_tokens=None):
        self.seen = messages
        if self.fail_before:
            raise self.fail_before
        return LLMResponse(content="你好。", model="fake-1", provider="fake",
                           usage=LLMUsage(completion_tokens=2))

    async def stream(self, messages, *, temperature=None,
                     max_output_tokens=None) -> AsyncIterator[LLMStreamChunk]:
        self.seen = messages
        if self.fail_before:
            raise self.fail_before
        yield LLMStreamChunk(delta="你好，")
        if self.fail_after:
            raise self.fail_after
        yield LLMStreamChunk(delta="漂泊者。")
        yield LLMStreamChunk(is_final=True, finish_reason="stop",
                             usage=LLMUsage(completion_tokens=4),
                             stats={"tokens_per_s": 200.0})


def client_for(llm: LLMProvider) -> TestClient:
    app = FastAPI()
    app.include_router(build_chat_router(lambda: llm))
    return TestClient(app)


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


BODY = {"messages": [
    {"role": "system", "content": "你正在扮演一个游戏角色。"},
    {"role": "user", "content": "今天在做什么？"},
    {"role": "assistant", "content": "在看雨。"},
    {"role": "user", "content": "然后呢？"},
]}


def test_chat_returns_the_whole_answer_and_passes_history_through():
    llm = FakeLLM()
    response = client_for(llm).post("/api/chat", json=BODY)
    assert response.status_code == 200
    assert response.json()["content"] == "你好。"
    assert [m.role.value for m in llm.seen] == ["system", "user", "assistant", "user"]
    assert llm.seen[0].content == "你正在扮演一个游戏角色。"


def test_stream_is_server_sent_events_in_order():
    response = client_for(FakeLLM()).post("/api/chat/stream", json=BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = events(response.text)
    assert [e for e, _ in parsed] == ["start", "delta", "delta", "done"]
    assert "".join(d["text"] for e, d in parsed if e == "delta") == "你好，漂泊者。"
    done = parsed[-1][1]
    assert done["stats"]["tokens_per_s"] == 200.0 and done["usage"]["completion_tokens"] == 4
    assert parsed[0][1] == {"provider": "fake", "model": "fake-1"}


def test_provider_unavailable_is_a_503_before_streaming_starts():
    llm = FakeLLM(fail_before=ProviderUnavailableError("Ollama is not running at x"))
    response = client_for(llm).post("/api/chat/stream", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "kind": "provider_unavailable", "error": "Ollama is not running at x"}
    assert client_for(llm).post("/api/chat", json=BODY).status_code == 503


def test_failure_mid_stream_arrives_as_an_error_event():
    llm = FakeLLM(fail_after=GenerationError("GPU ran out of memory"))
    parsed = events(client_for(llm).post("/api/chat/stream", json=BODY).text)
    assert [e for e, _ in parsed] == ["start", "delta", "error"]
    assert parsed[-1][1] == {"kind": "generation_failed", "error": "GPU ran out of memory"}


def test_generation_failure_on_plain_chat_is_a_502():
    llm = FakeLLM(fail_before=GenerationError("timed out"))
    response = client_for(llm).post("/api/chat", json=BODY)
    assert response.status_code == 502 and "timed out" in response.json()["detail"]["error"]


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "robot", "content": "hi"}]},
    {},
])
def test_malformed_requests_are_rejected(body):
    assert client_for(FakeLLM()).post("/api/chat/stream", json=body).status_code == 422


def test_health_reports_the_provider_check():
    assert client_for(FakeLLM()).get("/api/llm/health").json() == {"ok": True, "provider": "fake"}
