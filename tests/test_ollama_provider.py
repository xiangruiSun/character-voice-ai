"""Ollama adapter against a fake Ollama (``httpx.MockTransport``) — no server needed.

What matters most: only the answer is streamed (a thinking model's reasoning stays out),
the request carries the options the model needs, and every way Ollama can be missing or
broken surfaces as an error that says what to do.
"""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from cvai_core.errors import GenerationError, ProviderUnavailableError  # noqa: E402
from cvai_llm_providers.ollama_llm import OllamaLLMProvider  # noqa: E402
from cvai_types import LLMMessage, Role  # noqa: E402

MESSAGES = [
    LLMMessage(role=Role.SYSTEM, content="你是卡提希娅。"),
    LLMMessage(role=Role.USER, content="你好"),
]


def ndjson(*lines: dict) -> bytes:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()


STREAM = ndjson(
    {"message": {"role": "assistant", "content": "", "thinking": "用户在打招呼……"}},
    {"message": {"role": "assistant", "content": "你好，"}},
    {"message": {"role": "assistant", "content": "漂泊者。"}},
    {
        "message": {"role": "assistant", "content": ""},
        "done": True, "done_reason": "stop",
        "prompt_eval_count": 20, "eval_count": 50,
        "eval_duration": 250_000_000, "load_duration": 1_000_000_000,
    },
)


def provider(handler, **kwargs) -> OllamaLLMProvider:
    return OllamaLLMProvider(transport=httpx.MockTransport(handler), **kwargs)


def run(coro):
    return asyncio.run(coro)


async def collect(llm: OllamaLLMProvider):
    try:
        return [chunk async for chunk in llm.stream(MESSAGES)]
    finally:
        await llm.aclose()


def test_streams_answer_deltas_and_never_the_thinking():
    chunks = run(collect(provider(lambda request: httpx.Response(200, content=STREAM))))
    text = "".join(c.delta for c in chunks)
    assert text == "你好，漂泊者。"
    assert "用户在打招呼" not in text
    assert [c.delta for c in chunks[:-1]] == ["你好，", "漂泊者。"]  # incremental


def test_final_chunk_carries_usage_and_speed():
    final = run(collect(provider(lambda r: httpx.Response(200, content=STREAM))))[-1]
    assert final.is_final and final.finish_reason == "stop"
    assert final.usage.prompt_tokens == 20 and final.usage.completion_tokens == 50
    assert final.stats["tokens_per_s"] == 200.0
    assert final.stats["load_s"] == 1.0


def test_request_carries_history_model_and_options():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=STREAM)

    llm = provider(handler, model="qwen3:4b", think="true", num_ctx="4096",
                   max_output_tokens=300, thinking_token_budget=1000)
    run(collect(llm))
    assert seen["model"] == "qwen3:4b" and seen["stream"] is True
    assert seen["messages"] == [
        {"role": "system", "content": "你是卡提希娅。"},
        {"role": "user", "content": "你好"},
    ]
    assert seen["think"] is True  # strings from env vars are coerced
    # Thinking tokens count against num_predict, so the answer budget is topped up.
    assert seen["options"]["num_predict"] == 1300
    assert seen["options"]["num_ctx"] == 4096


def test_per_call_temperature_and_length_override_defaults():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=STREAM)

    llm = provider(handler, think=False, temperature=0.7)

    async def go():
        try:
            return [c async for c in llm.stream(MESSAGES, temperature=0.1, max_output_tokens=50)]
        finally:
            await llm.aclose()

    run(go())
    assert seen["options"]["temperature"] == 0.1
    assert seen["options"]["num_predict"] == 50  # no thinking budget when think is off


def test_complete_joins_the_stream():
    llm = provider(lambda r: httpx.Response(200, content=STREAM))
    response = run(llm.complete(MESSAGES))
    assert response.content == "你好，漂泊者。"
    assert response.usage.completion_tokens == 50
    assert response.provider == "ollama"


def test_structured_output_sends_the_schema_and_parses_json():
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        body = ndjson({"message": {"content": '{"text": "你好"}'}, "done": True})
        return httpx.Response(200, content=body)

    assert run(provider(handler).complete_structured(MESSAGES, schema)) == {"text": "你好"}
    assert seen["format"] == schema and seen["stream"] is False


def test_ollama_not_running_says_how_to_start_it():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnavailableError, match="Ollama is not running"):
        run(collect(provider(handler)))


def test_missing_model_says_which_pull_to_run():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'qwen3:4b' not found"})

    with pytest.raises(ProviderUnavailableError, match="ollama pull qwen3:4b"):
        run(collect(provider(handler)))


def test_out_of_memory_is_reported_as_such():
    def handler(request):
        return httpx.Response(500, json={"error": "CUDA error: out of memory"})

    with pytest.raises(GenerationError, match="ran out of memory"):
        run(collect(provider(handler)))


def test_an_error_inside_the_stream_is_not_swallowed():
    body = ndjson({"message": {"content": "你"}}, {"error": "model failed to load"})
    with pytest.raises(GenerationError, match="model failed to load"):
        run(collect(provider(lambda r: httpx.Response(200, content=body))))


def test_thinking_through_the_whole_budget_is_an_error_not_an_empty_reply():
    body = ndjson(
        {"message": {"content": "", "thinking": "嗯……" * 50}},
        {"message": {"content": ""}, "done": True, "done_reason": "length", "eval_count": 4496},
    )
    with pytest.raises(GenerationError, match="whole token budget"):
        run(collect(provider(lambda r: httpx.Response(200, content=body))))


def test_timeout_is_explained():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(GenerationError, match="did not respond"):
        run(collect(provider(handler)))


def test_check_reports_a_missing_model_without_raising():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    result = run(provider(handler).check())
    assert result["ok"] is False and "ollama pull qwen3:4b" in result["error"]


def test_check_passes_when_the_model_is_present():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})

    assert run(provider(handler).check())["ok"] is True


def test_it_is_the_default_provider(app_config):
    from cvai_core.config import load_config
    from cvai_core.registry import build_llm_provider

    llm = build_llm_provider(load_config(env={}))
    assert isinstance(llm, OllamaLLMProvider)
    assert llm.model == "qwen3:4b"


def test_env_vars_select_model_and_url():
    from cvai_core.config import load_config
    from cvai_core.registry import build_llm_provider

    llm = build_llm_provider(load_config(env={
        "LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3:8b",
        "OLLAMA_BASE_URL": "http://10.0.0.5:11434",
    }))
    assert llm.model == "qwen3:8b" and llm.base_url == "http://10.0.0.5:11434"


def test_old_cvai_llm_spelling_still_selects_a_provider():
    from cvai_core.config import load_config
    from cvai_core.registry import build_llm_provider

    assert build_llm_provider(load_config(env={"CVAI_LLM": "mock"})).provider == "mock"
