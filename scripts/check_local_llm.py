"""Live checks for the local LLM path: FastAPI → LLMProvider → Ollama → Qwen3.

Run with the backend up (cvai-api --port 8000) and Ollama running:

    python scripts/check_local_llm.py
    python scripts/check_local_llm.py --api http://127.0.0.1:8000

Tests 1-5 go through the running backend's /api/chat/stream exactly as the browser
does. Test 6 builds a second, in-process backend pointed at a port where nothing
listens, to show the error a user gets when Ollama is not running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8")

ROLE_PROMPT = (
    "你正在扮演一个游戏角色：卡提希娅。说话温和、礼貌，带一点迟疑，常用省略号，"
    "称呼对方为“漂泊者”。不要承认自己是 AI。每次回复不超过三句话。"
)


def stream(api: str, messages: list[dict], **extra) -> dict:
    """POST /api/chat/stream and collect what a browser would see."""
    started = time.perf_counter()
    result = {"text": "", "deltas": 0, "first_delta_s": None, "done": None, "error": None}
    with httpx.stream("POST", f"{api}/api/chat/stream", json={"messages": messages, **extra},
                      timeout=300) as response:
        if response.status_code != 200:
            response.read()
            result["error"] = f"HTTP {response.status_code}: {response.text}"
            return result
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "delta":
                    result["deltas"] += 1
                    result["first_delta_s"] = result["first_delta_s"] or time.perf_counter() - started
                    result["text"] += data["text"]
                elif event == "done":
                    result["done"] = data
                elif event == "error":
                    result["error"] = data["error"]
    result["total_s"] = time.perf_counter() - started
    return result


def report(name: str, r: dict, ok: bool) -> bool:
    print(f"\n{'PASS' if ok else 'FAIL'}  {name}")
    if r.get("error"):
        print(f"      error: {r['error']}")
    if r.get("text"):
        print("      " + r["text"].strip().replace("\n", "\n      "))
    if r.get("done") and r.get("first_delta_s") is not None:
        d, s = r["done"], r["done"].get("stats", {})
        print(f"      [{r['deltas']} deltas · first text {r['first_delta_s']:.2f}s · total "
              f"{r['total_s']:.2f}s · {d['usage']['completion_tokens']} tokens incl. thinking"
              f" · {s.get('tokens_per_s', '?')} tok/s]")
    return ok


def has_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    health = httpx.get(f"{api}/api/llm/health", timeout=10).json()
    print("health:", health)
    if not health.get("ok"):
        print("Backend reports the LLM is not ready; fix that first.")
        return 1

    results = []

    r = stream(api, [{"role": "user", "content": "你好，请用两句话介绍你自己。"}])
    results.append(report("1 normal Chinese conversation", r,
                          not r["error"] and has_chinese(r["text"])))

    history = [{"role": "user", "content": "我叫小林，我最喜欢的水果是芒果。记住了吗？"}]
    first = stream(api, history)
    history += [{"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "我叫什么名字？我最喜欢什么水果？"}]
    r = stream(api, history)
    results.append(report("2 multi-turn (recalls name and fruit from turn 1)", r,
                          "小林" in r["text"] and "芒果" in r["text"]))

    r = stream(api, [{"role": "system", "content": ROLE_PROMPT},
                     {"role": "user", "content": "今天在做什么？"}])
    results.append(report("3 system prompt / character role-play", r,
                          not r["error"] and "漂泊者" in r["text"]))

    r = stream(api, [{"role": "user", "content": "从一数到二十，用中文数字，每个数字之间用逗号分隔。"}])
    results.append(report("4 streaming (answer arrives in many increments)", r,
                          not r["error"] and r["deltas"] >= 10
                          and r["first_delta_s"] is not None
                          and r["first_delta_s"] < r["total_s"] - 0.1))

    r = stream(api, [{"role": "user", "content": "请写一段大约三百字的短文，介绍中国的春节习俗。"}],
               max_tokens=1200)
    results.append(report("5 longer answer", r,
                          not r["error"] and len(r["text"]) >= 200))

    # 6: a backend whose Ollama is unreachable — same app factory, dead port.
    import os

    from fastapi.testclient import TestClient

    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11999"
    os.environ["LLM_PROVIDER"] = "ollama"
    from cvai_api.app import create_app  # noqa: PLC0415

    with TestClient(create_app()) as client:
        response = client.post("/api/chat/stream", json={
            "messages": [{"role": "user", "content": "你好"}]})
        body = response.json()
    r = {"error": f"HTTP {response.status_code}: {body['detail']['error']}"}
    results.append(report("6 Ollama not running → readable error", r,
                          response.status_code == 503
                          and "not running" in body["detail"]["error"]))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
