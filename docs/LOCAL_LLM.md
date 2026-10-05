# Local LLM: Qwen3 4B through Ollama

The chat model runs entirely on this machine. No API key, no per-token cost, nothing
leaves the PC.

```
Browser ──► FastAPI (/api/chat, /api/chat/stream) ──► LLMProvider ──► OllamaProvider
                                                                         │
                                               Ollama (127.0.0.1:11434) ◄┘ ──► qwen3:4b
```

The browser only talks to FastAPI. It never sees Ollama's address, so swapping the model
or the runtime (another Ollama model, vLLM, a remote server, an API) is a config change
in `configs/app.yaml`; the frontend does not change.

| Piece | Where |
|---|---|
| Provider interface | `services/core/cvai_core/interfaces/llm.py` (`LLMProvider`) |
| Ollama implementation | `providers/llm/cvai_llm_providers/ollama_llm.py` (`OllamaLLMProvider`) |
| Chat routes | `apps/api/cvai_api/chat.py` |
| Chat page | `apps/web/chat.html`, served at `/chat` |
| Configuration | `configs/app.yaml` → `providers.llm`, overridden by `.env` |
| Live checks | `scripts/check_local_llm.py` |

---

## Prerequisites

- Windows 10/11 (tested: Windows 11, RTX 4090 24 GB, driver 610.60)
- An NVIDIA GPU is strongly recommended; Ollama falls back to CPU without one (much slower)
- Python 3.10+ and this repository installed:

```powershell
cd C:\Users\gaming\character-voice-ai
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,runtime]"
```

No Node.js is needed: the chat page is a single HTML file served by FastAPI.

## 1. Install and start Ollama

```powershell
winget install --id Ollama.Ollama -e
```

The installer starts Ollama in the system tray and registers it to start with Windows.
It listens on `127.0.0.1:11434` only (not reachable from other machines). Check it:

```powershell
curl.exe http://127.0.0.1:11434/api/version
```

If it is not running, start it from the Start menu (**Ollama**) or run the server
directly:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve
```

Stop it with **Quit Ollama** from the tray icon. Do not force-kill `ollama.exe`: its
model runner (`llama-server.exe`) is left behind holding RAM and VRAM.

## 2. Download Qwen3 4B

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen3:4b
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

About 2.5 GB (4-bit Q4_K_M). If `ollama` is on your PATH (open a new terminal after
installing), the shorter `ollama pull qwen3:4b` works too.

## 3. Configure

The defaults already select local Qwen; `.env` in the repository root (git-ignored)
can override them:

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Which `providers.llm` instance answers (`CVAI_LLM` is accepted as an alias) |
| `LLM_MODEL` | `qwen3:4b` | Any model you have pulled into Ollama |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Use `127.0.0.1`, not `localhost`: on Windows `localhost` may try IPv6 first |
| `OLLAMA_NUM_CTX` | `8192` | Context window. Larger costs VRAM (32768 ≈ 7.5 GB vs 3.9 GB) |
| `OLLAMA_THINK` | `true` | See "Thinking" below; leave on for `qwen3:4b` |

Temperature, answer length and the thinking budget live in `configs/app.yaml` under
`providers.llm.instances.ollama`; requests can override temperature and length.

## 4. Start the backend

```powershell
cd C:\Users\gaming\character-voice-ai
.\.venv\Scripts\cvai-api.exe --port 8000
```

## 5. Open the website

<http://127.0.0.1:8000/chat>

The header should show `ollama / qwen3:4b ✓`. Type a message, press Enter; the answer
streams in. The system-prompt box is the character prompt (leave it empty for a plain
assistant). **停止** cancels generation; **清空** starts a new conversation.

---

## Testing

**Qwen directly** (bypasses the backend):

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run qwen3:4b "你好，请用两句话介绍你自己。"
```

**Through FastAPI**:

```powershell
# readiness: is Ollama up and the model downloaded?
curl.exe http://127.0.0.1:8000/api/llm/health

# whole answer as JSON (Windows PowerShell needs the explicit UTF-8 handling for Chinese)
$body = @{ messages = @(@{ role = "user"; content = "你好，请用两句话介绍你自己。" }) } | ConvertTo-Json -Depth 5
$resp = Invoke-WebRequest -UseBasicParsing -Method Post -Uri http://127.0.0.1:8000/api/chat -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
([Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray()) | ConvertFrom-Json).content
```

For streaming and the full set of scenarios, use the check script. It runs six
scenarios (Chinese chat, multi-turn memory,
character role-play, streaming, a long answer, and Ollama being unreachable) through
`/api/chat/stream` and prints timings:

```powershell
.\.venv\Scripts\python.exe scripts\check_local_llm.py
```

**Unit tests** (no Ollama needed; a fake Ollama is used):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ollama_provider.py tests\test_chat_api.py -q
```

**In the browser**: open <http://127.0.0.1:8000/chat>, send `你好！我叫小林，请记住我的名字。`, then
`我叫什么名字？` — the second answer should contain 小林.

## API

`POST /api/chat/stream` (Server-Sent Events) and `POST /api/chat` (JSON) take the same
body. The server keeps no chat state: send the whole conversation each time, with the
character prompt as the `system` message.

```json
{
  "messages": [
    {"role": "system", "content": "你正在扮演一个游戏角色……"},
    {"role": "user", "content": "今天在做什么？"}
  ],
  "temperature": 0.7,
  "max_tokens": 400
}
```

Stream events, in order: `start` `{provider, model}`, then `delta` `{text}` repeatedly,
then `done` `{finish_reason, usage, stats, first_token_ms, total_ms}` or `error`
`{kind, error}`. Closing the connection cancels generation in Ollama.

Setup problems (Ollama down, model missing) return **HTTP 503** before streaming starts;
failures while generating arrive as an `error` event (or **502** on `/api/chat`).

## Thinking

`qwen3:4b` always reasons before answering; that cannot be switched off for this build
(with `think: false` the reasoning leaks into the answer instead). With thinking on,
Ollama returns the reasoning separately and only the answer is streamed. The cost is
latency: the first visible text appears after the reasoning, typically 1.5–5 s and up to
~10 s for prompts that invite careful working. If the model spends its whole token
budget thinking, the request fails with a clear error rather than an empty reply; raise
`thinking_token_budget` in `configs/app.yaml` if that happens.

For faster first text (useful later for voice), a non-thinking variant answers
immediately: `ollama pull qwen3:4b-instruct-2507-q4_K_M`, then set
`LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M` and `OLLAMA_THINK=false`.

## Baseline performance (RTX 4090, i9-14900KF, 32 GB RAM)

| Measure | Result |
|---|---|
| Generation speed | 210–225 tokens/s (thinking + answer) |
| First visible text, warm | 1.3–4.5 s typical, up to ~9 s; all of it is thinking |
| Model load | ~18 s first time after install; ~2 s when files are cached |
| VRAM | 3.9 GB at 8k context (7.5 GB at Ollama's default 32k) |
| RAM | ~3 GB for the model runner, under 100 MB for the Ollama service |
| GPU / CPU while generating | ~85 % GPU (100 % of layers on GPU); ~25 % CPU |

Ollama unloads the model after 30 minutes idle (`keep_alive`); the next request reloads it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Page header shows `✗ Ollama is not running` | Start Ollama (Start menu → Ollama, or `ollama.exe serve`). The backend reconnects on its own; no restart needed |
| `Model 'qwen3:4b' is not downloaded` | `ollama pull qwen3:4b` |
| `GPU ran out of memory` | Close other GPU programs, or lower `OLLAMA_NUM_CTX` |
| `did not respond within 180s` | Check `ollama ps`; the first request after a restart loads the model |
| `used its whole token budget thinking` | Raise `thinking_token_budget`, or use the instruct model above |
| `ollama` not recognised in a terminal | Open a new terminal, or use the full path `$env:LOCALAPPDATA\Programs\Ollama\ollama.exe` |
| Port 11434 already in use | Ollama is already running (tray); you don't need `ollama serve` |
| Is it using the GPU? | `ollama ps` → `PROCESSOR` should read `100% GPU` |
| Page loads but the browser can't reach the API | Open the page from the backend (`http://127.0.0.1:8000/chat`), not from disk |
