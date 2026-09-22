# Applications

## `api/` — the conversation backend (Milestones 9-13)

FastAPI over the orchestrator. Layered so almost none of it needs a web server to test:

| Module | What it is | Tested |
|---|---|---|
| `cvai_api.sessions` | assembles an orchestrator from config + profile + voice pack + engine | ✅ |
| `cvai_api.events` | maps internal `TurnEvent`s onto browser protocol messages | ✅ |
| `cvai_api.app` | routes and the WebSocket loop — transport only | via the two above |

```bash
pip install -e '.[runtime]'
uvicorn cvai_api.app:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | is it up, which engine is configured |
| `GET /characters` | who can be talked to |
| `POST /sessions` | open a conversation |
| `POST /sessions/{id}/turn` | one turn, JSON in, JSON out — the fastest way to check the whole pipeline with `curl` |
| `GET /sessions/{id}/audio/{path}` | a rendered chunk |
| `WS /sessions/{id}/ws` | the live conversation |
| `DELETE /sessions/{id}` | close it |

```bash
SID=$(curl -s -XPOST localhost:8000/sessions -H 'Content-Type: application/json' -d '{}' | jq -r .session_id)
curl -s -XPOST localhost:8000/sessions/$SID/turn \
     -H 'Content-Type: application/json' -d '{"text":"在吗"}' | jq
```

Sessions are in-memory. Spec §3 rules out a database until the character voice pipeline
works, so a restart loses conversations — an intended trade, not an oversight.

---

## `web/` — the frontend

**`dev-client.html` works today.** One file, no build step, no dependencies: open it in a
browser, point it at the API, and talk to the character. Text in, captions and audio out,
with a working barge-in button (Esc).

**The Next.js app is still to come.** Spec §29 is explicit that the website is not where
to start, and the research question in Milestones 1-7 has not been answered yet — there
is no chosen voice engine for a polished UI to showcase. When it is built it needs:
microphone capture via AudioWorklet, PCM streaming up the same WebSocket, and client-side
VAD to trigger barge-in without a button.

Two rules the real frontend must keep, both demonstrated in the dev client:

1. **State is observed, never decided** (spec §15). Every indicator reflects a `state`
   message from the server. The page never sets pipeline state itself — otherwise two
   state machines exist and they drift.
2. **The user sees text and hears audio, nothing else** (spec §12). The performance
   metadata never crosses the wire, so there is nothing for a frontend to display or
   act on by accident.
