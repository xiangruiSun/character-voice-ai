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
| `WS /sessions/{id}/ws` | the live conversation, both directions of audio |
| `DELETE /sessions/{id}` | close it |

The WebSocket carries audio each way. **Down:** rendered WAV chunks, each bracketed by an
`audio_begin`/`audio_end` pair so the client knows the format before the bytes arrive.
**Up:** raw little-endian 16-bit PCM at 16 kHz from the microphone, sent between
`user_audio_begin` and `user_audio_end`. Endpointing and barge-in are decided server-side
(spec §15), and a `transcript` message says what was heard before the character answers.

Speech-to-text is built on the first microphone frame, not when the session opens: a
typed conversation should not load an ASR model, and a missing `providers.stt` block
should only be an error for the user who actually speaks.

```bash
SID=$(curl -s -XPOST localhost:8000/sessions -H 'Content-Type: application/json' -d '{}' | jq -r .session_id)
curl -s -XPOST localhost:8000/sessions/$SID/turn \
     -H 'Content-Type: application/json' -d '{"text":"在吗"}' | jq
```

Sessions are in-memory. Spec §3 rules out a database until the character voice pipeline
works, so a restart loses conversations — an intended trade, not an oversight.

### `cvai-talk` — the same conversation from a terminal

```bash
cvai-talk denia_cn                  # type at her, hear her answer
cvai-talk denia_cn --say "在吗"      # one line and exit
cvai-talk denia_cn --audition       # one line per style, as a page to listen through
```

Same orchestrator, different transport. The browser client answers "does the whole thing
work end to end?"; this answers "does she sound right?", which is the question asked
twenty times a day while a voice pack is being built, often on a machine with no browser.
`--verbose` shows the performance plan, the chunking and which reference clip conditioned
each chunk — spec §12 keeps that away from *users*, but the person building the voice is
not a user.

**`--audition` is not a benchmark.** It puts one line per declared style side by side,
which makes style collapse (spec §27) obvious in thirty seconds, but nothing in it is
blind and there are no real recordings to compare against. It tells you a voice pack is
wrong; only `cvai-bench` can suggest one is right.

---

## `web/` — the frontend

**The app (`index.html` + `assets/app.css` + `assets/app.js`) is served at `/`.** No build
step and no dependencies. It keeps three kinds of state apart and renders each on its own:

| | States | Shown in |
|---|---|---|
| Connection | connecting · connected · degraded (LLM down) · disconnected | top bar pill |
| Microphone | off · pending (permission) · on — the media track really stops when off | status chip, mic button |
| Conversation (`VoiceState`) | idle · requesting_mic_permission · listening · speech_detected · transcribing · thinking · generating_speech · speaking · error | status bar, character header |

Voice turns are one utterance each: press the mic, speak, press 完成 (or pause — the
server's silence detector ends it). The microphone closes as soon as the utterance ends,
the transcript appears as your message, and the status bar walks through 正在识别语音 →
正在思考 → 正在生成语音 → 正在说话 → 准备就绪. Errors appear next to the composer with a
retry. Per-turn timings are under Settings → 开发者信息. `window.__voice.log` records
every state transition, which the browser tests read.

**`dev-client.html` is at `/dev`**: the original pipeline client, which shows raw server
states and keeps the microphone open continuously for barge-in.

Protocol additions the app relies on (server → client):

| Message | Meaning |
|---|---|
| `{"type": "vad", "event": "speech_start"}` | the server's detector heard speech |
| `{"type": "state", "state": "transcribing"}` | an utterance is complete and being transcribed |
| `{"type": "error", "code": "no_speech"}` | nothing recognisable was said; no turn runs |

and client → server, `user_audio_begin` accepts `"mode": "turn"`: the server stops
listening after one utterance instead of staying open (the default, `continuous`).

Microphone capture runs in an AudioWorklet (on the audio thread, so layout on the main
thread cannot swallow the first syllable of a sentence), resamples to 16 kHz, and streams
raw PCM up the socket. `echoCancellation` is on: without it, open speakers make the
character hear herself and barge in on her own voice.

**The Next.js app is still to come.** Spec §29 is explicit that the website is not where
to start, and the research question in Milestones 1-7 has not been answered yet — there
is no chosen voice engine for a polished UI to showcase.

Three rules the real frontend must keep, all demonstrated in the dev client:

1. **State is observed, never decided** (spec §15). Every indicator reflects a `state`
   message from the server. The page never sets pipeline state itself — otherwise two
   state machines exist and they drift.
2. **The user sees text and hears audio, nothing else** (spec §12). The performance
   metadata never crosses the wire, so there is nothing for a frontend to display or
   act on by accident.
3. **No voice activity detection in the browser.** The page streams audio and does not
   decide what it means. A client-side VAD would be a second endpointing state machine
   racing the server's, and the two would disagree exactly when it matters — mid-barge-in.
