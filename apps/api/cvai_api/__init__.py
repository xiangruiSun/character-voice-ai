"""HTTP and WebSocket API for the conversation loop (Milestones 9-13).

Layered so that almost none of it needs a web server to test:

* :mod:`cvai_api.sessions` — assembles an orchestrator from config, character profile,
  voice pack and engine, and reports what is missing when it cannot.
* :mod:`cvai_api.events` — maps internal ``TurnEvent``s onto the browser protocol,
  deliberately dropping the performance metadata the user must not see (spec §12).
* :mod:`cvai_api.app` — FastAPI routes and the WebSocket loop. Transport only.

Run it with ``uvicorn cvai_api.app:app`` after ``pip install -e '.[runtime]'``.
"""

from __future__ import annotations

from .events import audio_events, stream_to_messages, to_server_messages
from .sessions import Session, SessionManager

__all__ = [
    "Session",
    "SessionManager",
    "audio_events",
    "stream_to_messages",
    "to_server_messages",
]
