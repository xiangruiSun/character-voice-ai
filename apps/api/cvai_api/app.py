"""FastAPI application (Milestones 9-13).

Thin by design — session assembly lives in :mod:`cvai_api.sessions` and event mapping in
:mod:`cvai_api.events`, both testable without a web server. What is left here is
transport: routes, the WebSocket loop, and serving rendered audio.

    uvicorn cvai_api.app:app --reload --port 8000

Endpoints:

    GET  /health                      is the API up, and which engine is configured
    GET  /characters                  who can be talked to
    POST /sessions                    open a conversation
    DELETE /sessions/{id}             close it
    POST /sessions/{id}/turn          one turn, JSON in, events out (no streaming)
    GET  /sessions/{id}/audio/{path}  a rendered chunk
    WS   /sessions/{id}/ws            the live conversation

The REST turn endpoint exists because it is the fastest way to check a whole pipeline —
character, planner, normalizer, chunker, retriever, engine — with `curl` and no browser.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from cvai_conversation import OrchestratorConfig
from cvai_core.errors import CVAIError
from cvai_types import InterruptReason, InterruptSignal, TurnEventType

from .events import to_server_messages
from .sessions import SessionManager

log = logging.getLogger("cvai-api")

# Imported at module level, not inside create_app, and that matters: this module uses
# `from __future__ import annotations`, so FastAPI resolves parameter annotations as
# strings against the *module* globals. With the imports local to a function, the
# annotation `socket: WebSocket` resolves to nothing, FastAPI decides `socket` must be a
# query parameter, and every WebSocket handshake is rejected with 403 — while the REST
# routes keep working, so it looks like a transport problem rather than a typing one.
try:  # pragma: no cover - import shape depends on the installed extras
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    FastAPI = HTTPException = WebSocket = WebSocketDisconnect = Any  # type: ignore[misc,assignment]
    FileResponse = JSONResponse = Any  # type: ignore[misc,assignment]

_MISSING_FASTAPI = "the API needs FastAPI: pip install -e '.[runtime]'"


def create_app(
    manager: SessionManager | None = None,
    *,
    config_path: Path | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
) -> Any:
    if not FASTAPI_AVAILABLE:  # pragma: no cover - depends on extras
        raise SystemExit(_MISSING_FASTAPI)

    sessions = manager or SessionManager.from_config_file(
        config_path,
        orchestrator_config=orchestrator_config or OrchestratorConfig(),
    )
    @contextlib.asynccontextmanager
    async def lifespan(_: Any):
        # Modern lifespan rather than @app.on_event: the latter is deprecated, and
        # a DeprecationWarning in a library is a warning the project's own tests
        # should not be teaching people to ignore.
        yield
        await sessions.close_all()

    app = FastAPI(title="Character Voice AI", version="0.1.0", lifespan=lifespan)
    app.state.sessions = sessions

    # -- basics -------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "default_character": sessions.config.default_character,
            "tts": sessions.config.providers.tts.active,
            "open_sessions": len(sessions.sessions),
        }

    @app.get("/characters")
    def characters() -> dict[str, Any]:
        return {"characters": sessions.available_characters()}

    # -- sessions -----------------------------------------------------------------

    @app.post("/sessions")
    def open_session(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            session = sessions.create(
                payload.get("character_id"),
                engine=payload.get("engine"),
                enable_barge_in=bool(payload.get("enable_barge_in", True)),
            )
        except CVAIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "session_id": session.session_id,
            "character_id": session.character_id,
            "character_name": session.profile.character_name,
            "engine": session.engine,
            "state": session.orchestrator.state.value,
        }

    @app.delete("/sessions/{session_id}")
    async def close_session(session_id: str) -> dict[str, bool]:
        await sessions.close(session_id)
        return {"closed": True}

    @app.post("/sessions/{session_id}/turn")
    async def run_turn(session_id: str, payload: dict[str, Any]) -> JSONResponse:
        text = (payload or {}).get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="'text' is required")
        try:
            session = sessions.get(session_id)
        except CVAIError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        events = await session.orchestrator.collect(text)
        return JSONResponse(
            {
                "turn_id": events[0].turn_id if events else None,
                "text": next(
                    (
                        e.text
                        for e in events
                        if e.type is TurnEventType.TEXT_DELTA and e.is_final
                    ),
                    "",
                ),
                "audio": [
                    {
                        "chunk_index": e.chunk_index,
                        "text": e.text,
                        "url": _audio_url(session_id, session.orchestrator, e.audio_path),
                        "duration_s": e.duration_s,
                    }
                    for e in events
                    if e.type is TurnEventType.AUDIO
                ],
                "errors": [e.error for e in events if e.type is TurnEventType.ERROR],
            }
        )

    @app.get("/sessions/{session_id}/audio/{path:path}")
    def audio(session_id: str, path: str):
        try:
            return FileResponse(sessions.resolve_audio(session_id, path), media_type="audio/wav")
        except CVAIError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- websocket ----------------------------------------------------------------

    @app.websocket("/sessions/{session_id}/ws")
    async def websocket(socket: WebSocket, session_id: str) -> None:
        await socket.accept()
        try:
            session = sessions.get(session_id)
        except CVAIError as exc:
            await socket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await socket.close()
            return

        orchestrator = session.orchestrator
        turn_task: asyncio.Task | None = None

        async def run(text: str) -> None:
            try:
                async for event in orchestrator.run_turn(text):
                    for message in to_server_messages(event):
                        await socket.send_text(
                            json.dumps(message.model_dump(mode="json"), ensure_ascii=False)
                        )
                    if event.type is TurnEventType.AUDIO and event.audio_path:
                        # Binary frames follow their AudioBegin, so the client knows the
                        # format before the bytes arrive.
                        await socket.send_bytes(Path(event.audio_path).read_bytes())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("turn failed")
                await socket.send_text(
                    json.dumps({"type": "error", "code": "turn_failed", "message": str(exc)})
                )

        try:
            while True:
                raw = await socket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = payload.get("type")

                if kind == "user_text":
                    # A new turn supersedes the old one. Interrupting rather than
                    # queueing is what makes the character feel responsive instead of
                    # finishing a sentence nobody is listening to any more.
                    if turn_task and not turn_task.done():
                        await orchestrator.interrupt(
                            InterruptSignal(reason=InterruptReason.NEW_TURN)
                        )
                        await asyncio.gather(turn_task, return_exceptions=True)
                    turn_task = asyncio.create_task(run(payload.get("text", "")))

                elif kind == "interrupt":
                    await orchestrator.interrupt(
                        InterruptSignal(reason=InterruptReason.USER_SPEECH),
                        turn_id=payload.get("turn_id"),
                    )

                elif kind == "bye":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if turn_task and not turn_task.done():
                await orchestrator.interrupt(
                    InterruptSignal(reason=InterruptReason.USER_CANCEL)
                )
                await asyncio.gather(turn_task, return_exceptions=True)

    return app


def _audio_url(session_id: str, orchestrator, audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    base = Path(orchestrator._audio_root)  # noqa: SLF001 - same package boundary
    try:
        relative = Path(audio_path).resolve().relative_to(base.resolve())
    except ValueError:
        return None
    return f"/sessions/{session_id}/audio/{relative.as_posix()}"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(prog="cvai-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="path to app.yaml")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("pip install -e '.[runtime]'") from exc

    uvicorn.run(
        create_app(config_path=Path(args.config) if args.config else None),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
