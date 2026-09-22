"""API layer: session assembly and protocol mapping.

FastAPI is not installed in the base environment, so the routes themselves are not
exercised here — but almost nothing interesting lives in them. Session assembly (six
things wired together, each able to be missing) and the event→protocol mapping (where a
field can quietly go missing, or a field the user must not see can quietly leak) are
both pure, and both are tested.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cvai_api import SessionManager, stream_to_messages, to_server_messages
from cvai_audio_protocol import (
    AudioBegin,
    AudioEnd,
    CharacterText,
    ErrorMessage,
    StateChanged,
    TurnEnd,
)
from cvai_core.errors import CVAIError, VoicePackError
from cvai_types import ConversationState, TurnEvent, TurnEventType

from test_orchestrator import FakeLLM  # noqa: E402 - shared fake


# --------------------------------------------------------------------------------------
# Event mapping
# --------------------------------------------------------------------------------------


def test_state_events_become_state_messages():
    event = TurnEvent(
        type=TurnEventType.STATE,
        turn_id="t1",
        state=ConversationState.SPEAKING,
        detail="audio ready",
    )
    (message,) = to_server_messages(event)
    assert isinstance(message, StateChanged)
    assert message.state is ConversationState.SPEAKING
    assert message.turn_id == "t1"


def test_text_becomes_a_caption():
    event = TurnEvent(
        type=TurnEventType.TEXT_DELTA, turn_id="t1", text="哼，怎么了。", is_final=True
    )
    (message,) = to_server_messages(event)
    assert isinstance(message, CharacterText)
    assert message.text == "哼，怎么了。"
    assert message.is_final


def test_audio_is_bracketed_by_begin_and_end():
    event = TurnEvent(
        type=TurnEventType.AUDIO,
        turn_id="t1",
        chunk_index=2,
        text="随你便。",
        sample_rate=48000,
        duration_s=1.25,
        audio_path="/tmp/x.wav",
    )
    begin, end = to_server_messages(event)
    assert isinstance(begin, AudioBegin) and isinstance(end, AudioEnd)
    assert begin.sample_rate == 48000
    assert begin.chunk_index == end.chunk_index == 2
    assert end.duration_ms == pytest.approx(1250.0)


def test_performance_metadata_never_reaches_the_browser():
    """Spec §12: the user only sees or hears the text."""
    plan = TurnEvent(
        type=TurnEventType.PLAN,
        turn_id="t1",
        text="随你便。",
        detail="soft_teasing · slow · structured",
    )
    assert to_server_messages(plan) == []

    audio = TurnEvent(
        type=TurnEventType.AUDIO,
        turn_id="t1",
        chunk_index=0,
        text="随你便。",
        audio_path="/tmp/x.wav",
        reference_id="soft_teasing_01",
        engine="gpt_sovits",
        latency_ms=812.0,
    )
    dumped = " ".join(
        str(m.model_dump(mode="json")) for m in to_server_messages(audio)
    )
    assert "soft_teasing_01" not in dumped
    assert "gpt_sovits" not in dumped
    assert "812" not in dumped


def test_chunk_events_are_superseded_by_audio():
    event = TurnEvent(type=TurnEventType.CHUNK, turn_id="t1", text="随你便。", chunk_index=0)
    assert to_server_messages(event) == []


def test_turn_end_carries_whether_it_was_interrupted():
    normal = to_server_messages(TurnEvent(type=TurnEventType.TURN_END, turn_id="t1"))
    cut = to_server_messages(
        TurnEvent(type=TurnEventType.TURN_END, turn_id="t1", detail="interrupted")
    )
    assert isinstance(normal[0], TurnEnd) and normal[0].was_interrupted is False
    assert cut[0].was_interrupted is True


def test_errors_are_forwarded_as_recoverable():
    (message,) = to_server_messages(
        TurnEvent(type=TurnEventType.ERROR, turn_id="t1", error="engine exploded")
    )
    assert isinstance(message, ErrorMessage)
    assert "exploded" in message.message
    assert message.recoverable


def test_a_whole_turn_maps_to_a_sensible_message_sequence():
    events = [
        TurnEvent(type=TurnEventType.STATE, turn_id="t1", state=ConversationState.THINKING),
        TurnEvent(type=TurnEventType.PLAN, turn_id="t1", text="随你便。"),
        TurnEvent(type=TurnEventType.TEXT_DELTA, turn_id="t1", text="随你便。", is_final=True),
        TurnEvent(type=TurnEventType.CHUNK, turn_id="t1", text="随你便。", chunk_index=0),
        TurnEvent(
            type=TurnEventType.AUDIO,
            turn_id="t1",
            chunk_index=0,
            text="随你便。",
            audio_path="/tmp/x.wav",
            duration_s=1.0,
            sample_rate=24000,
        ),
        TurnEvent(type=TurnEventType.TURN_END, turn_id="t1"),
    ]
    kinds = [type(m).__name__ for m in stream_to_messages(events)]
    assert kinds == [
        "StateChanged",
        "CharacterText",
        "AudioBegin",
        "AudioEnd",
        "TurnEnd",
    ]


# --------------------------------------------------------------------------------------
# Session assembly
# --------------------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path, demo_pack, repo_root_path: Path, monkeypatch):
    """A SessionManager pointed at the synthetic pack, with fake providers."""
    from cvai_core.config import load_config
    from cvai_tts_providers import MockTTSProvider

    characters = tmp_path / "characters" / "profiles"
    characters.mkdir(parents=True)
    (characters / "demo_zh.yaml").write_text(
        "\n".join(
            [
                "character_id: demo_zh",
                "character_name: 测试角色",
                "voice:",
                "  voicepack_id: demo_zh",
                "  default_reference_style: neutral",
                "available_styles: [neutral, soft, teasing]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CVAI_REPO_ROOT", str(tmp_path))

    config = load_config(
        [repo_root_path / "configs" / "app.yaml"], use_env_overrides=False
    )
    config = config.model_copy(
        update={
            "default_character": "demo_zh",
            "paths": config.paths.model_copy(
                update={
                    "characters": "characters/profiles",
                    "voicepacks": str(demo_pack.root.parent),
                }
            ),
        }
    )
    from test_listening import FakeSTT

    return SessionManager(
        config=config,
        audio_root=tmp_path / "audio",
        tts_factory=lambda cfg, name: MockTTSProvider(sample_rate=16000),
        llm_factory=lambda cfg, name: FakeLLM(),
        stt_factory=lambda cfg, name: FakeSTT("外面下雨了吗"),
    )


def test_opening_a_session_wires_the_whole_stack(manager: SessionManager):
    session = manager.create()

    assert session.character_id == "demo_zh"
    assert session.engine == "mock"
    assert session.orchestrator.retriever.coverage()
    assert manager.get(session.session_id) is session


def test_a_turn_runs_through_a_managed_session(manager: SessionManager):
    session = manager.create()
    events = asyncio.run(session.orchestrator.collect("在吗"))

    audio = [e for e in events if e.type is TurnEventType.AUDIO]
    assert audio
    assert Path(audio[0].audio_path).is_file()


def test_an_unknown_session_is_a_clear_error(manager: SessionManager):
    with pytest.raises(CVAIError):
        manager.get("nope")


def test_a_pack_without_a_reference_bank_says_what_to_run(manager: SessionManager, monkeypatch):
    """The most likely first-run failure deserves an actionable message."""
    import cvai_api.sessions as module

    monkeypatch.setattr(module, "try_load_reference_bank", lambda paths: None)
    with pytest.raises(VoicePackError) as exc:
        manager.create()
    assert "cvai-prep build" in str(exc.value)


def test_closing_a_session_removes_it(manager: SessionManager):
    session = manager.create()
    asyncio.run(manager.close(session.session_id))
    assert session.session_id not in manager.sessions


def test_audio_paths_cannot_escape_the_session_directory(manager: SessionManager):
    """The path comes from a browser, so it is untrusted input."""
    session = manager.create()
    asyncio.run(session.orchestrator.collect("在吗"))

    with pytest.raises(CVAIError):
        manager.resolve_audio(session.session_id, "../../../etc/passwd")
    with pytest.raises(CVAIError):
        manager.resolve_audio(session.session_id, "nope.wav")


def test_resolving_a_real_audio_file_works(manager: SessionManager):
    session = manager.create()
    events = asyncio.run(session.orchestrator.collect("在吗"))
    audio = [e for e in events if e.type is TurnEventType.AUDIO][0]

    base = Path(manager.audio_root) / session.session_id
    relative = Path(audio.audio_path).relative_to(base)
    assert manager.resolve_audio(session.session_id, str(relative)).is_file()


def test_the_character_preferred_engine_wins(manager: SessionManager):
    """A Milestone 7 decision recorded in the profile takes effect everywhere."""
    seen: list[str | None] = []

    def factory(config, name):
        from cvai_tts_providers import MockTTSProvider

        seen.append(name)
        return MockTTSProvider()

    manager.tts_factory = factory
    profile = manager._characters.get("demo_zh")  # noqa: SLF001 - test reaching in
    manager._characters._cache["demo_zh"] = profile.model_copy(  # noqa: SLF001
        update={"voice": profile.voice.model_copy(update={"preferred_engine": "voxcpm"})}
    )

    manager.create(engine="gpt_sovits")
    assert seen == ["voxcpm"]


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
#
# These need FastAPI, which lives in the `runtime` extra, so they skip on a base install.
# They exist because one bug class is invisible to the pure tests above: this module uses
# `from __future__ import annotations`, so FastAPI resolves endpoint annotations against
# the *module* globals. Import `WebSocket` inside a function instead and every handshake
# is rejected with 403 while REST keeps working — which reads as a transport problem and
# is not. Only an actual handshake catches it.

fastapi = pytest.importorskip("fastapi", reason="install the 'runtime' extra")


@pytest.fixture
def client(manager: SessionManager):
    from fastapi.testclient import TestClient

    from cvai_api.app import create_app

    with TestClient(create_app(manager)) as test_client:
        yield test_client


def test_health_and_characters(client):
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["tts"]
    assert "demo_zh" in client.get("/characters").json()["characters"]


def test_rest_turn_returns_text_and_playable_audio(client):
    session_id = client.post("/sessions", json={}).json()["session_id"]
    turn = client.post(f"/sessions/{session_id}/turn", json={"text": "在吗"}).json()

    assert turn["text"]
    assert turn["audio"]
    assert not turn["errors"]

    for clip in turn["audio"]:
        response = client.get(clip["url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"


def test_a_turn_without_text_is_rejected(client):
    session_id = client.post("/sessions", json={}).json()["session_id"]
    assert client.post(f"/sessions/{session_id}/turn", json={"text": "  "}).status_code == 400


def test_an_unknown_session_is_a_404(client):
    assert client.post("/sessions/nope/turn", json={"text": "在吗"}).status_code == 404
    assert client.get("/sessions/nope/audio/x.wav").status_code == 404


def test_audio_traversal_is_refused_over_http(client):
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/turn", json={"text": "在吗"})
    assert client.get(f"/sessions/{session_id}/audio/../../etc/passwd").status_code == 404


def test_websocket_handshake_succeeds_and_streams_a_turn(client):
    """The regression test for the 403: the handshake itself must work."""
    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.websocket_connect(f"/sessions/{session_id}/ws") as socket:
        socket.send_text(json.dumps({"type": "user_text", "text": "在吗"}))
        kinds, audio_frames = [], 0
        for _ in range(60):
            message = socket.receive()
            if message.get("bytes"):
                audio_frames += 1
                continue
            payload = json.loads(message["text"])
            kinds.append(payload["type"])
            if payload["type"] == "turn_end":
                break

    assert kinds[0] == "state"
    assert "character_text" in kinds
    assert kinds.count("audio_begin") == kinds.count("audio_end") == audio_frames
    assert audio_frames >= 1
    assert kinds[-1] == "turn_end"


def test_websocket_audio_frames_are_real_wav(client):
    session_id = client.post("/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/sessions/{session_id}/ws") as socket:
        socket.send_text(json.dumps({"type": "user_text", "text": "在吗"}))
        for _ in range(60):
            message = socket.receive()
            if message.get("bytes"):
                assert message["bytes"][:4] == b"RIFF"
                return
            if json.loads(message["text"])["type"] == "turn_end":
                break
    pytest.fail("no audio frame arrived")


def test_websocket_for_an_unknown_session_reports_and_closes(client):
    with client.websocket_connect("/sessions/nope/ws") as socket:
        payload = json.loads(socket.receive()["text"])
    assert payload["type"] == "error"


# --------------------------------------------------------------------------------------
# The microphone path (Milestone 11)
# --------------------------------------------------------------------------------------
#
# The upstream half of the socket: raw 16-bit PCM in, a spoken turn out. The audio is
# synthetic (a tone between two stretches of room noise), because what is being tested
# here is the wiring — that binary frames reach the listener, that endpointing starts a
# turn, and that the transcript is echoed back — not the acoustics, which
# `test_listening.py` covers.


def _mic_frames() -> list[bytes]:
    from test_listening import floats_to_pcm16, room_noise, silence, tone
    from cvai_conversation import ListenerConfig

    config = ListenerConfig()
    return [
        floats_to_pcm16(room_noise(400)),
        floats_to_pcm16(tone(700)),
        floats_to_pcm16(silence(config.silence_end_ms + 100)),
    ]


def _collect(socket, limit: int = 80) -> tuple[list[dict], int]:
    messages, audio_frames = [], 0
    for _ in range(limit):
        message = socket.receive()
        if message.get("bytes"):
            audio_frames += 1
            continue
        payload = json.loads(message["text"])
        messages.append(payload)
        if payload["type"] == "turn_end":
            break
    return messages, audio_frames


def test_speaking_into_the_socket_runs_a_turn(client):
    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.websocket_connect(f"/sessions/{session_id}/ws") as socket:
        socket.send_text(json.dumps({"type": "user_audio_begin", "sample_rate": 16000}))
        for frame in _mic_frames():
            socket.send_bytes(frame)
        messages, audio_frames = _collect(socket)

    kinds = [message["type"] for message in messages]
    assert kinds[0] == "transcript"
    assert messages[0]["text"] == "外面下雨了吗"
    assert "character_text" in kinds
    assert kinds[-1] == "turn_end"
    assert audio_frames >= 1


def test_microphone_frames_before_user_audio_begin_are_ignored(client):
    """A page that starts streaming before saying so must not open a turn.

    Not pedantry: the capture node runs for a moment after the mic is switched off, and
    those frames arriving as a new utterance would have the character answer nothing.
    """
    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.websocket_connect(f"/sessions/{session_id}/ws") as socket:
        for frame in _mic_frames():
            socket.send_bytes(frame)
        # Nothing should have been produced, so a typed turn is the next thing heard.
        socket.send_text(json.dumps({"type": "user_text", "text": "在吗"}))
        messages, _ = _collect(socket)

    assert [m for m in messages if m["type"] == "transcript"] == []
    assert messages[-1]["type"] == "turn_end"


def test_ending_the_audio_stream_flushes_a_part_spoken_utterance(client):
    """Push-to-talk: releasing the button ends the utterance without the silence timer."""
    from test_listening import floats_to_pcm16, room_noise, tone

    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.websocket_connect(f"/sessions/{session_id}/ws") as socket:
        socket.send_text(json.dumps({"type": "user_audio_begin", "sample_rate": 16000}))
        socket.send_bytes(floats_to_pcm16(room_noise(400)))
        socket.send_bytes(floats_to_pcm16(tone(700)))
        socket.send_text(json.dumps({"type": "user_audio_end"}))
        messages, _ = _collect(socket)

    assert messages[0]["type"] == "transcript"
    assert messages[-1]["type"] == "turn_end"


def test_a_session_without_speech_to_text_says_so(manager: SessionManager):
    """The microphone is the only thing that needs STT, so its absence is only an
    error for the user who speaks — a typed conversation must still work."""
    from fastapi.testclient import TestClient

    from cvai_api.app import create_app
    from cvai_core.errors import CVAIError

    manager.stt_factory = None
    manager.config = manager.config.model_copy(
        update={"providers": manager.config.providers.model_copy(update={"stt": None})}
    )

    with TestClient(create_app(manager)) as test_client:
        session_id = test_client.post("/sessions", json={}).json()["session_id"]
        session = manager.get(session_id)
        with pytest.raises(CVAIError, match="speech-to-text"):
            session.voice_loop()
        # Typing still works.
        turn = test_client.post(f"/sessions/{session_id}/turn", json={"text": "在吗"})
        assert turn.status_code == 200
