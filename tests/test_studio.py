"""Character AI Studio: domain rules, services and API, against a temporary data dir.

Training runs through a fake engine (no GPU) and the queue is a function call, so the
whole job lifecycle — including failure, cancellation and retry — is exercised in-process.
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import wave
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("cryptography")

from cvai_studio.core.config import StudioSettings  # noqa: E402
from cvai_studio.core.logging import redact  # noqa: E402
from cvai_studio.core.security import FernetSecretStore, SecretError, mask_secret  # noqa: E402
from cvai_studio.db.models import AudioSample, ModelConnection, TrainingJob, VoiceModel  # noqa: E402
from cvai_studio.domain.enums import (  # noqa: E402
    ConnectionKind,
    ConnectionStatus,
    InvalidTransition,
    ProviderType,
    TrainingStatus,
    VoicePackStatus,
    transition,
)
from cvai_studio.providers.storage import LocalFilesystemStorage, StorageError  # noqa: E402
from cvai_studio.providers.training.base import (  # noqa: E402
    AdvancedOption,
    PresetOption,
    TrainingFailure,
    TrainingResult,
    VoiceTrainingProvider,
)
from cvai_studio.services import training as training_module  # noqa: E402
from cvai_studio.services.characters import CharacterData, CharacterService  # noqa: E402
from cvai_studio.services.context import Conflict, ServiceError, Unavailable, build_studio  # noqa: E402
from cvai_studio.services.model_connections import (  # noqa: E402
    ConnectionDraft,
    ConnectionPatch,
    ModelConnectionService,
    friendly_error,
    validate_endpoint,
)
from cvai_studio.services.training import JobRunner, TrainingRequest, TrainingService  # noqa: E402
from cvai_studio.services.voice_packs import SamplePatch, VoicePackCreate, VoicePackService  # noqa: E402


# -- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def studio(tmp_path: Path):
    settings = StudioSettings(
        data_dir=tmp_path / "data", database_url=f"sqlite:///{(tmp_path / 'studio.db').as_posix()}",
        gpt_sovits_dir=tmp_path / "engine", secret_key=None,
    )
    return build_studio(settings)


def tone_wav(seconds: float = 4.0, rate: int = 16000) -> bytes:
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
                      for i in range(int(seconds * rate)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(frames)
    return buffer.getvalue()


class FakeEngine(VoiceTrainingProvider):
    engine = "gpt_sovits"
    label = "Fake"
    base_model = "fake-1"

    def __init__(self, fail: bool = False, cancel_after: int | None = None) -> None:
        self.fail = fail
        self.cancel_after = cancel_after

    def presets(self):
        return [PresetOption("quick", "快速", "", {"sovits_epochs": 2, "gpt_epochs": 2, "batch_size": 4}),
                PresetOption("balanced", "均衡", "", {"sovits_epochs": 4, "gpt_epochs": 4, "batch_size": 8})]

    def advanced_options(self):
        return [AdvancedOption("batch_size", "批大小", "int", 8, 1, 32)]

    def validate_dataset(self, samples):
        return [] if samples else ["没有片段"]

    def prepare_dataset(self, samples, workdir):
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def train(self, dataset, workdir, config, *, job_id, emit, cancelled):
        from cvai_studio.providers.training.base import TrainingCancelled, TrainingEvent

        total = config["sovits_epochs"] + config["gpt_epochs"]
        for epoch in range(1, total + 1):
            if self.cancel_after is not None and epoch > self.cancel_after:
                if cancelled():
                    raise TrainingCancelled()
            emit(TrainingEvent(type="epoch_completed", epoch=epoch, total_epochs=total, progress=epoch / total))
        if self.fail:
            raise TrainingFailure("训练时显存不足", hint="减小批大小", detail="CUDA out of memory")
        out = workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "gpt.ckpt").write_bytes(b"g")
        (out / "sovits.pth").write_bytes(b"s")
        return TrainingResult({"gpt": out / "gpt.ckpt", "sovits": out / "sovits.pth"}, base_model="fake-1")

    def synthesize(self, text, *, checkpoint, reference_audio, reference_text, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(tone_wav(1.0))
        return out_path


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(training_module, "engines", lambda studio: {"gpt_sovits": fake})
    return fake


def ready_pack(studio, n: int = 25) -> str:
    """A pack with n approved, transcribed samples in READY — without running the pipeline."""
    with studio.db() as db:
        service = VoicePackService(studio, db)
        pack = service.create(VoicePackCreate(name="P", character_name="小测"))
        for i in range(n):
            key = studio.storage.save(f"{pack.storage_prefix}/processed/{i}.wav", tone_wav(5.0))
            pack.samples.append(AudioSample(audio_key=key, transcript=f"第{i}句话。", duration_s=5.0,
                                            style="neutral" if i % 2 else "soft", approved=True, position=i))
        pack.status = VoicePackStatus.NEEDS_REVIEW
        pack.total_files = n
        db.flush()
        service.confirm_rights(pack.id)
        service.approve_dataset(pack.id)
        return pack.id


# -- domain -----------------------------------------------------------------------------


def test_training_transitions_are_explicit():
    assert transition(TrainingStatus.QUEUED, TrainingStatus.VALIDATING) is TrainingStatus.VALIDATING
    assert transition(TrainingStatus.TRAINING, TrainingStatus.FAILED) is TrainingStatus.FAILED
    with pytest.raises(InvalidTransition):
        transition(TrainingStatus.COMPLETED, TrainingStatus.TRAINING)
    with pytest.raises(InvalidTransition):
        transition(TrainingStatus.QUEUED, TrainingStatus.TRAINING)      # no skipping stages


def test_voice_pack_and_connection_transitions():
    assert transition(VoicePackStatus.NEEDS_REVIEW, VoicePackStatus.READY)
    with pytest.raises(InvalidTransition):
        transition(VoicePackStatus.EMPTY, VoicePackStatus.READY)
    assert transition(ConnectionStatus.ERROR, ConnectionStatus.TESTING)
    with pytest.raises(InvalidTransition):
        transition(ConnectionStatus.DRAFT, ConnectionStatus.CONNECTED)   # must be tested


# -- secrets & storage ----------------------------------------------------------------------


def test_secrets_round_trip_and_never_appear_in_plaintext(studio):
    with studio.db() as db:
        service = ModelConnectionService(studio, db)
        connection = service.create(ConnectionDraft(
            name="API", kind=ConnectionKind.PUBLIC_API, provider_type=ProviderType.OPENAI_COMPATIBLE,
            base_url="https://api.example.com/v1", model_name="m", api_key="sk-proj-SECRETVALUE1234"))
        assert "SECRETVALUE" not in (connection.encrypted_api_key or "")
        assert connection.api_key_hint == "sk-••••••••1234"
        assert service.api_key(connection) == "sk-proj-SECRETVALUE1234"
    raw = (Path(studio.settings.database_url.removeprefix("sqlite:///"))).read_bytes()
    assert b"SECRETVALUE" not in raw


def test_a_different_secret_key_cannot_decrypt(tmp_path):
    token = FernetSecretStore.from_settings(None, tmp_path / "a").encrypt("sk-1")
    with pytest.raises(SecretError):
        FernetSecretStore.from_settings(None, tmp_path / "b").decrypt(token)


def test_mask_and_redact():
    assert mask_secret("short") == "••••••••"
    redacted = redact("key sk-proj-ABCDEFGH123456 used")
    assert redacted.startswith("key sk-proj-") and "123456" not in redacted and "[redacted]" in redacted
    assert "secret" not in redact("Authorization: Bearer secret")


def test_storage_refuses_paths_outside_the_data_dir(tmp_path):
    storage = LocalFilesystemStorage(tmp_path)
    for key in ("../escape.txt", "/abs.txt", "a/../../b"):
        with pytest.raises(StorageError):
            storage.local_path(key)
    assert storage.save("ok/file.txt", b"x") == "ok/file.txt"


# -- model connections ----------------------------------------------------------------------


def test_first_connection_becomes_default_and_editing_resets_status(studio):
    with studio.db() as db:
        service = ModelConnectionService(studio, db)
        a = service.create(ConnectionDraft(name="A", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
                                           base_url="http://localhost:11434/", model_name="qwen3:4b"))
        b = service.create(ConnectionDraft(name="B", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
                                           base_url="http://localhost:11434", model_name="qwen3:8b"))
        assert service.default_id() == a.id and a.base_url == "http://localhost:11434"
        service.set_default(b.id)
        assert service.default_id() == b.id
        b.status = ConnectionStatus.CONNECTED
        service.update(b.id, ConnectionPatch(model_name="qwen3:14b"))
        assert b.status is ConnectionStatus.DRAFT            # stale test result discarded


def test_hosted_mode_refuses_private_endpoints(studio):
    studio.settings.allow_private_model_endpoints = False
    with pytest.raises(ServiceError, match="内网"):
        validate_endpoint("http://127.0.0.1:11434", studio)
    with pytest.raises(ServiceError):
        validate_endpoint("ftp://example.com", studio)
    with pytest.raises(ServiceError, match="账号密码"):
        validate_endpoint("https://user:pw@example.com/v1", studio)


def test_connection_errors_are_explained():
    assert "ollama pull qwen3:4b" in friendly_error("Model 'qwen3:4b' is not downloaded", ProviderType.OLLAMA, "u", "qwen3:4b")[1]
    assert "API Key" in friendly_error("Error code: 401 - invalid api key", ProviderType.OPENAI_COMPATIBLE, "u", "m")[0]
    assert "额度" in friendly_error("429 insufficient_quota", ProviderType.OPENAI_COMPATIBLE, "u", "m")[0]
    # A provider that answers 503 is reachable, only busy: never "cannot connect".
    busy = friendly_error("Error code: 503 - {'error': {'status': 'UNAVAILABLE'}}", ProviderType.OPENAI_COMPATIBLE, "u", "m")
    assert busy[0] == "服务商暂时不可用"
    assert friendly_error("model gpt-503 not found", ProviderType.OPENAI_COMPATIBLE, "u", "gpt-503")[0].startswith("服务商没有")


def test_openai_connection_test_makes_one_attempt():
    """The test button must not sit through the client's retries and backoff."""
    import asyncio

    from cvai_llm_providers import OpenAILLMProvider

    seen: dict = {}

    class Completions:
        async def create(self, **kwargs):
            raise RuntimeError("Error code: 503 - overloaded")

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

        def with_options(self, **options):
            seen.update(options)
            return self

    provider = OpenAILLMProvider(model="m", api_key="k", base_url="https://example.invalid/v1")
    provider._client = Client()
    result = asyncio.run(provider.test_connection())
    assert result["ok"] is False and "503" in result["error"]
    assert seen["max_retries"] == 0 and seen["timeout"] <= 30


def test_testing_an_unreachable_ollama_marks_error(studio):
    with studio.db() as db:
        service = ModelConnectionService(studio, db)
        c = service.create(ConnectionDraft(name="Dead", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
                                           base_url="http://127.0.0.1:9", model_name="qwen3:4b"))
        tested = asyncio.run(service.test(c.id))
        assert tested.status is ConnectionStatus.ERROR and "无法连接" in tested.last_error


def test_a_connection_in_use_cannot_be_deleted(studio, engine):
    with studio.db() as db:
        service = ModelConnectionService(studio, db)
        c = service.create(ConnectionDraft(name="A", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
                                           base_url="http://localhost:11434", model_name="qwen3:4b"))
        CharacterService(studio, db).create(CharacterData(name="X", model_connection_id=c.id))
        with pytest.raises(Conflict):
            service.delete(c.id)


# -- voice packs ------------------------------------------------------------------------------


def test_upload_validates_decoding_and_zip_safety(studio):
    with studio.db() as db:
        service = VoicePackService(studio, db)
        pack = service.create(VoicePackCreate(name="P", character_name="小测"))
        assert service.add_file(pack.id, "line.wav", io.BytesIO(tone_wav())).accepted
        assert service.add_file(pack.id, "fake.wav", io.BytesIO(b"not audio")).rejected[0]["reason"] == "无法解码为音频"
        assert service.add_file(pack.id, "x.exe", io.BytesIO(b"MZ")).rejected
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../../evil.wav", tone_wav())          # traversal attempt
            z.writestr("dir/line2.wav", tone_wav())
            z.writestr("dir/line2.lab", "你好。")
        archive.seek(0)
        result = service.add_file(pack.id, "pack.zip", archive)
        assert len(result.accepted) == 3
        assert not (studio.settings.data_dir.parent / "evil.wav").exists()
        assert pack.total_files == 3 and pack.status is VoicePackStatus.UPLOADING
        stored = list(studio.storage.list(f"{pack.storage_prefix}/raw"))
        assert all(Path(k).stem not in ("evil", "line", "line2") for k in stored)   # server-side names


def test_review_rules(studio):
    with studio.db() as db:
        service = VoicePackService(studio, db)
        pack = service.create(VoicePackCreate(name="P", character_name="小测"))
        pack.samples.append(AudioSample(audio_key="k", transcript="", duration_s=3.0, approved=False))
        pack.status = VoicePackStatus.NEEDS_REVIEW
        db.flush()
        sample = pack.samples[0]
        with pytest.raises(ServiceError, match="没有文字"):
            service.update_sample(sample.id, SamplePatch(approved=True))
        service.update_sample(sample.id, SamplePatch(transcript="补上的文字。", approved=True))
        assert sample.approved and sample.human_edited and sample.transcript_source == "human"
        with pytest.raises(ServiceError, match="可用数据不足"):
            service.approve_dataset(pack.id)


# -- training -----------------------------------------------------------------------------------


def run_job(studio, job_id):
    JobRunner(studio, job_id).run()
    with studio.db() as db:
        return db.get(TrainingJob, job_id)


def test_training_lifecycle_creates_a_voice_model(studio, engine):
    pack_id = ready_pack(studio)
    queued = []
    with studio.db() as db:
        job = TrainingService(studio, db, dispatch=lambda jid: queued.append(jid) or "task-1").create(
            TrainingRequest(voice_pack_id=pack_id, preset="quick", advanced={"batch_size": 99}))
        job_id = job.id
        assert job.status is TrainingStatus.QUEUED and job.config["batch_size"] == 32   # clamped
    assert queued == [job_id]
    job = run_job(studio, job_id)
    assert job.status is TrainingStatus.COMPLETED and job.progress == 1.0
    kinds = [e["type"] for e in job.events]
    assert kinds[:4] == ["job_queued", "dataset_validating", "dataset_preparing", "training_started"]
    assert kinds.count("epoch_completed") == 4 and kinds[-2:] == ["evaluation_started", "completed"]
    with studio.db() as db:
        model = db.get(VoiceModel, job.voice_model_id)
        assert model.name == "小测 声音 v1" and set(model.styles) == {"neutral", "soft"}
        assert all(studio.storage.exists(k) for k in model.checkpoint.values())
        assert len(model.evaluation_metadata["samples"]) == 3


def test_training_failure_is_explained_and_retryable(studio, engine):
    engine.fail = True
    pack_id = ready_pack(studio)
    with studio.db() as db:
        job_id = TrainingService(studio, db, dispatch=lambda jid: "t").create(TrainingRequest(voice_pack_id=pack_id)).id
    job = run_job(studio, job_id)
    assert job.status is TrainingStatus.FAILED
    assert job.error_message == "训练时显存不足" and job.error_hint == "减小批大小" and "CUDA" in job.error_detail
    engine.fail = False
    with studio.db() as db:
        retried = TrainingService(studio, db, dispatch=lambda jid: "t").retry(job_id)
        assert retried.id != job_id and retried.status is TrainingStatus.QUEUED
    assert run_job(studio, retried.id).status is TrainingStatus.COMPLETED


def test_cancelling_a_running_job(studio, engine):
    pack_id = ready_pack(studio)
    with studio.db() as db:
        job_id = TrainingService(studio, db, dispatch=lambda jid: "t").create(TrainingRequest(voice_pack_id=pack_id)).id
    engine.cancel_after = 1
    original = engine.train

    def train_then_cancel(*args, **kwargs):
        with studio.db() as db:
            TrainingService(studio, db).cancel(job_id)          # user clicks Cancel mid-run
        return original(*args, **kwargs)
    engine.train = train_then_cancel
    job = run_job(studio, job_id)
    assert job.status is TrainingStatus.CANCELLED and job.voice_model_id is None


def test_queue_down_is_reported_not_hidden(studio, engine):
    pack_id = ready_pack(studio)

    def broker_down(job_id):
        raise ConnectionError("redis refused")
    with studio.db() as db:
        with pytest.raises(Unavailable, match="训练服务不可用"):
            TrainingService(studio, db, dispatch=broker_down).create(TrainingRequest(voice_pack_id=pack_id))
    with studio.db() as db:
        job = db.query(TrainingJob).one()
        assert job.status is TrainingStatus.FAILED and "Memurai" in job.error_hint


def test_training_requires_rights_and_an_approved_dataset(studio, engine):
    with studio.db() as db:
        pack = VoicePackService(studio, db).create(VoicePackCreate(name="P", character_name="X"))
        with pytest.raises(ServiceError, match="权"):
            TrainingService(studio, db, dispatch=lambda j: "t").create(TrainingRequest(voice_pack_id=pack.id))


# -- characters ------------------------------------------------------------------------------------


def test_character_references_by_id_and_follows_the_default(studio, engine):
    pack_id = ready_pack(studio)
    with studio.db() as db:
        job_id = TrainingService(studio, db, dispatch=lambda jid: "t").create(TrainingRequest(voice_pack_id=pack_id)).id
    voice_id = run_job(studio, job_id).voice_model_id
    with studio.db() as db:
        connections = ModelConnectionService(studio, db)
        a = connections.create(ConnectionDraft(name="A", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
                                               base_url="http://localhost:11434", model_name="qwen3:4b"))
        b = connections.create(ConnectionDraft(name="B", kind=ConnectionKind.PUBLIC_API,
                                               provider_type=ProviderType.OPENAI_COMPATIBLE,
                                               base_url="https://api.example.com/v1", model_name="m", api_key="sk-x"))
        service = CharacterService(studio, db)
        follows = service.create(CharacterData(name="跟随默认", voice_model_id=voice_id))
        pinned = service.create(CharacterData(name="指定", voice_model_id=voice_id, model_connection_id=b.id))
        assert service.connection_for(follows).id == a.id
        connections.set_default(b.id)
        assert service.connection_for(follows).id == b.id and service.connection_for(pinned).id == b.id
        parts = service.session_parts(pinned.id)
        assert parts.profile.character_id == pinned.id and parts.profile.voice.voicepack_id == voice_id
        assert parts.voicepack_root.name == voice_id and parts.checkpoint_id.count("|") == 1
        assert parts.llm.provider == "openai"


def test_incomplete_character_cannot_chat(studio):
    with studio.db() as db:
        service = CharacterService(studio, db)
        c = service.create(CharacterData(name="无声"))
        assert c.status.value == "incomplete"
        assert "还没有选择声音" in service.problems(c)
        with pytest.raises(ServiceError, match="还不能对话"):
            service.session_parts(c.id)


def test_avatar_upload_checks_the_bytes_not_the_name(studio):
    with studio.db() as db:
        service = CharacterService(studio, db)
        c = service.create(CharacterData(name="A"))
        with pytest.raises(ServiceError):
            service.set_avatar(c.id, b"<svg onload=alert(1)>")
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        assert service.set_avatar(c.id, png).avatar_key.endswith(".png")


# -- API --------------------------------------------------------------------------------------------


@pytest.fixture
def client(studio, engine, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cvai_studio.api import routes
    from cvai_studio.services.context import ServiceError as SE

    monkeypatch.setattr(routes, "get_studio", lambda: studio)
    monkeypatch.setattr(routes, "_dispatch", lambda task: (lambda entity_id: "task-x"))
    app = FastAPI()
    for router in routes.ROUTERS:
        app.include_router(router)
    app.add_exception_handler(SE, routes.service_error_handler)
    return TestClient(app)


def test_api_never_returns_api_keys(client):
    created = client.post("/api/model-connections", json={
        "name": "API", "kind": "public_api", "provider_type": "openai_compatible",
        "base_url": "https://api.example.com/v1", "model_name": "m", "api_key": "sk-proj-TOPSECRET9876"}).json()
    listed = client.get("/api/model-connections").text
    assert "TOPSECRET" not in str(created) and "TOPSECRET" not in listed
    assert created["api_key_hint"] == "sk-••••••••9876" and created["has_api_key"] and created["is_default"]


def test_api_errors_carry_message_and_hint(client):
    response = client.post("/api/model-connections", json={
        "name": "x", "kind": "local", "provider_type": "ollama", "base_url": "not a url", "model_name": "m"})
    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "地址格式不正确" and response.json()["detail"]["hint"]


def test_api_voice_pack_to_training_job(client, studio):
    pack = client.post("/api/voice-packs", json={"name": "P", "character_name": "小测"}).json()
    assert pack["status"] == "empty"
    up = client.post(f"/api/voice-packs/{pack['id']}/upload",
                     files=[("files", ("a.wav", tone_wav(), "audio/wav")), ("files", ("b.txt", b"\xff\xfe", "text/plain"))]).json()
    assert len(up["accepted"]) == 1 and up["rejected"][0]["reason"] == "文本不是 UTF-8 编码"
    response = client.post("/api/training-jobs", json={"voice_pack_id": pack["id"], "preset": "quick"})
    assert response.status_code == 400                                       # not reviewed yet
    pack_id = ready_pack(studio)
    job = client.post("/api/training-jobs", json={"voice_pack_id": pack_id, "preset": "quick"}).json()
    assert job["status"] == "queued" and job["events"][0]["type"] == "job_queued"
    assert client.post(f"/api/training-jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert client.get("/api/files/../studio.db").status_code == 404
