"""Studio HTTP API. Thin: validate, call a service, shape the response.

Deliberately no ``from __future__ import annotations``: FastAPI reads these annotations
at runtime.
"""

import asyncio
import json
import mimetypes
import time
from collections.abc import Iterator
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db.models import TrainingJob, VoicePack
from ..db.session import ping
from ..domain.enums import ProviderType, TRAINING_TERMINAL
from ..providers.storage import StorageError
from ..services.characters import CharacterData, CharacterPatch, CharacterService
from ..services.context import NotFound, ServiceError, Unavailable, get_studio
from ..services.model_connections import ConnectionDraft, ConnectionPatch, ModelConnectionService
from ..services.training import TrainingRequest, TrainingService
from ..services.voice_models import TEST_SENTENCES, VoiceModelService
from ..services.voice_packs import SamplePatch, VoicePackCreate, VoicePackService
from .schemas import (
    AudioSampleOut,
    CharacterOut,
    ModelConnectionOut,
    TrainingJobOut,
    VoiceModelOut,
    VoicePackOut,
    file_url,
)

FILE_PREFIXES = ("voicepacks/", "voice_models/", "generated/", "avatars/")


def db() -> Iterator[Session]:
    # Used as Depends(db, scope="function"): the commit happens before the response is
    # sent. With the default request scope the browser could refetch right after a write
    # and read the old row — e.g. an upload that then shows "0 files".
    with get_studio().db() as session:
        yield session


async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status,
                        content={"detail": {"message": exc.message, "hint": exc.hint}})


def _dispatch(task: str):
    from ..workers.celery_app import dispatch
    return dispatch(task)


# ======================================================================================
# system
# ======================================================================================

system = APIRouter(prefix="/api/system", tags=["system"])


@system.get("/health")
async def health(session: Session = Depends(db, scope="function")) -> dict[str, Any]:
    """Each dependency on its own line, so 'training is down' never reads as 'all is down'."""
    from ..workers.celery_app import redis_ok, worker_ok

    studio = get_studio()
    checks: dict[str, dict[str, Any]] = {"backend": {"ok": True}}
    checks["database"] = {"ok": ping(studio.settings.database_url)}
    redis_up = await asyncio.to_thread(redis_ok)
    checks["redis"] = {"ok": redis_up}
    checks["worker"] = {"ok": await asyncio.to_thread(worker_ok) if redis_up else False}
    service = ModelConnectionService(studio, session)
    default = service.default()
    if default is None:
        checks["llm"] = {"ok": False, "detail": "还没有默认对话模型"}
    else:
        provider = service.provider_for(default)
        try:
            result = await provider.check()
        finally:
            await provider.aclose()
        checks["llm"] = {"ok": bool(result.get("ok")), "detail": default.name}
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(studio.settings.gpt_sovits_url + "/docs")
        checks["voice_engine"] = {"ok": response.status_code == 200}
    except httpx.HTTPError:
        checks["voice_engine"] = {"ok": False}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


@system.get("/overview")
def overview(session: Session = Depends(db, scope="function")) -> dict[str, int]:
    """Counts for the first-run guide."""
    studio = get_studio()
    return {
        "model_connections": len(ModelConnectionService(studio, session).list()),
        "voice_packs": len(VoicePackService(studio, session).list()),
        "voice_models": len(VoiceModelService(studio, session).list()),
        "characters": len(CharacterService(studio, session).list()),
    }


files = APIRouter(prefix="/api/files", tags=["files"])


@files.get("/{key:path}")
def get_file(key: str) -> FileResponse:
    if not key.startswith(FILE_PREFIXES):
        raise NotFound("文件不存在")
    try:
        path = get_studio().storage.local_path(key)
    except StorageError as exc:
        raise NotFound("文件不存在") from exc
    if not path.is_file():
        raise NotFound("文件不存在")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")


# ======================================================================================
# model connections
# ======================================================================================

connections = APIRouter(prefix="/api/model-connections", tags=["model connections"])


def _connections(session: Session) -> ModelConnectionService:
    return ModelConnectionService(get_studio(), session)


@connections.get("")
def list_connections(session: Session = Depends(db, scope="function")) -> list[ModelConnectionOut]:
    service = _connections(session)
    default = service.default_id()
    return [ModelConnectionOut.of(c, default) for c in service.list()]


@connections.post("", status_code=201)
def create_connection(draft: ConnectionDraft, session: Session = Depends(db, scope="function")) -> ModelConnectionOut:
    service = _connections(session)
    connection = service.create(draft)
    return ModelConnectionOut.of(connection, service.default_id())


@connections.post("/test")
async def test_draft(draft: ConnectionDraft, session: Session = Depends(db, scope="function")) -> dict[str, Any]:
    """Test settings before saving (the wizard's button)."""
    return await _connections(session).probe(draft)


class ModelListQuery(BaseModel):
    provider_type: ProviderType
    base_url: str
    api_key: Optional[str] = None
    connection_id: Optional[str] = None     # reuse a stored key without re-entering it


@connections.post("/models")
async def discover_models(query: ModelListQuery, session: Session = Depends(db, scope="function")) -> dict[str, list[str]]:
    service = _connections(session)
    key = query.api_key
    if not key and query.connection_id:
        key = service.api_key(service.get(query.connection_id))
    return {"models": await service.list_models(query.provider_type, query.base_url, key)}


@connections.get("/{connection_id}")
def get_connection(connection_id: str, session: Session = Depends(db, scope="function")) -> ModelConnectionOut:
    service = _connections(session)
    return ModelConnectionOut.of(service.get(connection_id), service.default_id())


@connections.patch("/{connection_id}")
def update_connection(connection_id: str, patch: ConnectionPatch,
                      session: Session = Depends(db, scope="function")) -> ModelConnectionOut:
    service = _connections(session)
    return ModelConnectionOut.of(service.update(connection_id, patch), service.default_id())


@connections.delete("/{connection_id}", status_code=204)
def delete_connection(connection_id: str, session: Session = Depends(db, scope="function")) -> None:
    _connections(session).delete(connection_id)


@connections.post("/{connection_id}/test")
async def test_connection(connection_id: str, session: Session = Depends(db, scope="function")) -> ModelConnectionOut:
    service = _connections(session)
    connection = await service.test(connection_id)
    return ModelConnectionOut.of(connection, service.default_id())


@connections.get("/{connection_id}/models")
async def connection_models(connection_id: str, session: Session = Depends(db, scope="function")) -> dict[str, list[str]]:
    service = _connections(session)
    c = service.get(connection_id)
    return {"models": await service.list_models(c.provider_type, c.base_url, service.api_key(c))}


@connections.post("/{connection_id}/default")
def make_default(connection_id: str, session: Session = Depends(db, scope="function")) -> ModelConnectionOut:
    service = _connections(session)
    service.set_default(connection_id)
    return ModelConnectionOut.of(service.get(connection_id), connection_id)


# ======================================================================================
# voice packs
# ======================================================================================

packs = APIRouter(prefix="/api/voice-packs", tags=["voice packs"])


def _packs(session: Session) -> VoicePackService:
    return VoicePackService(get_studio(), session)


def _pack_out(service: VoicePackService, pack: VoicePack) -> VoicePackOut:
    return VoicePackOut.of(pack, service.summary(pack))


@packs.get("")
def list_packs(session: Session = Depends(db, scope="function")) -> list[VoicePackOut]:
    service = _packs(session)
    return [_pack_out(service, p) for p in service.list()]


@packs.post("", status_code=201)
def create_pack(data: VoicePackCreate, session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    return _pack_out(service, service.create(data))


@packs.get("/{pack_id}")
def get_pack(pack_id: str, session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    return _pack_out(service, service.get(pack_id))


@packs.patch("/{pack_id}")
def update_pack(pack_id: str, data: dict[str, Any], session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    return _pack_out(service, service.update(pack_id, data))


@packs.delete("/{pack_id}", status_code=204)
def delete_pack(pack_id: str, session: Session = Depends(db, scope="function")) -> None:
    _packs(session).delete(pack_id)


@packs.post("/{pack_id}/upload")
def upload(pack_id: str, files: list[UploadFile] = File(...),
           session: Session = Depends(db, scope="function")) -> dict[str, Any]:
    service = _packs(session)
    accepted, rejected = [], []
    for upload_file in files:
        result = service.add_file(pack_id, upload_file.filename or "file", upload_file.file)
        accepted += result.accepted
        rejected += result.rejected
    return {"accepted": accepted, "rejected": rejected,
            "pack": _pack_out(service, service.get(pack_id)).model_dump(mode="json")}


@packs.post("/{pack_id}/rights")
def confirm_rights(pack_id: str, session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    return _pack_out(service, service.confirm_rights(pack_id))


@packs.post("/{pack_id}/prepare", status_code=202)
def prepare(pack_id: str, session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    pack = service.mark_processing(pack_id)
    session.commit()          # the worker must see PROCESSING before it starts
    try:
        _dispatch("prepare_voice_pack")(pack_id)
    except Exception as exc:  # noqa: BLE001
        raise Unavailable("处理服务不可用", hint="请确认 Memurai（Redis）和训练进程已启动") from exc
    return _pack_out(service, pack)


@packs.get("/{pack_id}/samples")
def list_samples(pack_id: str, session: Session = Depends(db, scope="function")) -> list[AudioSampleOut]:
    return [AudioSampleOut.of(s) for s in _packs(session).get(pack_id).samples]


@packs.post("/{pack_id}/approve")
def approve_dataset(pack_id: str, session: Session = Depends(db, scope="function")) -> VoicePackOut:
    service = _packs(session)
    return _pack_out(service, service.approve_dataset(pack_id))


samples = APIRouter(prefix="/api/audio-samples", tags=["voice packs"])


@samples.patch("/{sample_id}")
def update_sample(sample_id: str, patch: SamplePatch, session: Session = Depends(db, scope="function")) -> AudioSampleOut:
    return AudioSampleOut.of(_packs(session).update_sample(sample_id, patch))


# ======================================================================================
# training
# ======================================================================================

training = APIRouter(prefix="/api/training-jobs", tags=["training"])


def _training(session: Session) -> TrainingService:
    return TrainingService(get_studio(), session, dispatch=_dispatch("run_training"))


def _job_out(session: Session, job: TrainingJob) -> TrainingJobOut:
    pack = session.get(VoicePack, job.voice_pack_id)
    return TrainingJobOut.of(job, pack.name if pack else None)


@training.get("/options")
def training_options(session: Session = Depends(db, scope="function")) -> list[dict]:
    return _training(session).options()


@training.get("")
def list_jobs(voice_pack_id: Optional[str] = None, session: Session = Depends(db, scope="function")) -> list[TrainingJobOut]:
    return [_job_out(session, j) for j in _training(session).list(voice_pack_id)]


@training.post("", status_code=202)
def create_job(request: TrainingRequest, session: Session = Depends(db, scope="function")) -> TrainingJobOut:
    return _job_out(session, _training(session).create(request))


@training.get("/{job_id}")
def get_job(job_id: str, session: Session = Depends(db, scope="function")) -> TrainingJobOut:
    return _job_out(session, _training(session).get(job_id))


@training.post("/{job_id}/cancel")
def cancel_job(job_id: str, session: Session = Depends(db, scope="function")) -> TrainingJobOut:
    return _job_out(session, _training(session).cancel(job_id))


class RetryRequest(BaseModel):
    preset: Optional[str] = None
    advanced: Optional[dict[str, float]] = None


@training.post("/{job_id}/retry", status_code=202)
def retry_job(job_id: str, body: RetryRequest, session: Session = Depends(db, scope="function")) -> TrainingJobOut:
    return _job_out(session, _training(session).retry(job_id, body.preset, body.advanced))


@training.get("/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events: a ``job`` snapshot whenever it changes, until it finishes."""
    studio = get_studio()
    with studio.db() as session:
        _training(session).get(job_id)              # 404 before streaming starts

    async def stream():
        last = None
        sent_at = time.monotonic()
        while not await request.is_disconnected():
            with studio.db() as session:
                job = session.get(TrainingJob, job_id)
                payload = _job_out(session, job).model_dump_json()
                terminal = job.status in TRAINING_TERMINAL
            if payload != last:
                last, sent_at = payload, time.monotonic()
                yield f"event: job\ndata: {payload}\n\n"
            elif time.monotonic() - sent_at > 15:
                sent_at = time.monotonic()
                yield ": keep-alive\n\n"
            if terminal:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ======================================================================================
# voice models
# ======================================================================================

voices = APIRouter(prefix="/api/voice-models", tags=["voice models"])


@voices.get("")
def list_voices(include_archived: bool = False, session: Session = Depends(db, scope="function")) -> list[VoiceModelOut]:
    return [VoiceModelOut.of(m) for m in VoiceModelService(get_studio(), session).list(include_archived)]


@voices.get("/test-sentences")
def test_sentences() -> list[dict]:
    return TEST_SENTENCES


@voices.get("/{model_id}")
def get_voice(model_id: str, session: Session = Depends(db, scope="function")) -> VoiceModelOut:
    return VoiceModelOut.of(VoiceModelService(get_studio(), session).get(model_id))


@voices.patch("/{model_id}")
def update_voice(model_id: str, data: dict[str, Any], session: Session = Depends(db, scope="function")) -> VoiceModelOut:
    return VoiceModelOut.of(VoiceModelService(get_studio(), session).update(model_id, data))


@voices.delete("/{model_id}", status_code=204)
def delete_voice(model_id: str, session: Session = Depends(db, scope="function")) -> None:
    VoiceModelService(get_studio(), session).delete(model_id)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    style: str = "neutral"


@voices.post("/{model_id}/synthesize")
async def synthesize(model_id: str, body: SynthesizeRequest, session: Session = Depends(db, scope="function")) -> dict[str, Any]:
    service = VoiceModelService(get_studio(), session)
    result = await asyncio.to_thread(service.synthesize, model_id, body.text, body.style)
    result["audio_url"] = file_url(result.pop("audio_key"))
    return result


# ======================================================================================
# characters
# ======================================================================================

characters = APIRouter(prefix="/api/characters", tags=["characters"])


def character_out(service: CharacterService, c) -> CharacterOut:
    from ..db.models import VoiceModel

    connection = service.connection_for(c)
    voice = service.db.get(VoiceModel, c.voice_model_id) if c.voice_model_id else None
    return CharacterOut(
        id=c.id, name=c.name, description=c.description, avatar_url=file_url(c.avatar_key),
        system_prompt=c.system_prompt, greeting=c.greeting,
        model_connection_id=c.model_connection_id, uses_default_model=c.model_connection_id is None,
        model_label=connection.name if connection else None, voice_model_id=c.voice_model_id,
        voice_label=voice.name if voice else None, default_voice_style=c.default_voice_style,
        status=c.status.value, problems=service.problems(c),
        created_at=c.created_at, updated_at=c.updated_at)


@characters.get("")
def list_characters(session: Session = Depends(db, scope="function")) -> list[CharacterOut]:
    service = CharacterService(get_studio(), session)
    return [character_out(service, c) for c in service.list()]


@characters.post("", status_code=201)
def create_character(data: CharacterData, session: Session = Depends(db, scope="function")) -> CharacterOut:
    service = CharacterService(get_studio(), session)
    return character_out(service, service.create(data))


@characters.get("/{character_id}")
def get_character(character_id: str, session: Session = Depends(db, scope="function")) -> CharacterOut:
    service = CharacterService(get_studio(), session)
    return character_out(service, service.get(character_id))


@characters.patch("/{character_id}")
def update_character(character_id: str, patch: CharacterPatch, session: Session = Depends(db, scope="function")) -> CharacterOut:
    service = CharacterService(get_studio(), session)
    return character_out(service, service.update(character_id, patch))


@characters.delete("/{character_id}", status_code=204)
def delete_character(character_id: str, session: Session = Depends(db, scope="function")) -> None:
    CharacterService(get_studio(), session).delete(character_id)


@characters.post("/{character_id}/avatar")
async def upload_avatar(character_id: str, file: UploadFile = File(...),
                        session: Session = Depends(db, scope="function")) -> CharacterOut:
    service = CharacterService(get_studio(), session)
    data = await file.read(5 * 1024 * 1024 + 1)
    return character_out(service, service.set_avatar(character_id, data))


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=100)
    system_prompt: str = ""
    model_connection_id: Optional[str] = None
    voice_model_id: Optional[str] = None
    voice_style: str = "neutral"


@characters.post("/preview")
async def preview(body: PreviewRequest, session: Session = Depends(db, scope="function")) -> dict[str, Any]:
    """The builder's Preview step: one reply from the chosen brain, in the chosen voice."""
    from cvai_types import LLMMessage, Role

    studio = get_studio()
    connections_service = ModelConnectionService(studio, session)
    connection = (connections_service.get(body.model_connection_id) if body.model_connection_id
                  else connections_service.default())
    if connection is None:
        raise ServiceError("还没有可用的对话模型", hint="请先在「模型」页面连接一个模型")
    provider = connections_service.provider_for(connection)
    system_prompt = (f"你是{body.name}。请始终以{body.name}的身份说话。\n\n{body.system_prompt}\n\n"
                     "用中文回答，不超过三句话，不使用表情符号。")
    started = time.perf_counter()
    try:
        response = await provider.complete([LLMMessage(role=Role.SYSTEM, content=system_prompt),
                                            LLMMessage(role=Role.USER, content=body.text)])
    except Exception as exc:  # noqa: BLE001
        from ..services.model_connections import friendly_error
        message, hint = friendly_error(str(exc), connection.provider_type, connection.base_url,
                                       connection.model_name)
        raise ServiceError(message, hint=hint) from exc
    finally:
        await provider.aclose()
    result: dict[str, Any] = {"reply": response.content.strip(),
                              "llm_ms": round((time.perf_counter() - started) * 1000)}
    if body.voice_model_id and result["reply"]:
        voice_service = VoiceModelService(studio, session)
        audio = await asyncio.to_thread(voice_service.synthesize, body.voice_model_id,
                                        result["reply"][:300], body.voice_style)
        result["audio_url"] = file_url(audio["audio_key"])
        result["tts_ms"] = audio["generation_ms"]
    return result


ROUTERS = [system, files, connections, packs, samples, training, voices, characters]
