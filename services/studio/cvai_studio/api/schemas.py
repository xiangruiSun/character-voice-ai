"""What the API returns. Explicit models: nothing secret or internal leaks by accident.

Notably absent: decrypted API keys (only ``api_key_hint``), absolute paths (files are
``/api/files/<key>`` URLs), and raw tracebacks outside a job's ``error_detail``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel


def file_url(key: str | None) -> str | None:
    return f"/api/files/{quote(key)}" if key else None


class ModelConnectionOut(BaseModel):
    id: str
    name: str
    kind: str
    provider_type: str
    connection_mode: str
    base_url: str
    model_name: str
    has_api_key: bool
    api_key_hint: str | None
    status: str
    last_error: str | None
    latency_ms: float | None
    last_tested_at: datetime | None
    is_default: bool
    generation_defaults: dict[str, Any]
    extra: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, c, default_id: str | None) -> "ModelConnectionOut":
        return cls(id=c.id, name=c.name, kind=c.kind.value, provider_type=c.provider_type.value,
                   connection_mode=c.connection_mode.value, base_url=c.base_url,
                   model_name=c.model_name, has_api_key=bool(c.encrypted_api_key),
                   api_key_hint=c.api_key_hint, status=c.status.value, last_error=c.last_error,
                   latency_ms=c.latency_ms, last_tested_at=c.last_tested_at,
                   is_default=c.id == default_id, generation_defaults=c.generation_defaults or {},
                   extra=c.extra or {}, created_at=c.created_at, updated_at=c.updated_at)


class AudioSampleOut(BaseModel):
    id: str
    audio_url: str
    source_file: str
    transcript: str
    transcript_source: str
    duration_s: float
    quality_score: float | None
    issues: list[str]
    emotion: str | None
    style: str
    approved: bool
    rejection_reason: str | None
    human_edited: bool

    @classmethod
    def of(cls, s) -> "AudioSampleOut":
        return cls(id=s.id, audio_url=file_url(s.audio_key), source_file=s.source_file,
                   transcript=s.transcript, transcript_source=s.transcript_source,
                   duration_s=round(s.duration_s, 2), quality_score=s.quality_score,
                   issues=list(s.issues or []), emotion=s.emotion, style=s.style,
                   approved=s.approved, rejection_reason=s.rejection_reason,
                   human_edited=s.human_edited)


class VoicePackOut(BaseModel):
    id: str
    name: str
    character_name: str
    language: str
    description: str
    source: str
    notes: str
    status: str
    rights_confirmed: bool
    total_files: int
    total_duration_s: float
    usable_duration_s: float
    total_bytes: int
    rejected_files: list[dict]
    progress: dict[str, Any]
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    summary: dict[str, Any] | None = None

    @classmethod
    def of(cls, p, summary: dict | None = None) -> "VoicePackOut":
        return cls(id=p.id, name=p.name, character_name=p.character_name, language=p.language,
                   description=p.description, source=p.source, notes=p.notes, status=p.status.value,
                   rights_confirmed=p.rights_confirmed, total_files=p.total_files,
                   total_duration_s=round(p.total_duration_s, 1),
                   usable_duration_s=round(p.usable_duration_s, 1), total_bytes=p.total_bytes,
                   rejected_files=list(p.rejected_files or [])[-50:], progress=p.progress or {},
                   last_error=p.last_error, created_at=p.created_at, updated_at=p.updated_at,
                   summary=summary)


class TrainingJobOut(BaseModel):
    id: str
    voice_pack_id: str
    voice_pack_name: str | None = None
    engine: str
    base_model: str
    preset: str
    status: str
    current_stage: str
    progress: float
    current_epoch: int
    total_epochs: int
    config: dict[str, Any]
    events: list[dict]
    metrics: dict[str, Any]
    error_message: str | None
    error_hint: str | None
    error_detail: str | None
    voice_model_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def of(cls, j, pack_name: str | None = None) -> "TrainingJobOut":
        return cls(id=j.id, voice_pack_id=j.voice_pack_id, voice_pack_name=pack_name,
                   engine=j.engine, base_model=j.base_model, preset=j.preset, status=j.status.value,
                   current_stage=j.current_stage, progress=round(j.progress, 4),
                   current_epoch=j.current_epoch, total_epochs=j.total_epochs, config=j.config or {},
                   events=list(j.events or [])[-60:], metrics=j.metrics or {},
                   error_message=j.error_message, error_hint=j.error_hint,
                   error_detail=j.error_detail, voice_model_id=j.voice_model_id,
                   created_at=j.created_at, started_at=j.started_at, completed_at=j.completed_at)


class VoiceModelOut(BaseModel):
    id: str
    name: str
    voice_pack_id: str | None
    training_job_id: str | None
    engine: str
    base_model: str
    version: int
    status: str
    approved: bool
    styles: list[str]
    evaluation: list[dict]
    created_at: datetime

    @classmethod
    def of(cls, m) -> "VoiceModelOut":
        evaluation = [{"text": s["text"], "audio_url": file_url(s["audio_key"])}
                      for s in (m.evaluation_metadata or {}).get("samples", [])]
        return cls(id=m.id, name=m.name, voice_pack_id=m.voice_pack_id,
                   training_job_id=m.training_job_id, engine=m.engine, base_model=m.base_model,
                   version=m.version, status=m.status.value, approved=m.approved,
                   styles=list(m.styles or []), evaluation=evaluation, created_at=m.created_at)


class CharacterOut(BaseModel):
    id: str
    name: str
    description: str
    avatar_url: str | None
    system_prompt: str
    greeting: str
    model_connection_id: str | None
    uses_default_model: bool
    model_label: str | None
    voice_model_id: str | None
    voice_label: str | None
    default_voice_style: str
    status: str
    problems: list[str]
    created_at: datetime
    updated_at: datetime
