"""SQLAlchemy models. Metadata only — audio and checkpoints live in storage, by key.

Portable across SQLite (local) and PostgreSQL (hosted): string ids, enums stored as
plain strings, JSON for open-ended metadata, timezone-aware timestamps.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..domain.enums import (
    CharacterStatus,
    ConnectionKind,
    ConnectionMode,
    ConnectionStatus,
    ProviderType,
    TrainingStatus,
    VoiceModelStatus,
    VoicePackStatus,
)


def new_id(prefix: str) -> str:
    # Starts with a letter and stays lowercase: these double as voice pack / character
    # slugs for the conversation pipeline, whose ids must match ^[a-z][a-z0-9_]*$.
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum(enum_cls):
    return SAEnum(enum_cls, native_enum=False, length=32,
                  values_callable=lambda e: [m.value for m in e], validate_strings=True)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)


class AppSetting(Base):
    """Small key/value settings, e.g. ``default_model_connection_id``."""

    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)


class ModelConnection(Timestamped, Base):
    __tablename__ = "model_connections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("mc"))
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[ConnectionKind] = mapped_column(_enum(ConnectionKind))
    provider_type: Mapped[ProviderType] = mapped_column(_enum(ProviderType))
    connection_mode: Mapped[ConnectionMode] = mapped_column(
        _enum(ConnectionMode), default=ConnectionMode.BACKEND_PROXY)
    base_url: Mapped[str] = mapped_column(String(500))
    model_name: Mapped[str] = mapped_column(String(200))
    encrypted_api_key: Mapped[str | None] = mapped_column(Text)
    api_key_hint: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[ConnectionStatus] = mapped_column(_enum(ConnectionStatus),
                                                     default=ConnectionStatus.DRAFT)
    last_error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    capabilities: Mapped[dict[str, Any]] = mapped_column(default=dict)
    #: temperature, max_tokens, stream, think, ...
    generation_defaults: Mapped[dict[str, Any]] = mapped_column(default=dict)
    #: provider-specific extras (organization, project, ...)
    extra: Mapped[dict[str, Any]] = mapped_column(default=dict)


class VoicePack(Timestamped, Base):
    __tablename__ = "voice_packs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("vp"))
    name: Mapped[str] = mapped_column(String(100))
    character_name: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(16), default="zh-CN")
    description: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[VoicePackStatus] = mapped_column(_enum(VoicePackStatus),
                                                    default=VoicePackStatus.EMPTY)
    #: Storage prefix of the pack directory (raw/, processed/, clean/, metadata/ …).
    storage_prefix: Mapped[str] = mapped_column(String(200))
    rights_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    rights_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    usable_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    total_bytes: Mapped[int] = mapped_column(Integer, default=0)
    #: [{"name": "...", "reason": "..."}] — files refused at upload.
    rejected_files: Mapped[list[Any]] = mapped_column(default=list)
    #: Preparation progress while PROCESSING: {"stage": ..., "done": n, "total": n}.
    progress: Mapped[dict[str, Any]] = mapped_column(default=dict)
    last_error: Mapped[str | None] = mapped_column(Text)

    samples: Mapped[list["AudioSample"]] = relationship(
        back_populates="voice_pack", cascade="all, delete-orphan", order_by="AudioSample.position")


class AudioSample(Timestamped, Base):
    __tablename__ = "audio_samples"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("as"))
    voice_pack_id: Mapped[str] = mapped_column(ForeignKey("voice_packs.id", ondelete="CASCADE"),
                                               index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    audio_key: Mapped[str] = mapped_column(String(300))
    source_file: Mapped[str] = mapped_column(String(300), default="")
    transcript: Mapped[str] = mapped_column(Text, default="")
    #: "provided" (uploaded .lab/.txt), an ASR model id, or "human"
    transcript_source: Mapped[str] = mapped_column(String(64), default="")
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float | None] = mapped_column(Float)
    #: Automatic findings are hints ("potential issue"), never verdicts.
    issues: Mapped[list[Any]] = mapped_column(default=list)
    emotion: Mapped[str | None] = mapped_column(String(32))
    style: Mapped[str] = mapped_column(String(32), default="neutral")
    approved: Mapped[bool] = mapped_column(Boolean, default=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(200))
    human_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    voice_pack: Mapped[VoicePack] = relationship(back_populates="samples")


class TrainingJob(Timestamped, Base):
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("tj"))
    voice_pack_id: Mapped[str] = mapped_column(ForeignKey("voice_packs.id"), index=True)
    engine: Mapped[str] = mapped_column(String(32))
    base_model: Mapped[str] = mapped_column(String(64))
    preset: Mapped[str] = mapped_column(String(32), default="balanced")
    status: Mapped[TrainingStatus] = mapped_column(_enum(TrainingStatus),
                                                   default=TrainingStatus.QUEUED)
    current_stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    #: Structured events so far: [{"type": "epoch_completed", "at": ..., ...}]
    events: Mapped[list[Any]] = mapped_column(default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    output_prefix: Mapped[str | None] = mapped_column(String(200))
    log_key: Mapped[str | None] = mapped_column(String(200))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_hint: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(64))
    voice_model_id: Mapped[str | None] = mapped_column(String(32))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VoiceModel(Timestamped, Base):
    __tablename__ = "voice_models"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("vm"))
    name: Mapped[str] = mapped_column(String(100))
    voice_pack_id: Mapped[str | None] = mapped_column(ForeignKey("voice_packs.id"))
    training_job_id: Mapped[str | None] = mapped_column(String(32))
    engine: Mapped[str] = mapped_column(String(32))
    base_model: Mapped[str] = mapped_column(String(64))
    #: Engine-specific checkpoint keys, e.g. {"gpt": "voice_models/vm_x/checkpoints/a.ckpt",
    #: "sovits": "…/b.pth"}. Several versions of one pack are several VoiceModels.
    checkpoint: Mapped[dict[str, Any]] = mapped_column(default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[VoiceModelStatus] = mapped_column(_enum(VoiceModelStatus),
                                                     default=VoiceModelStatus.READY)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    #: Storage prefix of a voice-pack-format directory holding the reference bank.
    reference_prefix: Mapped[str] = mapped_column(String(200))
    styles: Mapped[list[Any]] = mapped_column(default=list)
    evaluation_metadata: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Character(Timestamped, Base):
    __tablename__ = "characters"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("ch"))
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    avatar_key: Mapped[str | None] = mapped_column(String(300))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    greeting: Mapped[str] = mapped_column(Text, default="")
    #: None → use the default model connection.
    model_connection_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_connections.id", ondelete="SET NULL"))
    voice_model_id: Mapped[str | None] = mapped_column(
        ForeignKey("voice_models.id", ondelete="SET NULL"))
    default_voice_style: Mapped[str] = mapped_column(String(32), default="neutral")
    #: A hand-written YAML profile this character extends (dialogue examples, lore).
    legacy_profile_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[CharacterStatus] = mapped_column(_enum(CharacterStatus),
                                                    default=CharacterStatus.READY)
