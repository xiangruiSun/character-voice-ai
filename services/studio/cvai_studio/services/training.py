"""Training jobs: created by the API, executed by the worker, observed by the UI.

The API only ever creates/cancels jobs and reads their state. ``run`` executes in the
Celery worker, committing each state change in its own short transaction so the API
(and the browser, via SSE) sees progress as it happens.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger, with_ids
from ..db.models import TrainingJob, VoicePack
from ..domain.enums import (
    TRAINING_TERMINAL,
    TrainingStatus,
    VoicePackStatus,
    training_is_active,
    transition,
)
from ..providers.training.base import (
    TrainingCancelled,
    TrainingEvent,
    TrainingFailure,
    TrainingSample,
    VoiceTrainingProvider,
)
from ..providers.training.gpt_sovits import default_provider
from .context import Conflict, NotFound, ServiceError, Studio, Unavailable

log = get_logger(__name__)
MAX_EVENTS = 400
#: Where each stage sits on the overall progress bar.
STAGE_SPAN = {
    TrainingStatus.VALIDATING: (0.0, 0.02),
    TrainingStatus.PREPARING: (0.02, 0.10),
    TrainingStatus.TRAINING: (0.10, 0.92),
    TrainingStatus.EVALUATING: (0.92, 1.0),
}


class TrainingRequest(BaseModel):
    voice_pack_id: str
    engine: str = "gpt_sovits"
    preset: str = "balanced"
    advanced: dict[str, float] = Field(default_factory=dict)


def engines(studio: Studio) -> dict[str, VoiceTrainingProvider]:
    return {"gpt_sovits": default_provider(studio.settings)}


def now() -> datetime:
    return datetime.now(timezone.utc)


def add_event(job: TrainingJob, kind: str, **data: Any) -> None:
    event = {"type": kind, "at": now().isoformat(), **{k: v for k, v in data.items() if v is not None}}
    job.events = (list(job.events or []) + [event])[-MAX_EVENTS:]


class TrainingService:
    def __init__(self, studio: Studio, db: Session,
                 dispatch: Callable[[str], str] | None = None) -> None:
        self.studio = studio
        self.db = db
        self.dispatch = dispatch

    # -- queries -----------------------------------------------------------------------

    def get(self, job_id: str) -> TrainingJob:
        job = self.db.get(TrainingJob, job_id)
        if job is None:
            raise NotFound("训练任务不存在")
        return job

    def list(self, voice_pack_id: str | None = None) -> list[TrainingJob]:
        query = select(TrainingJob).order_by(TrainingJob.created_at.desc())
        if voice_pack_id:
            query = query.where(TrainingJob.voice_pack_id == voice_pack_id)
        return list(self.db.scalars(query))

    def options(self) -> list[dict]:
        out = []
        for key, engine in engines(self.studio).items():
            ok, reason = engine.available() if hasattr(engine, "available") else (True, "")
            out.append({
                "engine": key, "label": engine.label, "base_model": engine.base_model,
                "available": ok, "unavailable_reason": reason or None,
                "presets": [vars(p) for p in engine.presets()],
                "advanced": [vars(o) for o in engine.advanced_options()],
            })
        return out

    # -- commands (API) ------------------------------------------------------------------

    def create(self, request: TrainingRequest) -> TrainingJob:
        pack = self.db.get(VoicePack, request.voice_pack_id)
        if pack is None:
            raise NotFound("声音包不存在")
        if not pack.rights_confirmed:
            raise ServiceError("开始训练前，请先确认你有权使用这些音频")
        if pack.status is not VoicePackStatus.READY:
            raise ServiceError("请先完成数据审核并确认数据集", hint="在「审核」步骤点击「确认数据集」")
        running = [j for j in self.list(pack.id) if training_is_active(j.status)]
        if running:
            raise Conflict("这个声音包已有训练在进行中")
        engine = engines(self.studio).get(request.engine)
        if engine is None:
            raise ServiceError(f"未知的训练引擎：{request.engine}")
        try:
            config = engine.resolve_config(request.preset, request.advanced)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        job = TrainingJob(voice_pack_id=pack.id, engine=request.engine, base_model=engine.base_model,
                          preset=request.preset, config=config, status=TrainingStatus.QUEUED,
                          current_stage="queued",
                          total_epochs=int(config.get("sovits_epochs", 0)) + int(config.get("gpt_epochs", 0)))
        add_event(job, "job_queued")
        self.db.add(job)
        self.db.flush()
        self._dispatch(job)
        with_ids(log, training_job_id=job.id, voice_pack_id=pack.id).info("queued (%s)", request.preset)
        return job

    def _dispatch(self, job: TrainingJob) -> None:
        if self.dispatch is None:
            raise Unavailable("训练服务未配置")
        # Commit first: the worker can pick the task up within milliseconds, and must
        # find the job row when it does.
        self.db.commit()
        try:
            job.task_id = self.dispatch(job.id)
        except Exception as exc:  # noqa: BLE001 - broker down, worker missing, ...
            job.status = TrainingStatus.FAILED
            job.error_message = "训练服务不可用"
            job.error_hint = "请确认 Memurai（Redis）和训练进程已启动，然后重试"
            job.error_detail = str(exc)[:2000]
            add_event(job, "failed", message=job.error_message)
            self.db.commit()       # the request rolls back on the error below; keep this
            raise Unavailable(job.error_message, hint=job.error_hint) from exc

    def cancel(self, job_id: str) -> TrainingJob:
        job = self.get(job_id)
        if job.status in TRAINING_TERMINAL:
            return job
        if job.status is TrainingStatus.QUEUED:
            # Not picked up yet: the worker will see CANCELLED and skip it.
            job.status = TrainingStatus.CANCELLED
            job.completed_at = now()
            add_event(job, "cancelled")
        else:
            job.status = transition(job.status, TrainingStatus.CANCEL_REQUESTED)
            add_event(job, "cancel_requested")
        return job

    def retry(self, job_id: str, preset: str | None = None,
              advanced: dict[str, float] | None = None) -> TrainingJob:
        old = self.get(job_id)
        if old.status not in (TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            raise Conflict("只有失败或已取消的训练可以重试")
        return self.create(TrainingRequest(
            voice_pack_id=old.voice_pack_id, engine=old.engine, preset=preset or old.preset,
            advanced=advanced if advanced is not None else {
                k: v for k, v in old.config.items() if isinstance(v, (int, float))}))


# -- execution (worker) -------------------------------------------------------------------


class JobRunner:
    """Executes one job. Owns its own short DB sessions so progress is visible live."""

    def __init__(self, studio: Studio, job_id: str) -> None:
        self.studio = studio
        self.job_id = job_id
        self.log = with_ids(log, training_job_id=job_id)

    def _update(self, fn: Callable[[TrainingJob], None]) -> TrainingJob:
        with self.studio.db() as db:
            job = db.get(TrainingJob, self.job_id)
            fn(job)
            return job

    def _status(self) -> TrainingStatus | None:
        # A just-queued job may not be visible for a moment; wait briefly before giving up.
        for _ in range(20):
            with self.studio.db() as db:
                job = db.get(TrainingJob, self.job_id)
                if job is not None:
                    return job.status
            time.sleep(0.25)
        return None

    def _enter(self, status: TrainingStatus, event: str, label: str) -> None:
        def apply(job: TrainingJob) -> None:
            if job.status is TrainingStatus.CANCEL_REQUESTED:
                raise TrainingCancelled()
            job.status = transition(job.status, status)
            job.current_stage = label
            job.progress = STAGE_SPAN[status][0]
            add_event(job, event)
        self._update(apply)

    def run(self) -> None:
        from .voice_models import VoiceModelService

        status = self._status()
        if status is None:
            self.log.warning("job not found; nothing to run")
            return
        if status is not TrainingStatus.QUEUED:
            self.log.info("skipping: no longer queued")
            return
        self._update(lambda job: setattr(job, "started_at", now()))
        workdir = self.studio.storage.local_path(f"training/{self.job_id}")
        try:
            self._enter(TrainingStatus.VALIDATING, "dataset_validating", "正在检查数据")
            with self.studio.db() as db:
                job = db.get(TrainingJob, self.job_id)
                pack = db.get(VoicePack, job.voice_pack_id)
                engine = engines(self.studio)[job.engine]
                config = dict(job.config)
                samples = [TrainingSample(self.studio.storage.local_path(s.audio_key), s.transcript, s.style)
                           for s in pack.samples if s.approved and s.transcript.strip()]
            problems = engine.validate_dataset(samples)
            if problems:
                raise TrainingFailure("数据集不满足训练要求", hint="；".join(problems))

            self._enter(TrainingStatus.PREPARING, "dataset_preparing", "正在准备训练数据")
            dataset = engine.prepare_dataset(samples, workdir)

            self._enter(TrainingStatus.TRAINING, "training_started", "正在训练")
            result = engine.train(dataset, workdir, config, job_id=self.job_id,
                                  emit=self._on_event, cancelled=self._cancel_requested)

            self._enter(TrainingStatus.EVALUATING, "evaluation_started", "正在评估并生成试听")
            with self.studio.db() as db:
                job = db.get(TrainingJob, self.job_id)
                model = VoiceModelService(self.studio, db).create_from_training(job, result, engine)
                model_id = model.id

            def finish(job: TrainingJob) -> None:
                job.status = transition(job.status, TrainingStatus.COMPLETED)
                job.progress = 1.0
                job.current_stage = "训练完成"
                job.voice_model_id = model_id
                job.completed_at = now()
                add_event(job, "completed", voice_model_id=model_id)
            self._update(finish)
            shutil.rmtree(workdir / "dataset", ignore_errors=True)
            self.log.info("completed → %s", model_id)
        except TrainingCancelled:
            def cancelled(job: TrainingJob) -> None:
                job.status = TrainingStatus.CANCELLED
                job.current_stage = "已取消"
                job.completed_at = now()
                add_event(job, "cancelled")
            self._update(cancelled)
            self.log.info("cancelled")
        except TrainingFailure as exc:
            self._fail(exc.message, exc.hint, exc.detail)
        except Exception as exc:  # noqa: BLE001 - anything else is still a failed job
            import traceback

            self._fail("训练意外中止", "请展开高级详情查看原因；调整设置后可以重试",
                       traceback.format_exc()[-4000:])
            self.log.exception("crashed")

    def _fail(self, message: str, hint: str, detail: str) -> None:
        def failed(job: TrainingJob) -> None:
            job.status = TrainingStatus.FAILED
            job.error_message, job.error_hint, job.error_detail = message, hint, detail[-8000:]
            job.current_stage = "训练失败"
            job.completed_at = now()
            add_event(job, "failed", message=message)
        self._update(failed)
        self.log.warning("failed: %s", message)

    def _cancel_requested(self) -> bool:
        return self._status() is TrainingStatus.CANCEL_REQUESTED

    def _on_event(self, event: TrainingEvent) -> None:
        def apply(job: TrainingJob) -> None:
            if event.type == "epoch_completed":
                low, high = STAGE_SPAN[TrainingStatus.TRAINING]
                job.current_epoch = event.epoch or job.current_epoch
                job.total_epochs = event.total_epochs or job.total_epochs
                job.progress = low + (high - low) * (event.progress or 0)
                add_event(job, "epoch_completed", epoch=event.epoch, total_epochs=event.total_epochs,
                          phase=event.phase)
            elif event.type == "stage" and event.phase:
                job.current_stage = {"sovits": "正在训练音色", "gpt": "正在训练语调"}.get(event.phase, job.current_stage)
            elif event.type == "stage" and event.message and job.status is TrainingStatus.PREPARING:
                job.current_stage = "正在准备训练数据"
            elif event.type == "metric":
                metrics = dict(job.metrics or {})
                metrics.update({f"{event.phase}.{k}": v for k, v in event.data.items()})
                job.metrics = metrics
            elif event.type == "checkpoint_saved":
                add_event(job, "checkpoint_saved", phase=event.phase, **event.data)
        self._update(apply)


def recover_orphaned_jobs(studio: Studio) -> int:
    """A worker that died mid-job leaves it 'active' forever; mark those failed."""
    count = 0
    with studio.db() as db:
        for job in db.scalars(select(TrainingJob)):
            if training_is_active(job.status) and job.status is not TrainingStatus.QUEUED:
                job.status = TrainingStatus.FAILED
                job.error_message = "训练进程在任务进行中退出"
                job.error_hint = "重新启动训练进程后，可以重试这个任务"
                job.completed_at = now()
                add_event(job, "failed", message=job.error_message)
                count += 1
    return count
