"""Celery tasks. Thin: each calls a service and records failure on the entity."""

from __future__ import annotations

from ..core.logging import get_logger, with_ids
from ..services.context import get_studio
from ..services.training import JobRunner
from ..services.voice_packs import VoicePackService
from .celery_app import app

log = get_logger(__name__)


@app.task(name="cvai_studio.workers.tasks.prepare_voice_pack")
def prepare_voice_pack(pack_id: str) -> None:
    studio = get_studio()
    tagged = with_ids(log, voice_pack_id=pack_id)

    def report(pack_id: str, progress: dict) -> None:
        # Progress goes out in its own transaction so the UI sees each stage.
        from ..db.models import VoicePack

        with studio.db() as db:
            db.get(VoicePack, pack_id).progress = progress

    try:
        with studio.db() as db:
            VoicePackService(studio, db).prepare(pack_id, progress=report)
        tagged.info("prepared")
    except Exception as exc:  # noqa: BLE001
        tagged.exception("preparation failed")
        with studio.db() as db:
            VoicePackService(studio, db).mark_failed(pack_id, f"{type(exc).__name__}: {exc}")


@app.task(name="cvai_studio.workers.tasks.run_training")
def run_training(job_id: str) -> None:
    studio = get_studio()
    try:
        JobRunner(studio, job_id).run()
    except Exception as exc:  # noqa: BLE001 - never leave a job stuck in a running state
        with_ids(log, training_job_id=job_id).exception("training task crashed")
        from ..db.models import TrainingJob
        from ..domain.enums import TRAINING_TERMINAL, TrainingStatus

        with studio.db() as db:
            job = db.get(TrainingJob, job_id)
            if job is not None and job.status not in TRAINING_TERMINAL:
                job.status = TrainingStatus.FAILED
                job.error_message = "训练进程出错"
                job.error_hint = "请展开高级详情查看原因，然后重试"
                job.error_detail = f"{type(exc).__name__}: {exc}"
