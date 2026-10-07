"""The training worker: one Celery process, one GPU, jobs run one at a time.

    cvai-studio-worker                     (or: python -m cvai_studio.workers.celery_app)

Celery is not officially supported on Windows; its single-process ``solo`` pool works
there and is the right shape anyway — one GPU, one job at a time.
"""

from __future__ import annotations

import logging
import os
import sys

from celery import Celery
from celery.signals import worker_ready

from ..core.config import get_settings

settings = get_settings()
app = Celery("cvai_studio", broker=settings.redis_url, include=["cvai_studio.workers.tasks"])
app.conf.update(
    task_default_queue="studio",
    task_acks_late=False,             # a job is not re-run after a crash; it is marked failed
    worker_prefetch_multiplier=1,
    task_ignore_result=True,          # state lives in the database, not in a result backend
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=2,
    broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 5},
    worker_hijack_root_logger=False,
)


def redis_ok(timeout: float = 1.0) -> bool:
    import redis

    try:
        return bool(redis.Redis.from_url(settings.redis_url, socket_connect_timeout=timeout,
                                         socket_timeout=timeout).ping())
    except Exception:  # noqa: BLE001
        return False


def worker_ok(timeout: float = 1.0) -> bool:
    if not redis_ok():
        return False
    try:
        return bool(app.control.ping(timeout=timeout))
    except Exception:  # noqa: BLE001
        return False


def dispatch(task_name: str):
    """A callable that queues ``task_name(entity_id)`` or raises if the queue is down."""
    def send(entity_id: str) -> str:
        if not redis_ok():
            raise ConnectionError(f"Redis is not reachable at {settings.redis_url}")
        result = app.send_task(f"cvai_studio.workers.tasks.{task_name}", args=[entity_id],
                               queue="studio", retry=False)
        return result.id
    return send


@worker_ready.connect
def _recover(**_):  # pragma: no cover - runs inside the worker
    from ..services.context import get_studio
    from ..services.training import recover_orphaned_jobs

    count = recover_orphaned_jobs(get_studio())
    if count:
        logging.getLogger(__name__).warning("marked %d interrupted training job(s) as failed", count)


def main() -> int:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    argv = ["worker", "--pool=solo", "-Q", "studio", "--loglevel=INFO", "-n", "studio@%h"]
    app.worker_main(argv + sys.argv[1:])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
