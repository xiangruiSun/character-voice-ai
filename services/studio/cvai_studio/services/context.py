"""The Studio's shared dependencies, built once per process (API or worker)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from ..core.config import StudioSettings, get_settings
from ..core.security import FernetSecretStore, SecretStore
from ..db.session import migrate, session_factory, session_scope
from ..providers.storage import LocalFilesystemStorage, StorageProvider


class ServiceError(Exception):
    """A use case cannot proceed. ``status`` is the HTTP code the API maps it to;
    ``hint`` is the user-facing suggestion of what to do about it."""

    status = 400

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class NotFound(ServiceError):
    status = 404


class Conflict(ServiceError):
    status = 409


class Unavailable(ServiceError):
    status = 503


@dataclass
class Studio:
    settings: StudioSettings
    sessions: sessionmaker[Session]
    storage: StorageProvider
    secrets: SecretStore

    @contextmanager
    def db(self) -> Iterator[Session]:
        with session_scope(self.sessions) as session:
            yield session


def build_studio(settings: StudioSettings, *, run_migrations: bool = True) -> Studio:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if run_migrations:
        migrate(settings.database_url)
    return Studio(
        settings=settings,
        sessions=session_factory(settings.database_url),
        storage=LocalFilesystemStorage(settings.data_dir),
        secrets=FernetSecretStore.from_settings(settings.secret_key, settings.data_dir),
    )


@lru_cache(maxsize=1)
def get_studio() -> Studio:
    return build_studio(get_settings())
