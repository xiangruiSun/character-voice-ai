"""Studio settings, from the environment (and the repository's git-ignored ``.env``)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cvai_core.config import load_dotenv
from cvai_core.paths import repo_root
from pydantic import BaseModel, Field


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class StudioSettings(BaseModel):
    #: Where uploads, prepared datasets, checkpoints and the SQLite file live.
    data_dir: Path
    database_url: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    #: Encrypts stored API keys. Generated into ``data_dir`` on first use when unset.
    secret_key: str | None = None
    #: The product is self-hosted first: the backend talks to models on this machine
    #: and on the user's own network. A hosted deployment turns both off.
    self_hosted_mode: bool = True
    allow_private_model_endpoints: bool = True
    #: The GPT-SoVITS checkout used for training and synthesis.
    gpt_sovits_dir: Path
    gpt_sovits_url: str = "http://127.0.0.1:9880"
    max_upload_mb: int = Field(default=2048, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> StudioSettings:
    load_dotenv(repo_root() / ".env")
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or repo_root() / "data").resolve()
    return StudioSettings(
        data_dir=data_dir,
        database_url=os.environ.get("STUDIO_DATABASE_URL")
        or f"sqlite:///{(data_dir / 'studio.db').as_posix()}",
        redis_url=os.environ.get("STUDIO_REDIS_URL", "redis://127.0.0.1:6379/0"),
        secret_key=os.environ.get("STUDIO_SECRET_KEY") or None,
        self_hosted_mode=_flag("SELF_HOSTED_MODE", True),
        allow_private_model_endpoints=_flag("ALLOW_PRIVATE_MODEL_ENDPOINTS", True),
        gpt_sovits_dir=Path(
            os.environ.get("GPT_SOVITS_DIR") or repo_root() / "models" / "gpt_sovits" / "src"
        ),
        gpt_sovits_url=os.environ.get("GPT_SOVITS_URL", "http://127.0.0.1:9880"),
        max_upload_mb=int(os.environ.get("STUDIO_MAX_UPLOAD_MB", "2048")),
    )
