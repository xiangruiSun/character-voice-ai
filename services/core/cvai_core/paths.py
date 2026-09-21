"""Filesystem layout helpers.

Every path in the project resolves through here, so relocating the repo or pointing the
voice pack root at a big disk is a config change rather than a code change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from cvai_types import (
    CHECKPOINTS_DIR,
    CLEAN_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REJECTED_DIR,
    TRANSCRIPTS_DIR,
    DATASET_FILENAME,
    DATASETS_DIR,
    EVALUATION_DIR,
    MANIFEST_FILENAME,
    METADATA_DIR,
    REFERENCES_DIR,
    REFERENCES_FILENAME,
    VOICEPACK_DIRS,
)

from .errors import ConfigError

#: Marker files that identify the repository root when walking upwards.
_ROOT_MARKERS: Final = ("pyproject.toml", ".git")

#: Environment variable that overrides root discovery, for containers where the repo is
#: mounted somewhere unusual.
ENV_REPO_ROOT: Final = "CVAI_REPO_ROOT"


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upwards from ``start``."""
    override = os.environ.get(ENV_REPO_ROOT)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise ConfigError(f"{ENV_REPO_ROOT} points at {path}, which is not a directory")
        return path

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if candidate.is_dir() and any((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    raise ConfigError(
        "could not locate the repository root; set CVAI_REPO_ROOT or run from inside the repo"
    )


class VoicePackPaths:
    """Resolved paths for one voice pack.

    Read-only convenience over the layout constants in ``cvai_types.voicepack``; it does
    not create anything (the scaffolder does that) so importing it is always safe.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    @property
    def raw(self) -> Path:
        return self.root / RAW_DIR

    @property
    def processed(self) -> Path:
        return self.root / PROCESSED_DIR

    @property
    def clean(self) -> Path:
        return self.root / CLEAN_DIR

    @property
    def rejected(self) -> Path:
        return self.root / REJECTED_DIR

    @property
    def transcripts(self) -> Path:
        return self.root / TRANSCRIPTS_DIR

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_DIR

    @property
    def references(self) -> Path:
        return self.root / REFERENCES_DIR

    @property
    def datasets(self) -> Path:
        return self.root / DATASETS_DIR

    @property
    def checkpoints(self) -> Path:
        return self.root / CHECKPOINTS_DIR

    @property
    def evaluation(self) -> Path:
        return self.root / EVALUATION_DIR

    @property
    def manifest_file(self) -> Path:
        return self.metadata / MANIFEST_FILENAME

    @property
    def dataset_file(self) -> Path:
        return self.metadata / DATASET_FILENAME

    @property
    def references_file(self) -> Path:
        return self.metadata / REFERENCES_FILENAME

    def all_dirs(self) -> list[Path]:
        return [self.root / name for name in VOICEPACK_DIRS]

    def resolve(self, relative: str) -> Path:
        """Resolve a pack-relative manifest path, refusing to escape the pack."""
        target = (self.root / relative).resolve()
        if not str(target).startswith(str(self.root)):
            raise ConfigError(f"path {relative!r} escapes the voice pack root")
        return target

    def relativize(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VoicePackPaths({self.root})"


def voicepack_root(voicepack_id: str, packs_root: Path | None = None) -> VoicePackPaths:
    base = packs_root or (repo_root() / "voicepacks")
    return VoicePackPaths(Path(base) / voicepack_id)


def character_profile_path(character_id: str, root: Path | None = None) -> Path:
    base = root or (repo_root() / "characters" / "profiles")
    return Path(base) / f"{character_id}.yaml"
