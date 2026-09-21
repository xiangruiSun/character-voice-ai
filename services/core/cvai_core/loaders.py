"""Loading and saving the project's YAML/JSON artifacts.

All disk formats live here so the rest of the code deals in typed objects only. YAML for
hand-edited files (character profiles, voice pack manifests, configs), JSON for
machine-generated ones (datasets, reference banks, run records) — JSON because those get
large, are diffed by tooling rather than humans, and must round-trip exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from cvai_types import (
    CharacterProfile,
    CVAIModel,
    DatasetManifest,
    ReferenceBank,
    TestSentenceSet,
    VoicePackManifest,
)

from .errors import ConfigError, VoicePackError
from .interfaces.character import CharacterProvider
from .paths import VoicePackPaths, character_profile_path, repo_root, voicepack_root

M = TypeVar("M", bound=CVAIModel)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def load_model_yaml(model: type[M], path: Path) -> M:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{model.__name__} file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _validate(model, data, path)


def load_model_json(model: type[M], path: Path) -> M:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{model.__name__} file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _validate(model, data, path)


def _validate(model: type[M], data: Any, path: Path) -> M:
    try:
        return model.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"{path} is not a valid {model.__name__}:\n{exc}") from exc


def save_model_yaml(instance: CVAIModel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = instance.model_dump(mode="json", exclude_none=False)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload, handle, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    return path


def save_model_json(instance: CVAIModel, path: Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = instance.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    return path


# --------------------------------------------------------------------------------------
# Voice pack
# --------------------------------------------------------------------------------------


def load_voicepack_manifest(paths: VoicePackPaths) -> VoicePackManifest:
    return load_model_yaml(VoicePackManifest, paths.manifest_file)


def load_reference_bank(paths: VoicePackPaths) -> ReferenceBank:
    return load_model_json(ReferenceBank, paths.references_file)


def load_dataset_manifest(paths: VoicePackPaths) -> DatasetManifest:
    return load_model_json(DatasetManifest, paths.dataset_file)


def try_load_reference_bank(paths: VoicePackPaths) -> ReferenceBank | None:
    """Reference bank if present, otherwise ``None`` — a fresh pack has none yet."""
    return load_reference_bank(paths) if paths.references_file.is_file() else None


def try_load_dataset_manifest(paths: VoicePackPaths) -> DatasetManifest | None:
    return load_dataset_manifest(paths) if paths.dataset_file.is_file() else None


def open_voicepack(
    voicepack_id: str, packs_root: Path | None = None
) -> tuple[VoicePackPaths, VoicePackManifest]:
    paths = voicepack_root(voicepack_id, packs_root)
    if not paths.root.is_dir():
        raise VoicePackError(f"voice pack {voicepack_id!r} not found at {paths.root}")
    return paths, load_voicepack_manifest(paths)


# --------------------------------------------------------------------------------------
# Characters
# --------------------------------------------------------------------------------------


def load_character_profile(path: Path) -> CharacterProfile:
    return load_model_yaml(CharacterProfile, path)


class FilesystemCharacterProvider(CharacterProvider):
    """Reads ``characters/profiles/<id>.yaml`` (spec §3: local filesystem for the MVP).

    Profiles are cached after first load; call :meth:`reload` when editing them live.
    """

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self.profiles_dir = Path(
            profiles_dir or (repo_root() / "characters" / "profiles")
        )
        self._cache: dict[str, CharacterProfile] = {}

    def get(self, character_id: str) -> CharacterProfile:
        if character_id in self._cache:
            return self._cache[character_id]
        path = character_profile_path(character_id, self.profiles_dir)
        if not path.is_file():
            raise ConfigError(
                f"character profile {character_id!r} not found at {path}; "
                f"known: {self.list_ids()}"
            )
        profile = load_character_profile(path)
        if profile.character_id != character_id:
            raise ConfigError(
                f"{path} declares character_id={profile.character_id!r} "
                f"but is filed as {character_id!r}"
            )
        self._cache[character_id] = profile
        return profile

    def list_ids(self) -> list[str]:
        if not self.profiles_dir.is_dir():
            return []
        return sorted(p.stem for p in self.profiles_dir.glob("*.yaml"))

    def reload(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------------------
# Benchmark inputs
# --------------------------------------------------------------------------------------


def load_sentence_set(path: Path) -> TestSentenceSet:
    """Test sentences, in YAML because a human curates them."""
    return load_model_yaml(TestSentenceSet, path)
