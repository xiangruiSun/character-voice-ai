"""Run directories and reproducibility logging (spec §23).

Spec §23 lists what must be logged for a voice experiment to be reproducible: model
version, Voice Pack version, training configuration, reference sample, generation
parameters, generated output and evaluation result. This module owns the *where* — a
self-contained run directory — and the *when* — an append-only event log written as the
run proceeds, so a crashed run still leaves an auditable trail.

    runs/<run_id>/
        run.json        the BenchmarkRun manifest (rewritten as it grows)
        config.json     the exact merged config used
        env.json        interpreter, platform, git commit, package versions
        events.jsonl    append-only event log
        audio/          generated wav files, one subdirectory per candidate
        blind/          blind test set + key (key is written separately on purpose)
        report.md       human-readable summary
"""

from __future__ import annotations

import json
import os
import platform
import random
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvai_types import CVAIModel, utcnow

from .paths import repo_root


def new_run_id(prefix: str = "bench") -> str:
    """Sortable, unique, filesystem-safe run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}-{stamp}-{suffix}"


def git_commit(root: Path | None = None) -> str | None:
    """Current commit hash, or ``None`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or repo_root()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_is_dirty(root: Path | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes.

    Recorded in ``env.json``: a benchmark run from a dirty tree is not reproducible from
    its commit hash alone, and it is better to know that at read time than to discover it
    when a result cannot be recreated.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root or repo_root()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def environment_snapshot(root: Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": utcnow().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(root),
        "git_dirty": git_is_dirty(root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for package in ("pydantic", "yaml", "torch", "numpy", "soundfile", "librosa"):
        snapshot[f"version_{package}"] = _module_version(package)
    return snapshot


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


class RunPaths:
    """Directory layout for one run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def blind(self) -> Path:
        return self.root / "blind"

    @property
    def run_file(self) -> Path:
        return self.root / "run.json"

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    @property
    def env_file(self) -> Path:
        return self.root / "env.json"

    @property
    def events_file(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def report_file(self) -> Path:
        return self.root / "report.md"

    @property
    def ratings_file(self) -> Path:
        return self.root / "ratings.json"

    def candidate_audio_dir(self, candidate_id: str) -> Path:
        return self.audio / candidate_id

    def ensure(self) -> "RunPaths":
        for directory in (self.root, self.audio, self.blind):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def create_run(
    runs_root: Path,
    run_id: str | None = None,
    prefix: str = "bench",
) -> RunPaths:
    identifier = run_id or new_run_id(prefix)
    return RunPaths(Path(runs_root) / identifier).ensure()


class RunLogger:
    """Append-only JSONL event log.

    Append-only and flushed per event: a run that dies halfway through still explains
    what it had done, which matters when a run takes hours on a GPU box.
    """

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.paths.ensure()

    def event(self, kind: str, **fields: Any) -> None:
        payload = {"at": utcnow().isoformat(), "kind": kind, **fields}
        with self.paths.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def snapshot_environment(self, root: Path | None = None) -> dict[str, Any]:
        data = environment_snapshot(root)
        self.paths.env_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return data

    def snapshot_config(self, config: CVAIModel) -> None:
        self.paths.config_file.write_text(
            json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_run(self, run: CVAIModel) -> None:
        self.paths.run_file.write_text(
            json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
