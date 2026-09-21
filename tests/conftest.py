"""Shared fixtures.

The path bootstrap lets the suite run straight from a checkout without
``pip install -e .`` — useful in CI images and when a contributor has not installed the
project yet. ``pyproject.toml``'s ``pythonpath`` setting covers the same ground when
pytest is invoked from the repository root; this makes it work from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOTS = [
    "packages/shared_types",
    "packages/audio_protocol",
    "services/core",
    "services/evaluation",
    "services/reference_retrieval",
    "services/voice_preprocessing",
    "services/voice_training",
    "services/sidecar",
    "services/text_normalizer",
    "services/speech_planner",
    "services/conversation",
    "providers/tts",
    "providers/stt",
    "providers/llm",
]
for _root in _SOURCE_ROOTS:
    _path = str(REPO_ROOT / _root)
    if (REPO_ROOT / _root).is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def repo_root_path() -> Path:
    return REPO_ROOT


@pytest.fixture
def demo_pack(tmp_path: Path):
    """A freshly built synthetic voice pack in a temporary directory."""
    from cvai_evaluation.demo_pack import build_demo_voicepack

    return build_demo_voicepack(tmp_path / "voicepacks" / "demo_zh")


@pytest.fixture(scope="session")
def shared_demo_pack(tmp_path_factory):
    """One synthetic pack for the whole session.

    Building it writes a few dozen WAV files, which is slow to repeat per test. Use
    this only where the test *reads* the pack; anything that mutates it wants the
    function-scoped `demo_pack`.
    """
    from cvai_evaluation.demo_pack import build_demo_voicepack

    root = tmp_path_factory.mktemp("shared-pack") / "voicepacks" / "demo_zh"
    return build_demo_voicepack(root)


@pytest.fixture
def app_config(repo_root_path: Path):
    from cvai_core.config import load_config

    return load_config([repo_root_path / "configs" / "app.yaml"])
