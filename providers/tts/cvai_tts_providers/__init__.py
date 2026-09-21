"""TTS provider adapters.

Importing this package registers every adapter with ``cvai_core.registry.TTS_PROVIDERS``.
Nothing else in the codebase imports these classes by name — configuration selects them
by key, which is what keeps model-specific code out of the conversation and evaluation
layers (spec §23).

Adapter status as of Milestone 1:

===============  ==========================================================
``mock``         fully working, dependency-free, used by tests and the demo
``gpt_sovits``   written against ``api_v2``; live validation in Milestone 3
``qwen3_tts``    written against ``qwen-tts``; live validation in Milestone 4
``fish_speech``  written against the OpenAudio API; validation in Milestone 5
``index_tts``    optional candidate (emotion vectors); Milestone 5b
``cosyvoice``    optional candidate (streaming); Milestone 5c
``voxcpm``       optional candidate (48 kHz, LoRA); Milestone 5d
===============  ==========================================================
"""

from __future__ import annotations

from .base import HttpSidecarProvider, derive_seed, finalize_result
from .cosyvoice import CosyVoiceProvider
from .fish_speech import FishSpeechProvider
from .gpt_sovits import GPTSoVITSProvider
from .index_tts import IndexTTSProvider
from .mock import MockTTSProvider
from .qwen3_tts import QwenTTSProvider
from .sidecar import GenericSidecarProvider
from .voxcpm import VoxCPMProvider

__all__ = [
    "CosyVoiceProvider",
    "FishSpeechProvider",
    "GPTSoVITSProvider",
    "GenericSidecarProvider",
    "HttpSidecarProvider",
    "IndexTTSProvider",
    "MockTTSProvider",
    "QwenTTSProvider",
    "VoxCPMProvider",
    "derive_seed",
    "finalize_result",
]
