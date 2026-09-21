"""Core services: configuration, provider interfaces, registry, loaders, run logging.

This package is the only thing every other package is allowed to depend on. It contains
no model code, no engine-specific logic and no conversation logic — keeping those out is
what makes the adapters swappable (spec §23).
"""

from __future__ import annotations

from .audio import (
    DEFAULT_CHARS_PER_SECOND,
    estimate_speech_duration,
    read_wav_properties,
    read_wav_samples,
    write_wav,
)
from .config import (
    AppConfig,
    BenchmarkDefaults,
    PathsConfig,
    ProviderInstanceConfig,
    ProviderSlotConfig,
    ProvidersConfig,
    env_overrides,
    interpolate,
    load_config,
    load_yaml,
)
from .errors import (
    ConfigError,
    CVAIError,
    ProviderUnavailableError,
    ReferenceBankError,
    RegistryError,
    SynthesisError,
    UnsupportedFeatureError,
    VoicePackError,
)
from .interfaces import (
    CharacterProvider,
    ConversationSession,
    LLMProvider,
    ReferenceRetriever,
    SpeechToTextProvider,
    TTSProvider,
)
from .loaders import (
    FilesystemCharacterProvider,
    load_character_profile,
    load_dataset_manifest,
    load_reference_bank,
    load_sentence_set,
    load_voicepack_manifest,
    open_voicepack,
    save_model_json,
    save_model_yaml,
    try_load_dataset_manifest,
    try_load_reference_bank,
)
from .logging_setup import configure_logging, get_logger
from .paths import VoicePackPaths, character_profile_path, repo_root, voicepack_root
from .registry import (
    LLM_PROVIDERS,
    STT_PROVIDERS,
    TTS_PROVIDERS,
    ProviderRegistry,
    build_llm_provider,
    build_stt_provider,
    build_tts_provider,
)
from .runlog import RunLogger, RunPaths, create_run, environment_snapshot, new_run_id
from .voicepack import (
    Issue,
    Severity,
    ValidationReport,
    default_styles,
    scaffold_voicepack,
    validate_voicepack,
)

__all__ = [
    # audio
    "DEFAULT_CHARS_PER_SECOND",
    "estimate_speech_duration",
    "read_wav_properties",
    "read_wav_samples",
    "write_wav",
    # config
    "AppConfig",
    "BenchmarkDefaults",
    "PathsConfig",
    "ProviderInstanceConfig",
    "ProviderSlotConfig",
    "ProvidersConfig",
    "env_overrides",
    "interpolate",
    "load_config",
    "load_yaml",
    # errors
    "CVAIError",
    "ConfigError",
    "ProviderUnavailableError",
    "ReferenceBankError",
    "RegistryError",
    "SynthesisError",
    "UnsupportedFeatureError",
    "VoicePackError",
    # interfaces
    "CharacterProvider",
    "ConversationSession",
    "LLMProvider",
    "ReferenceRetriever",
    "SpeechToTextProvider",
    "TTSProvider",
    # loaders
    "FilesystemCharacterProvider",
    "load_character_profile",
    "load_dataset_manifest",
    "load_reference_bank",
    "load_sentence_set",
    "load_voicepack_manifest",
    "open_voicepack",
    "save_model_json",
    "save_model_yaml",
    "try_load_dataset_manifest",
    "try_load_reference_bank",
    # logging
    "configure_logging",
    "get_logger",
    # paths
    "VoicePackPaths",
    "character_profile_path",
    "repo_root",
    "voicepack_root",
    # registry
    "LLM_PROVIDERS",
    "STT_PROVIDERS",
    "TTS_PROVIDERS",
    "ProviderRegistry",
    "build_llm_provider",
    "build_stt_provider",
    "build_tts_provider",
    # runlog
    "RunLogger",
    "RunPaths",
    "create_run",
    "environment_snapshot",
    "new_run_id",
    # voicepack
    "Issue",
    "Severity",
    "ValidationReport",
    "default_styles",
    "scaffold_voicepack",
    "validate_voicepack",
]
