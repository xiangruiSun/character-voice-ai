"""Engine sidecar (decision D1).

Every TTS engine runs in its own process with its own Python environment, because the
six candidates pin mutually incompatible torch and CUDA versions. This package is the
uniform HTTP surface they present, so the application and the benchmark talk to one
protocol regardless of which engine is behind it.

The split of responsibility is the point:

* the **client adapter** (``providers/tts/``) maps an engine-neutral request onto that
  engine's own keyword arguments — the interesting, reviewable, unit-tested part, kept
  in this repository next to the capability declaration it has to agree with;
* the **sidecar** loads the model once and calls it with what it was handed.

Protocol: ``docs/SIDECAR_PROTOCOL.md``.
"""

from __future__ import annotations

from .handlers import (
    CHECKPOINT_PATH,
    ENGINE_VERSION_HEADER,
    HEALTH_PATH,
    SAMPLE_RATE_HEADER,
    SYNTHESIZE_PATH,
    EngineAdapter,
    Sidecar,
    SidecarError,
    SynthesizeRequest,
    SynthesizeResponse,
    encode_wav,
    read_wav_file,
    resample_linear,
    to_float_samples,
)

__all__ = [
    "CHECKPOINT_PATH",
    "ENGINE_VERSION_HEADER",
    "EngineAdapter",
    "HEALTH_PATH",
    "SAMPLE_RATE_HEADER",
    "SYNTHESIZE_PATH",
    "Sidecar",
    "SidecarError",
    "SynthesizeRequest",
    "SynthesizeResponse",
    "encode_wav",
    "read_wav_file",
    "resample_linear",
    "to_float_samples",
]
