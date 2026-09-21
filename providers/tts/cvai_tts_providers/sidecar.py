"""The CVAI sidecar protocol.

Four of the candidate engines (Qwen3-TTS, IndexTTS-2.5, CosyVoice3, VoxCPM2) are used
through their Python APIs rather than a built-in HTTP server. Running them in-process is
not an option — decision D1, they have incompatible torch/CUDA pins — so each gets a
small uniform HTTP wrapper.

The split of responsibility is deliberate:

* **The adapter, in this repo**, maps our engine-neutral request onto that engine's
  *native keyword arguments*. That mapping is the interesting, reviewable,
  unit-testable part, and it stays in version control next to the capability
  declaration it has to agree with.
* **The sidecar, in the engine's own container**, is dumb: it loads the model once and
  calls it with the keyword arguments it was handed.

Protocol:

    GET  /health            -> {"engine", "version", "ready"}
    POST /v1/checkpoint     <- {"checkpoint_id"}
    POST /v1/synthesize     <- {"call", "kwargs", "seed", "sample_rate"}
                            -> audio/wav bytes
                               X-CVAI-Sample-Rate, X-CVAI-Engine-Version headers

``call`` names the engine method (``generate_voice_clone``, ``inference_zero_shot``,
``infer``…), so one sidecar image can expose every entry point an engine has without the
protocol growing a new field each time.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from cvai_core.errors import SynthesisError
from cvai_types import ProviderCapabilities, TTSRequest, TTSResult

from .base import HttpSidecarProvider, TimedCall, derive_seed, finalize_result, write_audio_bytes

SYNTHESIZE_PATH = "/v1/synthesize"
CHECKPOINT_PATH = "/v1/checkpoint"


class GenericSidecarProvider(HttpSidecarProvider):
    """Base for adapters that drive an engine's Python API through a sidecar."""

    #: Engine method the sidecar should call.
    call_name: str = "synthesize"

    def __init__(self, base_url: str, *, engine_version: str = "unknown", **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self.engine_version = engine_version
        self.loaded_checkpoint: str | None = None

    # -- to implement in each engine adapter ---------------------------------------

    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:  # pragma: no cover - abstract
        ...

    @abc.abstractmethod
    def native_kwargs(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        """Translate the neutral request into this engine's own arguments."""

    # -- shared ---------------------------------------------------------------------

    async def load_checkpoint(self, checkpoint_id: str) -> None:
        await self._request(
            "POST", CHECKPOINT_PATH, json={"checkpoint_id": checkpoint_id}
        )
        self.loaded_checkpoint = checkpoint_id

    async def synthesize(self, request: TTSRequest, output_path: Path) -> TTSResult:
        capabilities = self.capabilities()
        if capabilities.requires_reference_text and (
            request.reference is None or not request.reference.transcript.strip()
        ):
            raise SynthesisError(
                f"{self.engine} requires the reference transcript; the Reference Bank "
                "should always carry one (decision D3)"
            )
        if request.checkpoint_id and request.checkpoint_id != self.loaded_checkpoint:
            await self.load_checkpoint(request.checkpoint_id)

        seed = derive_seed(request, salt=self.engine_version)
        kwargs = self.native_kwargs(request, seed)
        kwargs.update(request.engine_params)

        body = {
            "call": self.call_name,
            "kwargs": kwargs,
            "seed": seed,
            "sample_rate": request.output_sample_rate,
        }

        with TimedCall() as timer:
            audio = await self._request(
                "POST", SYNTHESIZE_PATH, json=body, expect_audio=True
            )
            if not audio:
                raise SynthesisError(f"{self.engine} sidecar returned no audio")
            write_audio_bytes(audio, output_path)

        return finalize_result(
            request,
            capabilities,
            output_path,
            latency_ms=timer.elapsed_ms,
            seed=seed,
            resolved_params={"call": self.call_name, **kwargs},
        )
