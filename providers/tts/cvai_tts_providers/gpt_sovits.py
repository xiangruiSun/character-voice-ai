"""GPT-SoVITS adapter (Milestone 3).

Speaks the engine's own ``api_v2.py`` HTTP interface rather than a wrapper of ours,
because that API already exposes the two things this project needs most:

* ``prompt_text`` / ``ref_audio_path`` — reference-conditioned synthesis, which is the
  only way GPT-SoVITS expresses style, so the Reference Bank does all the emotional work
  for this engine;
* ``/set_gpt_weights`` and ``/set_sovits_weights`` — checkpoint hot-swap, which is how we
  avoid "loading the model from disk for every sentence" (spec §27).

Parameter names and defaults below follow ``api_v2.py`` as surveyed in
``docs/TECH_LANDSCAPE.md``. They are pinned in ``configs/providers/tts.gpt_sovits.yaml``
so a change upstream is a config edit, not a code change.

**Status:** written against the documented API and exercised by unit tests with a stub
transport; not yet run against a live sidecar. Milestone 3 is where that happens.
"""

from __future__ import annotations

import asyncio

from pathlib import Path
from typing import Any

from cvai_core.errors import SynthesisError
from cvai_core.registry import TTS_PROVIDERS
from cvai_types import AdaptationMode, ProviderCapabilities, TTSRequest, TTSResult

from .base import HttpSidecarProvider, TimedCall, derive_seed, finalize_result, write_audio_bytes

#: GPT-SoVITS language codes. V1 only ever sends Mandarin.
_LANG = "zh"

#: One api_v2 server holds one set of weights for everyone. What is loaded, and the
#: lock that makes "swap weights, then synthesize" atomic, belong to the server — not
#: to each adapter instance — or two characters' sessions would speak in each other's
#: voices.
_SERVER_CHECKPOINT: dict[str, str | None] = {}
_SERVER_LOCKS: dict[str, asyncio.Lock] = {}


def _server_lock(base_url: str) -> asyncio.Lock:
    lock = _SERVER_LOCKS.get(base_url)
    if lock is None:
        lock = _SERVER_LOCKS[base_url] = asyncio.Lock()
    return lock


@TTS_PROVIDERS.register("gpt_sovits")
class GPTSoVITSProvider(HttpSidecarProvider):
    engine = "gpt_sovits"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9880",
        *,
        engine_version: str = "v4",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
        parallel_infer: bool = True,
        media_type: str = "wav",
        native_sample_rate: int = 48000,
        default_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url, **kwargs)
        self.engine_version = engine_version
        #: Weights for this character's voice, used when a request names none.
        self.default_checkpoint = default_checkpoint
        self.defaults: dict[str, Any] = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "fragment_interval": fragment_interval,
            "repetition_penalty": repetition_penalty,
            "sample_steps": sample_steps,
            "super_sampling": super_sampling,
            "parallel_infer": parallel_infer,
            "media_type": media_type,
        }
        self.native_sample_rate = native_sample_rate
        self.loaded_checkpoint: str | None = None

    # -- interface ----------------------------------------------------------------

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            engine=self.engine,
            engine_version=self.engine_version,
            supported_adaptation_modes=[
                AdaptationMode.ZERO_SHOT,
                AdaptationMode.FINETUNED,
            ],
            native_sample_rate=self.native_sample_rate,
            supports_reference_audio=True,
            # Not strictly required, but quality drops sharply without it, and the
            # Reference Bank always has it.
            requires_reference_text=False,
            supports_multiple_references=True,  # aux_ref_audio_paths
            supports_instruct=False,
            supports_emotion_vector=False,
            supports_duration_control=False,
            supports_speed_factor=True,  # speed_factor
            supports_seed=True,
            supports_streaming=True,  # streaming_mode 1/2/3
            supports_hot_checkpoint_swap=True,
            license="MIT",
            commercial_use=True,
        )

    @property
    def health_path(self) -> str:
        # api_v2 has no dedicated health route; a trivial control ping is the cheapest
        # liveness probe that does not synthesize anything.
        return "/docs"

    async def load_checkpoint(self, checkpoint_id: str) -> None:
        """Point the sidecar at a character's GPT and SoVITS weights.

        ``checkpoint_id`` is ``"<gpt_weights_path>|<sovits_weights_path>"`` — the engine
        keeps the two stages separate and both must be switched together, so they travel
        as one id rather than as two fields that can drift apart.
        """
        parts = [p for p in checkpoint_id.split("|") if p]
        if len(parts) != 2:
            raise SynthesisError(
                "GPT-SoVITS checkpoint_id must be '<gpt_weights>|<sovits_weights>', "
                f"got {checkpoint_id!r}"
            )
        gpt_weights, sovits_weights = parts
        await self._request(
            "GET", "/set_gpt_weights", params={"weights_path": gpt_weights}
        )
        await self._request(
            "GET", "/set_sovits_weights", params={"weights_path": sovits_weights}
        )
        self.loaded_checkpoint = checkpoint_id
        _SERVER_CHECKPOINT[self.base_url] = checkpoint_id

    async def synthesize(self, request: TTSRequest, output_path: Path) -> TTSResult:
        if request.reference is None:
            raise SynthesisError(
                "GPT-SoVITS always needs a reference clip: it has no other way to "
                "express style"
            )
        checkpoint = request.checkpoint_id or self.default_checkpoint
        seed = derive_seed(request, salt=self.engine_version)
        payload = self._build_payload(request, seed)

        async with _server_lock(self.base_url):
            if checkpoint and checkpoint != _SERVER_CHECKPOINT.get(self.base_url):
                await self.load_checkpoint(checkpoint)
            with TimedCall() as timer:
                audio = await self._request("POST", "/tts", json=payload, expect_audio=True)
                if not audio:
                    raise SynthesisError("GPT-SoVITS returned an empty response body")
                write_audio_bytes(audio, output_path)

        return finalize_result(
            request,
            self.capabilities(),
            output_path,
            latency_ms=timer.elapsed_ms,
            seed=seed,
            resolved_params=payload,
        )

    # -- payload ------------------------------------------------------------------

    def _build_payload(self, request: TTSRequest, seed: int) -> dict[str, Any]:
        reference = request.reference
        assert reference is not None  # guarded by the caller

        payload: dict[str, Any] = dict(self.defaults)
        payload.update(
            {
                "text": request.text,
                "text_lang": _LANG,
                "ref_audio_path": reference.audio_path,
                "prompt_text": reference.transcript,
                "prompt_lang": _LANG,
                "seed": seed,
                # Style reaches this engine only as speed and reference choice.
                "speed_factor": round(
                    self.defaults["speed_factor"] * request.controls.speed_factor(), 4
                ),
                "streaming_mode": bool(request.stream),
            }
        )
        # Per-request overrides win over adapter defaults; the benchmark uses this to
        # sweep sampling parameters without editing configuration.
        payload.update(request.engine_params)
        return payload
