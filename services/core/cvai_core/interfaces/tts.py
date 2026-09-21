"""``TTSProvider`` — the interface every voice engine hides behind.

Designed against six real engines whose control surfaces differ substantially (see
``docs/TECH_LANDSCAPE.md``). Three consequences show up directly in this interface:

* :meth:`capabilities` is mandatory, and callers are expected to read it. An adapter that
  silently drops an emotion vector makes a listening test unreadable, so dropped controls
  are reported on the result instead.
* :meth:`load_checkpoint` exists because some engines can hot-swap a character checkpoint
  over their API (GPT-SoVITS ``/set_gpt_weights``). Spec §27 lists "loading the model from
  disk for every sentence" as a failure mode; this is the hook that avoids it.
* :meth:`synthesize_stream` is optional and defaults to raising, so Milestone 12 can add
  streaming per engine without every adapter being rewritten.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from pathlib import Path

from cvai_types import ProviderCapabilities, TTSHealth, TTSRequest, TTSResult

from ..errors import UnsupportedFeatureError


class TTSProvider(abc.ABC):
    """Synthesize one character utterance."""

    #: Registry key and the value written into ``TTSResult.engine``.
    engine: str = "unknown"

    # -- required ----------------------------------------------------------------

    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """What this adapter can honour. Must be cheap and must not touch the network."""

    @abc.abstractmethod
    async def synthesize(self, request: TTSRequest, output_path: Path) -> TTSResult:
        """Render ``request`` to a WAV file at ``output_path``.

        Implementations must:

        * write the file before returning, and return its real duration and sample rate;
        * populate ``resolved_params`` with exactly what was sent to the engine — this is
          what makes a run reproducible (spec §23);
        * populate ``dropped_controls`` from
          :meth:`ProviderCapabilities.unsupported_controls`;
        * raise :class:`~cvai_core.errors.ProviderUnavailableError` when the engine is
          unreachable, and :class:`~cvai_core.errors.SynthesisError` when the engine
          answered but failed. The benchmark distinguishes the two.
        """

    # -- optional ----------------------------------------------------------------

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[bytes]:
        """Yield PCM/encoded chunks as they are produced (Milestone 12)."""
        raise UnsupportedFeatureError(f"{self.engine} does not support streaming")
        yield b""  # pragma: no cover - makes the function an async generator

    async def health(self) -> TTSHealth:
        """Probe readiness. Default assumes a local, always-available implementation."""
        return TTSHealth(engine=self.engine, available=True, detail="no probe implemented")

    async def warmup(self) -> None:
        """Load weights / open connections ahead of the first request."""
        return None

    async def load_checkpoint(self, checkpoint_id: str) -> None:
        """Switch to a character-specific checkpoint.

        Only meaningful when ``capabilities().supports_hot_checkpoint_swap`` is true.
        """
        raise UnsupportedFeatureError(
            f"{self.engine} cannot switch checkpoints at runtime"
        )

    async def aclose(self) -> None:
        """Release sockets, subprocesses and GPU memory."""
        return None

    # -- helpers -----------------------------------------------------------------

    def describe(self) -> str:
        caps = self.capabilities()
        return f"{caps.engine}@{caps.engine_version}"
