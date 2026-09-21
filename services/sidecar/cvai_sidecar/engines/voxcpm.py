"""VoxCPM2 sidecar engine.

The simplest of the four: ``pip install voxcpm``, one ``generate`` call, and it is the
only candidate that runs on CPU or MPS as well as CUDA — which makes it the one engine
that can be smoke-tested on a laptop before a GPU box is involved.

Output is 48 kHz native. The sample rate is reported as-is and never downsampled here;
that resolution is part of what the listening test is judging.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..handlers import SidecarError, to_float_samples

ALLOWED_CALLS = ("generate",)

NATIVE_SAMPLE_RATE = 48000


class VoxCPMEngine:
    name = "voxcpm"

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_id = model or "openbmb/VoxCPM2"
        self.version = self.model_id
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        from voxcpm import VoxCPM  # noqa: PLC0415

        self._model = VoxCPM.from_pretrained(self.model_id, device=self.device)

    def ready(self) -> bool:
        return self._model is not None

    def load_checkpoint(self, checkpoint_id: str) -> None:
        self.model_id = checkpoint_id
        self.version = checkpoint_id
        self.load()

    def call(
        self, method: str, kwargs: dict[str, Any], seed: int | None
    ) -> tuple[Sequence[float], int]:
        if self._model is None:
            raise SidecarError("model is not loaded", 503)
        if seed is not None:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)

        result = self._model.generate(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            audio, rate = result
            return to_float_samples(audio), int(rate)
        return to_float_samples(result), NATIVE_SAMPLE_RATE


def build(**options: Any) -> VoxCPMEngine:
    return VoxCPMEngine(**{k: v for k, v in options.items() if v is not None})
