"""IndexTTS-2.5 sidecar engine.

``IndexTTS2.infer`` writes to a path rather than returning samples, so this round-trips
through a temporary file. Slightly wasteful and entirely fine: the write is microseconds
next to the synthesis.

This is the only candidate with emotion control decoupled from speaker identity, which
is why it is in the benchmark at all — so ``emo_vector``, ``emo_alpha`` and
``duration_factor`` are passed straight through from the client adapter rather than
being reinterpreted here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

from ..handlers import SidecarError, read_wav_file

ALLOWED_CALLS = ("infer",)


class IndexTTSEngine:
    name = "index_tts"

    def __init__(
        self,
        *,
        model: str | None = None,
        model_dir: str = "checkpoints",
        config_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.version = model or "IndexTTS-2.5"
        self.model_dir = model_dir
        self.config_path = config_path or f"{model_dir}/config.yaml"
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        from indextts.infer_v2_5 import IndexTTS2  # noqa: PLC0415

        self._model = IndexTTS2(cfg_path=self.config_path, model_dir=self.model_dir)

    def ready(self) -> bool:
        return self._model is not None

    def load_checkpoint(self, checkpoint_id: str) -> None:
        raise NotImplementedError(
            "IndexTTS-2.5 has no documented fine-tuning path; it competes zero-shot"
        )

    def call(
        self, method: str, kwargs: dict[str, Any], seed: int | None
    ) -> tuple[Sequence[float], int]:
        if self._model is None:
            raise SidecarError("model is not loaded", 503)
        if seed is not None:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.wav"
            self._model.infer(output_path=str(output), **kwargs)
            if not output.is_file():
                raise SidecarError("IndexTTS wrote no output file")
            return read_wav_file(str(output))


def build(**options: Any) -> IndexTTSEngine:
    return IndexTTSEngine(**{k: v for k, v in options.items() if v is not None})
