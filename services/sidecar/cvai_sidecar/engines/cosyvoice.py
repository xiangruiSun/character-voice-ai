"""CosyVoice / Fun-CosyVoice3 sidecar engine.

Its inference methods are generators that yield ``{"tts_speech": tensor}`` chunks, so a
non-streaming request concatenates them. The client adapter routes to
``inference_instruct2`` when it supplied an instruction and ``inference_zero_shot``
otherwise; the mapping lives here so the adapter does not need to know the method names.

Prompt audio is loaded and cached per path: CosyVoice wants a tensor, and re-reading the
same reference WAV for every line of a conversation is pure waste.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..handlers import SidecarError, to_float_samples

ALLOWED_CALLS = ("inference_zero_shot", "inference_instruct2", "inference_cross_lingual")


class CosyVoiceEngine:
    name = "cosyvoice"

    def __init__(
        self,
        *,
        model: str | None = None,
        model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B",
        device: str = "cuda",
        prompt_sample_rate: int = 16000,
    ) -> None:
        self.version = model or "Fun-CosyVoice3-0.5B"
        self.model_dir = model_dir
        self.device = device
        self.prompt_sample_rate = prompt_sample_rate
        self._model: Any = None
        self._prompts: dict[str, Any] = {}

    def load(self) -> None:
        from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

        self._model = AutoModel(model_dir=self.model_dir)

    def ready(self) -> bool:
        return self._model is not None

    def load_checkpoint(self, checkpoint_id: str) -> None:
        self.model_dir = checkpoint_id
        self.version = checkpoint_id
        self._prompts.clear()
        self.load()

    def call(
        self, method: str, kwargs: dict[str, Any], seed: int | None
    ) -> tuple[Sequence[float], int]:
        if self._model is None:
            raise SidecarError("model is not loaded", 503)
        if seed is not None:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)

        arguments = dict(kwargs)
        # The client sends a path; the engine wants a loaded tensor.
        wav_path = arguments.pop("prompt_wav", None)
        if wav_path:
            arguments["prompt_speech_16k"] = self._prompt_tensor(str(wav_path))

        instruct = arguments.pop("instruct_text", None)
        if instruct and method == "inference_zero_shot":
            method = "inference_instruct2"
            arguments["instruct_text"] = instruct
        elif instruct:
            arguments["instruct_text"] = instruct

        function = getattr(self._model, method, None)
        if function is None:
            raise SidecarError(f"{self.name} has no method {method!r}", 400)

        chunks = function(**arguments)
        return self._collect(chunks), int(getattr(self._model, "sample_rate", 24000))

    def _prompt_tensor(self, path: str) -> Any:
        cached = self._prompts.get(path)
        if cached is not None:
            return cached
        from cosyvoice.utils.file_utils import load_wav  # noqa: PLC0415

        tensor = load_wav(path, self.prompt_sample_rate)
        self._prompts[path] = tensor
        return tensor

    @staticmethod
    def _collect(chunks: Iterable[Any]) -> list[float]:
        samples: list[float] = []
        for chunk in chunks:
            audio = chunk["tts_speech"] if isinstance(chunk, dict) else chunk
            samples.extend(to_float_samples(audio))
        return samples


def build(**options: Any) -> CosyVoiceEngine:
    return CosyVoiceEngine(**{k: v for k, v in options.items() if v is not None})
