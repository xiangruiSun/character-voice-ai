"""Qwen3-TTS sidecar engine.

Wraps ``Qwen3TTSModel`` from the ``qwen-tts`` package. The one non-obvious
responsibility is caching: ``create_voice_clone_prompt(ref_audio, ref_text)`` builds a
reusable prompt object, and rebuilding it for every sentence is this engine's version of
reloading weights from disk (spec §27). The client adapter sends
``voice_clone_prompt_key`` — the Reference Bank clip id — and the cache is keyed on that,
so a conversation that rotates through six reference clips pays the cost six times, not
once per line.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..handlers import SidecarError, to_float_samples

ALLOWED_CALLS = ("generate_voice_clone", "generate_custom_voice", "generate_voice_design")


class QwenTTSEngine:
    name = "qwen3_tts"

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_prompt_cache: int = 32,
    ) -> None:
        self.model_id = model or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.version = self.model_id
        self.device = device
        self.dtype = dtype
        self.max_prompt_cache = max_prompt_cache
        self._model: Any = None
        self._prompts: dict[str, Any] = {}

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=getattr(torch, self.dtype),
        )

    def ready(self) -> bool:
        return self._model is not None

    def load_checkpoint(self, checkpoint_id: str) -> None:
        """Swap to a fine-tuned checkpoint.

        Reloading the model invalidates every cached prompt object: they hold state
        derived from the old weights, and reusing them silently produces the previous
        voice — the kind of bug that is only noticed after a listening test.
        """
        self.model_id = checkpoint_id
        self.version = checkpoint_id
        self._prompts.clear()
        self.load()

    def call(
        self, method: str, kwargs: dict[str, Any], seed: int | None
    ) -> tuple[Sequence[float], int]:
        if self._model is None:
            raise SidecarError("model is not loaded", 503)

        arguments = dict(kwargs)
        key = arguments.pop("voice_clone_prompt_key", None)

        if seed is not None:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)

        if method == "generate_voice_clone" and key:
            arguments["voice_clone_prompt"] = self._prompt_for(
                key, arguments.pop("ref_audio", None), arguments.pop("ref_text", None)
            )

        function = getattr(self._model, method, None)
        if function is None:
            raise SidecarError(f"{self.name} has no method {method!r}", 400)

        result = function(**arguments)
        return _unpack(result)

    def _prompt_for(self, key: str, ref_audio: Any, ref_text: Any) -> Any:
        cached = self._prompts.get(key)
        if cached is not None:
            return cached
        if ref_audio is None:
            raise SidecarError(
                f"no cached prompt for reference {key!r} and no ref_audio supplied", 400
            )
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text
        )
        if len(self._prompts) >= self.max_prompt_cache:
            self._prompts.pop(next(iter(self._prompts)))
        self._prompts[key] = prompt
        return prompt


def _unpack(result: Any) -> tuple[list[float], int]:
    """``generate_*`` returns ``(wavs, sample_rate)`` with ``wavs`` batched."""
    if isinstance(result, tuple) and len(result) == 2:
        wavs, rate = result
        first = wavs[0] if isinstance(wavs, (list, tuple)) or hasattr(wavs, "shape") else wavs
        return to_float_samples(first), int(rate)
    raise SidecarError(f"unexpected return shape from qwen-tts: {type(result).__name__}")


def build(**options: Any) -> QwenTTSEngine:
    # Drop unset options so the class defaults apply, rather than passing None through.
    return QwenTTSEngine(**{k: v for k, v in options.items() if v is not None})
