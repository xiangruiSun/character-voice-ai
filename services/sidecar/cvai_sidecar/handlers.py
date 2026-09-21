"""Sidecar request handling, independent of the web framework.

The HTTP wiring lives in ``server.py``; everything that can be wrong lives here, so it
can be tested without FastAPI, without an engine, and without a GPU. What a sidecar
actually has to get right is small and unglamorous:

* call the engine method the adapter asked for, with the arguments it sent;
* keep the model and any per-reference prompt objects loaded across requests;
* return real WAV bytes with the true sample rate in a header;
* fail with a readable message instead of returning something that looks like audio.

That last point matters more than it sounds. An engine that silently returns a default
voice, or 200 ms of silence, produces a file that reaches the listening test looking
like a result. A 500 with a reason gets recorded against the candidate and shows up in
the run report.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

SYNTHESIZE_PATH = "/v1/synthesize"
CHECKPOINT_PATH = "/v1/checkpoint"
HEALTH_PATH = "/health"

SAMPLE_RATE_HEADER = "X-CVAI-Sample-Rate"
ENGINE_VERSION_HEADER = "X-CVAI-Engine-Version"


class SidecarError(RuntimeError):
    """Something the caller should see, with an HTTP status attached."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineAdapter(Protocol):
    """What a per-engine module must provide."""

    name: str
    version: str

    def load(self) -> None:
        """Load weights. Called once at start-up, never per request."""

    def ready(self) -> bool:
        ...

    def call(self, method: str, kwargs: dict[str, Any], seed: int | None) -> tuple[Sequence[float], int]:
        """Run one synthesis. Returns ``(samples in [-1, 1], sample_rate)``."""

    def load_checkpoint(self, checkpoint_id: str) -> None:
        ...


@dataclass
class SynthesizeRequest:
    call: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    sample_rate: int | None = None

    @classmethod
    def parse(cls, payload: Any) -> "SynthesizeRequest":
        if not isinstance(payload, dict):
            raise SidecarError("request body must be a JSON object", 400)
        call = payload.get("call")
        if not isinstance(call, str) or not call:
            raise SidecarError("'call' must name an engine method", 400)
        kwargs = payload.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise SidecarError("'kwargs' must be an object", 400)
        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise SidecarError("'seed' must be an integer or null", 400)
        rate = payload.get("sample_rate")
        if rate is not None and (not isinstance(rate, int) or rate < 8000 or rate > 48000):
            raise SidecarError("'sample_rate' must be between 8000 and 48000", 400)
        return cls(call=call, kwargs=kwargs, seed=seed, sample_rate=rate)


@dataclass
class SynthesizeResponse:
    audio_wav: bytes
    sample_rate: int
    engine_version: str

    def headers(self) -> dict[str, str]:
        return {
            SAMPLE_RATE_HEADER: str(self.sample_rate),
            ENGINE_VERSION_HEADER: self.engine_version,
        }


class Sidecar:
    """Stateful wrapper around one engine adapter."""

    #: Methods an adapter may expose. An allow-list rather than free dispatch: ``call``
    #: arrives over HTTP, and reflecting it onto arbitrary attributes of a loaded model
    #: object would turn this into a remote code execution surface.
    def __init__(self, adapter: EngineAdapter, *, allowed_calls: Sequence[str]) -> None:
        self.adapter = adapter
        self.allowed_calls = set(allowed_calls)
        self.loaded_checkpoint: str | None = None
        self._loaded = False

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        self.adapter.load()
        self._loaded = True

    def health(self) -> dict[str, Any]:
        return {
            "engine": self.adapter.name,
            "version": self.adapter.version,
            "ready": bool(self._loaded and self.adapter.ready()),
            "checkpoint": self.loaded_checkpoint,
        }

    # -- endpoints -----------------------------------------------------------------

    def set_checkpoint(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not payload.get("checkpoint_id"):
            raise SidecarError("'checkpoint_id' is required", 400)
        checkpoint_id = str(payload["checkpoint_id"])
        if checkpoint_id == self.loaded_checkpoint:
            return {"loaded": checkpoint_id, "changed": False}
        try:
            self.adapter.load_checkpoint(checkpoint_id)
        except NotImplementedError as exc:
            raise SidecarError(
                f"{self.adapter.name} cannot switch checkpoints at runtime: {exc}", 400
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SidecarError(f"loading checkpoint failed: {exc}") from exc
        self.loaded_checkpoint = checkpoint_id
        return {"loaded": checkpoint_id, "changed": True}

    def synthesize(self, payload: Any) -> SynthesizeResponse:
        request = SynthesizeRequest.parse(payload)
        if request.call not in self.allowed_calls:
            raise SidecarError(
                f"call {request.call!r} is not exposed by the {self.adapter.name} "
                f"sidecar; allowed: {sorted(self.allowed_calls)}",
                400,
            )
        if not self._loaded:
            raise SidecarError(f"{self.adapter.name} is still loading", 503)

        try:
            samples, rate = self.adapter.call(request.call, request.kwargs, request.seed)
        except SidecarError:
            raise
        except TypeError as exc:
            # Almost always an adapter/sidecar version mismatch on argument names —
            # worth distinguishing from a genuine engine failure.
            raise SidecarError(
                f"{self.adapter.name}.{request.call} rejected its arguments: {exc}. "
                "The client adapter and this sidecar are probably out of sync.",
                400,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SidecarError(f"{self.adapter.name} synthesis failed: {exc}") from exc

        samples = list(samples)
        if not samples:
            raise SidecarError(
                f"{self.adapter.name} returned no audio. Refusing to answer with an "
                "empty file: silence in a listening test looks like a result."
            )
        if rate <= 0:
            raise SidecarError(f"{self.adapter.name} reported sample rate {rate}")

        if request.sample_rate and request.sample_rate != rate:
            samples = resample_linear(samples, rate, request.sample_rate)
            rate = request.sample_rate

        return SynthesizeResponse(
            audio_wav=encode_wav(samples, rate),
            sample_rate=rate,
            engine_version=self.adapter.version,
        )


# --------------------------------------------------------------------------------------
# Audio encoding
# --------------------------------------------------------------------------------------


def encode_wav(samples: Sequence[float], sample_rate: int, channels: int = 1) -> bytes:
    """Float samples in [-1, 1] to 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    frames = bytearray()
    for value in samples:
        clamped = -1.0 if value < -1.0 else (1.0 if value > 1.0 else float(value))
        frames += struct.pack("<h", int(round(clamped * 32767)))
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def resample_linear(
    samples: Sequence[float], source_rate: int, target_rate: int
) -> list[float]:
    """Linear resample.

    Only used when a caller explicitly asks for a rate the engine does not produce. The
    benchmark normally leaves ``sample_rate`` unset precisely so that a 48 kHz engine's
    output is judged at 48 kHz rather than quietly downsampled here.
    """
    if source_rate == target_rate or not samples:
        return list(samples)
    ratio = target_rate / source_rate
    count = max(1, int(len(samples) * ratio))
    out: list[float] = []
    for index in range(count):
        position = index / ratio
        left = int(position)
        right = min(len(samples) - 1, left + 1)
        fraction = position - left
        out.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return out


def read_wav_file(path: str) -> tuple[list[float], int]:
    """Read a 16-bit WAV back into floats.

    Some engines only expose a "write to this path" API, so the adapter round-trips
    through a temporary file. Standard library only, to keep the sidecar image thin.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SidecarError(f"{path}: expected 16-bit PCM")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    count = len(raw) // 2
    values = struct.unpack(f"<{count}h", raw)
    floats = [v / 32767.0 for v in values]
    if channels > 1:
        floats = [
            sum(floats[i : i + channels]) / channels
            for i in range(0, len(floats) - channels + 1, channels)
        ]
    return floats, rate


def to_float_samples(data: Any) -> list[float]:
    """Normalize whatever an engine returned into a flat list of floats.

    Engines return numpy arrays, torch tensors, nested lists or ``(array, rate)``
    tuples. Each adapter unwraps the tuple; this handles the array itself.
    """
    if hasattr(data, "detach"):  # torch tensor
        data = data.detach().cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (list, tuple)):
            # Channel-major (C, N) — downmix; dialogue is mono and nothing downstream
            # wants two channels.
            channels = [to_float_samples(row) for row in data]
            length = min(len(c) for c in channels)
            return [sum(c[i] for c in channels) / len(channels) for i in range(length)]
        return [float(v) for v in data]
    raise SidecarError(f"cannot interpret engine output of type {type(data).__name__}")


def build_sidecar(
    adapter_factory: Callable[[], EngineAdapter], allowed_calls: Sequence[str]
) -> Sidecar:
    return Sidecar(adapter_factory(), allowed_calls=allowed_calls)
