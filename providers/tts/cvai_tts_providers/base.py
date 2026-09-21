"""Shared adapter machinery.

Two things live here so no adapter has to reinvent them: deterministic seed derivation,
and the HTTP plumbing for engines that run in a sidecar process (decision D1 — every
engine has its own torch/CUDA pin, so none of them may be imported into this process).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from cvai_core.errors import ProviderUnavailableError, SynthesisError
from cvai_core.interfaces.tts import TTSProvider
from cvai_types import ProviderCapabilities, TTSHealth, TTSRequest, TTSResult

#: Timeout for a single synthesis call. Generous: a first request on a cold GPU can take
#: tens of seconds, and failing it would be misread as the engine being broken.
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_CONNECT_TIMEOUT_S = 5.0


def derive_seed(request: TTSRequest, salt: str = "") -> int:
    """A seed that is explicit if given and reproducible if not.

    An engine left to its own devices produces a different take every run, which makes a
    regression impossible to distinguish from sampling noise. When the request carries no
    seed we derive one from its content, so the same request always renders the same way.
    """
    if request.seed is not None:
        return request.seed
    material = "|".join(
        [
            salt,
            request.request_id,
            request.text,
            request.controls.effective_reference_style,
            request.reference.reference_id if request.reference else "",
            str(request.checkpoint_id or ""),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class TimedCall:
    """Context manager measuring wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start = 0.0

    def __enter__(self) -> "TimedCall":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class HttpSidecarProvider(TTSProvider):
    """Base for adapters that talk to an engine running in another process.

    ``httpx`` is imported lazily: Milestone 1 installs only pydantic and PyYAML, and the
    mock engine must keep working on a machine where no sidecar dependency exists.
    """

    engine = "http-sidecar"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.headers = headers or {}
        self._client: Any = None

    def _httpx(self) -> Any:
        try:
            import httpx  # noqa: PLC0415 - deliberately lazy
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderUnavailableError(
                f"{self.engine} adapter needs httpx; install the 'runtime' extra "
                "(pip install -e '.[runtime]')"
            ) from exc
        return httpx

    def client(self) -> Any:
        if self._client is None:
            httpx = self._httpx()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
                headers=self.headers,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expect_audio: bool = False,
    ) -> Any:
        httpx = self._httpx()
        client = self.client()
        try:
            response = await client.request(method, path, json=json, params=params)
        except Exception as exc:  # httpx.ConnectError and relatives
            if isinstance(exc, httpx.HTTPError):
                raise ProviderUnavailableError(
                    f"{self.engine} sidecar at {self.base_url} is unreachable: {exc}"
                ) from exc
            raise
        if response.status_code >= 400:
            raise SynthesisError(
                f"{self.engine} sidecar returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        return response.content if expect_audio else response

    async def health(self) -> TTSHealth:
        try:
            response = await self._request("GET", self.health_path)
        except ProviderUnavailableError as exc:
            return TTSHealth(engine=self.engine, available=False, detail=str(exc))
        except SynthesisError as exc:
            return TTSHealth(engine=self.engine, available=False, detail=str(exc))
        return TTSHealth(
            engine=self.engine, available=True, detail=f"HTTP {response.status_code}"
        )

    @property
    def health_path(self) -> str:
        return "/health"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # Subclasses must still implement these.
    def capabilities(self) -> ProviderCapabilities:  # pragma: no cover - abstract
        raise NotImplementedError

    async def synthesize(
        self, request: TTSRequest, output_path: Path
    ) -> TTSResult:  # pragma: no cover - abstract
        raise NotImplementedError


def write_audio_bytes(payload: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def finalize_result(
    request: TTSRequest,
    capabilities: ProviderCapabilities,
    output_path: Path,
    *,
    latency_ms: float,
    seed: int | None,
    resolved_params: dict[str, Any],
    sample_rate: int | None = None,
    duration_s: float | None = None,
) -> TTSResult:
    """Build a :class:`TTSResult`, probing the written file for its real properties.

    Probing rather than trusting the engine's claim is deliberate: a mismatch between the
    requested and delivered sample rate is a real and easy-to-miss bug, and it shows up
    here as a wrong number in the run record rather than as a mysteriously dull-sounding
    listening test.
    """
    from cvai_core.audio import read_wav_properties

    if sample_rate is None or duration_s is None:
        properties = read_wav_properties(output_path)
        sample_rate = sample_rate or properties.sample_rate
        duration_s = duration_s or properties.duration_s

    return TTSResult(
        request_id=request.request_id,
        audio_path=str(output_path),
        sample_rate=sample_rate,
        duration_s=duration_s,
        engine=capabilities.engine,
        engine_version=capabilities.engine_version,
        adaptation_mode=request.adaptation_mode,
        checkpoint_id=request.checkpoint_id,
        reference_id=request.reference.reference_id if request.reference else None,
        seed=seed,
        resolved_params=resolved_params,
        dropped_controls=capabilities.unsupported_controls(request.controls),
        latency_ms=round(latency_ms, 3),
    )
