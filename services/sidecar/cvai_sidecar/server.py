"""FastAPI wiring for an engine sidecar.

Thin on purpose — everything worth testing is in :mod:`cvai_sidecar.handlers`. This file
only maps HTTP onto those handlers and starts the engine once at boot.

Run inside an engine's own container:

    CVAI_SIDECAR_ENGINE=qwen3_tts \\
    CVAI_SIDECAR_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
    python -m cvai_sidecar.server --host 0.0.0.0 --port 9881

The engine module is chosen by name at start-up, so one image layout and one entry point
serve every engine; only the base image and the pinned requirements differ.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from typing import Any

from .handlers import (
    CHECKPOINT_PATH,
    HEALTH_PATH,
    SYNTHESIZE_PATH,
    EngineAdapter,
    Sidecar,
    SidecarError,
)

log = logging.getLogger("cvai-sidecar")

ENGINE_ENV = "CVAI_SIDECAR_ENGINE"
MODEL_ENV = "CVAI_SIDECAR_MODEL"
DEVICE_ENV = "CVAI_SIDECAR_DEVICE"


def load_adapter(engine: str, **options: Any) -> tuple[EngineAdapter, list[str]]:
    """Import ``cvai_sidecar.engines.<engine>`` and build its adapter."""
    try:
        module = importlib.import_module(f".engines.{engine}", package=__package__)
    except ImportError as exc:
        raise SystemExit(
            f"no sidecar engine module for {engine!r}: {exc}\n"
            "Available: qwen3_tts, index_tts, cosyvoice, voxcpm"
        ) from exc
    adapter = module.build(**options)
    return adapter, list(module.ALLOWED_CALLS)


def create_app(engine: str | None = None, **options: Any):
    """Build the ASGI app. Imports FastAPI lazily so this module is importable without it."""
    try:
        from fastapi import FastAPI, Request  # noqa: PLC0415
        from fastapi.responses import JSONResponse, Response  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - only in a container
        raise SystemExit(
            "the sidecar needs FastAPI: pip install 'fastapi' 'uvicorn[standard]'"
        ) from exc

    engine_name = engine or os.environ.get(ENGINE_ENV)
    if not engine_name:
        raise SystemExit(f"set {ENGINE_ENV} or pass --engine")

    options.setdefault("model", os.environ.get(MODEL_ENV))
    options.setdefault("device", os.environ.get(DEVICE_ENV, "cuda"))
    adapter, allowed = load_adapter(engine_name, **options)
    sidecar = Sidecar(adapter, allowed_calls=allowed)

    app = FastAPI(title=f"cvai-sidecar:{engine_name}", docs_url="/docs")

    @app.on_event("startup")
    def _startup() -> None:
        # Loading at boot, not on the first request: a cold first synthesis inside a
        # benchmark shows up as a latency outlier and pollutes the timings.
        log.info("loading %s …", adapter.name)
        sidecar.start()
        log.info("%s ready", adapter.name)

    @app.get(HEALTH_PATH)
    def health() -> dict[str, Any]:
        return sidecar.health()

    @app.post(CHECKPOINT_PATH)
    async def checkpoint(request: Request):
        try:
            return sidecar.set_checkpoint(await request.json())
        except SidecarError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

    @app.post(SYNTHESIZE_PATH)
    async def synthesize(request: Request):
        try:
            result = sidecar.synthesize(await request.json())
        except SidecarError as exc:
            log.warning("synthesis rejected: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
        return Response(
            content=result.audio_wav,
            media_type="audio/wav",
            headers=result.headers(),
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cvai-sidecar")
    parser.add_argument("--engine", default=os.environ.get(ENGINE_ENV))
    parser.add_argument("--model", default=os.environ.get(MODEL_ENV))
    parser.add_argument("--device", default=os.environ.get(DEVICE_ENV, "cuda"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper())
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install 'uvicorn[standard]'") from exc

    app = create_app(args.engine, model=args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
