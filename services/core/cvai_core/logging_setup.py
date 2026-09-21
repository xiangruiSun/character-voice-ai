"""Logging configuration.

One place, so that a CLI, the benchmark runner and (later) the FastAPI app all produce
the same output. Nothing exotic: stderr, level from config, and a helper that keeps the
noisy third-party loggers quiet during long runs.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_FORMAT: Final = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_NOISY: Final = ("httpx", "httpcore", "urllib3", "asyncio", "matplotlib")

_configured = False


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
