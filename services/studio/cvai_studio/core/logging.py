"""Structured logging with entity ids, and a filter that keeps credentials out of logs."""

from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
    re.compile(r"(?i)(api[_-]?key[\"'=:\s]+)[^\s,\"'}]+"),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + "[redacted]", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


def get_logger(name: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logging.LoggerAdapter(logger, {})


def with_ids(logger: logging.LoggerAdapter, **ids: str | None) -> logging.LoggerAdapter:
    """``log = with_ids(log, training_job_id=job.id)`` → every line carries the id."""
    tags = " ".join(f"{k}={v}" for k, v in ids.items() if v)

    class _Tagged(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"[{tags}] {msg}", kwargs

    return _Tagged(logger.logger, {})
