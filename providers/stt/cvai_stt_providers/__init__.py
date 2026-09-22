"""Speech-to-text adapters. Importing this package registers them."""

from __future__ import annotations

from .funasr_stt import FunASRSTTProvider
from .openai_stt import OpenAISTTProvider

__all__ = ["FunASRSTTProvider", "OpenAISTTProvider"]
