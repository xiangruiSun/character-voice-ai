"""LLM adapters. Importing this package registers them."""

from __future__ import annotations

from .mock_llm import MockLLMProvider
from .openai_llm import OpenAILLMProvider

__all__ = ["MockLLMProvider", "OpenAILLMProvider"]
