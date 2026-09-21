"""LLM adapters. Importing this package registers them."""

from __future__ import annotations

from .openai_llm import OpenAILLMProvider

__all__ = ["OpenAILLMProvider"]
