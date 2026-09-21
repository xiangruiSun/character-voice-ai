"""OpenAI speech-to-text adapter (Milestone 11).

Spec §3: start here, and keep the interface open for ``faster-whisper`` or FunASR's
streaming Paraformer later. The ``openai`` package is imported lazily so that Milestone 1
runs with only pydantic and PyYAML installed.

Two Chinese-specific notes carried forward from the survey:

* ``language="zh"`` must be pinned. Left to auto-detect, short Mandarin utterances get
  misidentified often enough to break a conversation turn.
* Character and world proper nouns should be passed as a prompt. Generic ASR mangles
  them, and the LLM then answers a question the user did not ask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cvai_core.errors import ProviderUnavailableError
from cvai_core.interfaces.stt import SpeechToTextProvider
from cvai_core.registry import STT_PROVIDERS
from cvai_types import STTCapabilities, Transcript


@STT_PROVIDERS.register("openai")
class OpenAISTTProvider(SpeechToTextProvider):
    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-transcribe",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._client: Any = None

    def capabilities(self) -> STTCapabilities:
        return STTCapabilities(
            provider=self.provider,
            model=self.model,
            supports_streaming=False,
            supports_word_timestamps=False,
            supports_hotwords=True,  # via the prompt field
            languages=["zh-CN"],
        )

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415 - lazy
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError(
                "the openai package is not installed; pip install -e '.[runtime]'"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": self.timeout_s}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "zh-CN",
        hotwords: list[str] | None = None,
    ) -> Transcript:
        client = self._client_or_raise()
        prompt = "、".join(hotwords) if hotwords else None
        with Path(audio_path).open("rb") as handle:
            response = await client.audio.transcriptions.create(
                model=self.model,
                file=handle,
                language="zh",
                prompt=prompt,
            )
        return Transcript(
            text=getattr(response, "text", "") or "",
            language="zh-CN",
            provider=self.provider,
            model=self.model,
        )
