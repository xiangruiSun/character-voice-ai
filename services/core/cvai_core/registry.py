"""Provider registry and factories.

Adapters register themselves with a decorator; configuration names them by key. Nothing
outside ``providers/`` ever imports a concrete adapter class, which is what keeps
model-specific code out of conversation logic (spec §23).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from .config import AppConfig, ProviderSlotConfig
from .errors import ConfigError, RegistryError

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    """A name → factory map for one kind of provider."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise RegistryError(
                    f"{self.kind} provider {key!r} is already registered by "
                    f"{self._factories[key]!r}"
                )
            self._factories[key] = factory
            return factory

        return decorator

    def register_instance(self, key: str, factory: Callable[..., T]) -> None:
        """Imperative registration, for tests and dynamically discovered adapters."""
        if key in self._factories:
            raise RegistryError(f"{self.kind} provider {key!r} is already registered")
        self._factories[key] = factory

    def unregister(self, key: str) -> None:
        self._factories.pop(key, None)

    def keys(self) -> list[str]:
        return sorted(self._factories)

    def create(self, key: str, options: dict[str, Any] | None = None) -> T:
        factory = self._factories.get(key)
        if factory is None:
            raise RegistryError(
                f"unknown {self.kind} provider {key!r}; registered: {self.keys()}"
            )
        try:
            return factory(**(options or {}))
        except TypeError as exc:
            raise ConfigError(
                f"bad options for {self.kind} provider {key!r}: {exc}"
            ) from exc


#: Global registries. Adapter modules import these and decorate their classes.
TTS_PROVIDERS: ProviderRegistry[Any] = ProviderRegistry("tts")
STT_PROVIDERS: ProviderRegistry[Any] = ProviderRegistry("stt")
LLM_PROVIDERS: ProviderRegistry[Any] = ProviderRegistry("llm")


def _build(
    registry: ProviderRegistry[Any],
    slot: ProviderSlotConfig | None,
    name: str | None,
    kind: str,
) -> Any:
    if slot is None:
        raise ConfigError(f"no {kind} providers are configured")
    instance_name, instance = slot.resolve(name)
    provider = registry.create(instance.type, instance.options)
    # Remember which config instance produced this object; run manifests record it so a
    # result can be traced back to the exact configuration block.
    setattr(provider, "instance_name", instance_name)
    return provider


def build_tts_provider(config: AppConfig, name: str | None = None) -> Any:
    _ensure_adapters_imported()
    return _build(TTS_PROVIDERS, config.providers.tts, name, "tts")


def build_stt_provider(config: AppConfig, name: str | None = None) -> Any:
    _ensure_adapters_imported()
    return _build(STT_PROVIDERS, config.providers.stt, name, "stt")


def build_llm_provider(config: AppConfig, name: str | None = None) -> Any:
    _ensure_adapters_imported()
    return _build(LLM_PROVIDERS, config.providers.llm, name, "llm")


_ADAPTERS_IMPORTED = False


def _ensure_adapters_imported() -> None:
    """Import adapter packages so their decorators run.

    Deliberately lazy and failure-tolerant: an adapter package whose optional dependency
    is missing (say ``httpx`` for the sidecar adapters) must not stop the mock provider
    from working, which is what keeps Milestone 1 runnable with only pydantic and PyYAML
    installed.
    """
    global _ADAPTERS_IMPORTED
    if _ADAPTERS_IMPORTED:
        return
    for module in ("cvai_tts_providers", "cvai_stt_providers", "cvai_llm_providers"):
        try:
            __import__(module)
        except ImportError:  # pragma: no cover - depends on install extras
            continue
    _ADAPTERS_IMPORTED = True
