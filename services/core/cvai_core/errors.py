"""Exception hierarchy.

One base class so callers can catch everything from this project, and specific types
where a caller realistically wants to react differently — in particular
``ProviderUnavailableError``, which the benchmark treats as "skip this candidate and keep
going" rather than "abort the run".
"""

from __future__ import annotations


class CVAIError(Exception):
    """Base for every error raised by this project."""


class ConfigError(CVAIError):
    """Malformed, missing or contradictory configuration."""


class RegistryError(CVAIError):
    """Unknown provider key, or a duplicate registration."""


class VoicePackError(CVAIError):
    """A voice pack is missing, malformed or internally inconsistent."""


class ReferenceBankError(VoicePackError):
    """The reference bank cannot satisfy a request at all."""


class ProviderUnavailableError(CVAIError):
    """A provider exists but cannot serve right now (sidecar down, weights missing).

    Distinct from a bug: the benchmark records it against the candidate and continues, so
    one offline engine does not destroy a four-hour run.
    """


class SynthesisError(CVAIError):
    """A TTS request failed."""


class UnsupportedFeatureError(CVAIError):
    """A provider was asked for something its capabilities do not claim."""
