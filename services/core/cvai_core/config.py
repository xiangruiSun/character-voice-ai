"""Configuration system.

Spec §23: "use configuration files for model paths and character profiles". The loader
supports three layers, applied in order:

1. one or more YAML files (later files override earlier ones, deep-merged);
2. ``${VAR}`` / ``${VAR:-default}`` interpolation against the process environment;
3. ``CVAI__section__key=value`` environment overrides, which is how a container or a
   one-off benchmark run redirects a sidecar URL without editing a file.

Secrets are never written into YAML: an API key is referenced as ``${OPENAI_API_KEY}``
and resolved at load time. A missing variable without a default is a load-time error, not
a runtime surprise three minutes into a benchmark.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final

import yaml
from cvai_types import CVAIModel, Slug, sha256_text
from pydantic import Field, model_validator

from .errors import ConfigError

ENV_PREFIX: Final = "CVAI__"
ENV_SEPARATOR: Final = "__"

_INTERPOLATION_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class ProviderInstanceConfig(CVAIModel):
    """One configured provider instance.

    ``type`` is the registry key (which adapter class), the instance name is the key in
    the parent mapping (which configuration of it). Keeping those separate is what allows
    two GPT-SoVITS sidecars — say v4 and v2Pro — to be benchmarked side by side.
    """

    type: str = Field(min_length=1, max_length=60)
    enabled: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class ProviderSlotConfig(CVAIModel):
    """All instances for one provider kind, plus which one is active by default."""

    active: str = Field(min_length=1, max_length=60)
    instances: dict[str, ProviderInstanceConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _active_exists(self) -> "ProviderSlotConfig":
        if self.active not in self.instances:
            raise ConfigError(
                f"active provider {self.active!r} is not defined; "
                f"known instances: {sorted(self.instances)}"
            )
        return self

    def resolve(self, name: str | None = None) -> tuple[str, ProviderInstanceConfig]:
        key = name or self.active
        instance = self.instances.get(key)
        if instance is None:
            raise ConfigError(
                f"unknown provider instance {key!r}; known: {sorted(self.instances)}"
            )
        if not instance.enabled:
            raise ConfigError(f"provider instance {key!r} is disabled in config")
        return key, instance


class ProvidersConfig(CVAIModel):
    tts: ProviderSlotConfig
    stt: ProviderSlotConfig | None = None
    llm: ProviderSlotConfig | None = None


class PathsConfig(CVAIModel):
    """All roots are relative to the repository root unless absolute."""

    voicepacks: str = "voicepacks"
    characters: str = "characters/profiles"
    runs: str = "runs"
    models: str = "models"


class BenchmarkDefaults(CVAIModel):
    base_seed: int = Field(default=20260921, ge=0)
    output_sample_rate: int | None = Field(default=None, ge=8000, le=48000)
    #: Generations per sentence per candidate. >1 exposes run-to-run instability, which
    #: matters: a model that is excellent on average but occasionally mangles a line is
    #: worse in production than a steady one.
    repeats: int = Field(default=1, ge=1, le=10)
    #: Continue past a candidate whose sidecar is unavailable instead of aborting.
    skip_unavailable: bool = True


class AppConfig(CVAIModel):
    """Root configuration object."""

    default_character: Slug = "denia_cn"
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR)$")
    paths: PathsConfig = Field(default_factory=PathsConfig)
    providers: ProvidersConfig
    benchmark: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)

    #: Hash of the merged config, recorded in every run manifest (spec §23).
    config_hash: str = ""

    def with_hash(self) -> "AppConfig":
        payload = self.model_dump(mode="json", exclude={"config_hash"})
        digest = sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        return self.model_copy(update={"config_hash": digest})


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; overlay wins. Lists replace rather than concatenate.

    List-replace is deliberate: concatenating a list of benchmark candidates across two
    config files produces duplicates that are painful to debug.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def interpolate(value: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in every string in a structure."""
    environment = env if env is not None else dict(os.environ)

    def _expand(text: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in environment:
                return environment[name]
            if default is not None:
                return default
            raise ConfigError(
                f"config references ${{{name}}} but it is not set and has no default; "
                f"use ${{{name}:-fallback}} if it is optional"
            )

        return _INTERPOLATION_RE.sub(_sub, text)

    if isinstance(value, str):
        return _expand(value)
    if isinstance(value, dict):
        return {k: interpolate(v, environment) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, environment) for v in value]
    return value


def _coerce_scalar(raw: str) -> Any:
    """Turn an env-var string into bool/int/float/None/JSON where it clearly is one."""
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.strip().startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def env_overrides(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Collect ``CVAI__a__b=value`` into a nested dict."""
    environment = env if env is not None else dict(os.environ)
    result: dict[str, Any] = {}
    for key, raw in environment.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].split(ENV_SEPARATOR)
        if not path or any(not part for part in path):
            raise ConfigError(f"malformed config override variable {key!r}")
        cursor = result
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigError(
                    f"override {key!r} conflicts with an earlier scalar override"
                )
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return data


def load_config(
    paths: list[Path] | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    use_env_overrides: bool = True,
) -> AppConfig:
    """Load, merge, interpolate and validate configuration.

    ``paths`` defaults to ``configs/app.yaml`` under the repository root.
    """
    from .paths import repo_root  # local import: paths imports errors, not config

    files = paths if paths is not None else [repo_root() / "configs" / "app.yaml"]

    merged: dict[str, Any] = {}
    for file in files:
        merged = _deep_merge(merged, load_yaml(Path(file)))

    if use_env_overrides:
        merged = _deep_merge(merged, env_overrides(env))
    if overrides:
        merged = _deep_merge(merged, overrides)

    merged = interpolate(merged, env)

    try:
        config = AppConfig.model_validate(merged)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration: {exc}") from exc
    return config.with_hash()


def load_provider_config(path: Path) -> dict[str, Any]:
    """Load a standalone provider config file (``configs/providers/*.yaml``)."""
    return interpolate(load_yaml(Path(path)))
