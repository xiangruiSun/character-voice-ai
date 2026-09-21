"""Configuration layering and provider construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cvai_core.config import (
    AppConfig,
    env_overrides,
    interpolate,
    load_config,
)
from cvai_core.errors import ConfigError, RegistryError
from cvai_core.registry import ProviderRegistry, build_tts_provider

BASE = {
    "default_character": "demo_zh",
    "providers": {
        "tts": {
            "active": "mock",
            "instances": {"mock": {"type": "mock", "options": {"sample_rate": 16000}}},
        }
    },
}


def _write(tmp_path: Path, data: dict, name: str = "app.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_layers_merge_deeply_with_later_files_winning(tmp_path: Path):
    base = _write(tmp_path, BASE)
    overlay = _write(
        tmp_path,
        {"providers": {"tts": {"instances": {"mock": {"options": {"sample_rate": 48000}}}}}},
        "overlay.yaml",
    )
    config = load_config([base, overlay], use_env_overrides=False)
    assert config.providers.tts.instances["mock"].options["sample_rate"] == 48000
    # Untouched keys survive the merge.
    assert config.providers.tts.instances["mock"].type == "mock"


def test_env_overrides_are_typed(tmp_path: Path):
    base = _write(tmp_path, BASE)
    env = {
        "CVAI__log_level": "DEBUG",
        "CVAI__benchmark__repeats": "3",
        "CVAI__benchmark__skip_unavailable": "false",
    }
    config = load_config([base], env=env)
    assert config.log_level == "DEBUG"
    assert config.benchmark.repeats == 3
    assert config.benchmark.skip_unavailable is False


def test_env_override_parsing():
    parsed = env_overrides({"CVAI__a__b": "1", "CVAI__a__c": "x", "OTHER": "ignored"})
    assert parsed == {"a": {"b": 1, "c": "x"}}


def test_interpolation_uses_defaults_and_fails_loudly_without_them():
    assert interpolate("${MISSING:-fallback}", {}) == "fallback"
    assert interpolate({"k": "${SET}"}, {"SET": "v"}) == {"k": "v"}
    with pytest.raises(ConfigError):
        interpolate("${REQUIRED}", {})


def test_active_provider_must_exist(tmp_path: Path):
    broken = dict(BASE)
    broken["providers"] = {
        "tts": {"active": "ghost", "instances": {"mock": {"type": "mock"}}}
    }
    path = _write(tmp_path, broken, "broken.yaml")
    with pytest.raises(ConfigError):
        load_config([path], use_env_overrides=False)


def test_config_hash_is_stable_and_content_sensitive(tmp_path: Path):
    base = _write(tmp_path, BASE)
    first = load_config([base], use_env_overrides=False)
    second = load_config([base], use_env_overrides=False)
    assert first.config_hash == second.config_hash

    changed = load_config([base], overrides={"log_level": "DEBUG"}, use_env_overrides=False)
    assert changed.config_hash != first.config_hash


def test_registry_rejects_duplicate_keys_and_unknown_lookups():
    registry: ProviderRegistry[str] = ProviderRegistry("test")
    registry.register_instance("a", lambda: "A")
    with pytest.raises(RegistryError):
        registry.register_instance("a", lambda: "A2")
    with pytest.raises(RegistryError):
        registry.create("missing")
    assert registry.create("a") == "A"
    assert registry.keys() == ["a"]


def test_registry_reports_bad_options_as_config_errors():
    registry: ProviderRegistry[str] = ProviderRegistry("test")
    registry.register_instance("a", lambda known=1: "A")
    with pytest.raises(ConfigError):
        registry.create("a", {"unknown_option": 2})


def test_build_tts_provider_from_the_real_config(app_config: AppConfig):
    provider = build_tts_provider(app_config)
    assert provider.engine == "mock"
    assert provider.instance_name == "mock"
    assert provider.capabilities().engine == "mock"


def test_named_instances_can_be_built(app_config: AppConfig):
    """The benchmark addresses engines by instance name, not by the active default."""
    provider = build_tts_provider(app_config, "gpt_sovits")
    caps = provider.capabilities()
    assert caps.engine == "gpt_sovits"
    assert caps.supports_hot_checkpoint_swap is True
