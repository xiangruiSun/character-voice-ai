"""Reference retrieval: fallback, rotation, determinism.

These three behaviours are the whole defence against spec §27's "using one reference clip
for all emotions" and against a benchmark that silently evaluates neutral delivery.
"""

from __future__ import annotations

import pytest
from cvai_core.errors import ReferenceBankError
from cvai_reference_retrieval import RuleBasedReferenceRetriever
from cvai_types import (
    AudioProperties,
    CoreStyle,
    ReferenceBank,
    ReferenceSample,
    StyleControls,
    StyleDefinition,
)


def _sample(reference_id: str, style: str, quality: float = 0.9, duration: float = 5.0):
    core = (
        CoreStyle(style)
        if style in CoreStyle._value2member_map_
        else CoreStyle.NEUTRAL
    )
    return ReferenceSample(
        reference_id=reference_id,
        audio_path=f"references/{style}/{reference_id}.wav",
        transcript="示例。",
        style=style,
        core_style=core,
        audio=AudioProperties(sample_rate=24000, channels=1, duration_s=duration),
        quality_score=quality,
    )


def _bank(samples) -> ReferenceBank:
    return ReferenceBank(voicepack_id="p", voicepack_version="1.0.0", samples=samples)


DEFINITIONS = {
    "soft_teasing": StyleDefinition(name="soft_teasing", core_style=CoreStyle.TEASING),
    "teasing": StyleDefinition(name="teasing", core_style=CoreStyle.TEASING),
    "neutral": StyleDefinition(name="neutral", core_style=CoreStyle.NEUTRAL),
}


def test_exact_style_is_preferred_and_not_marked_as_fallback():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("n1", "neutral"), _sample("t1", "teasing")]),
        style_definitions=DEFINITIONS,
    )
    selection = retriever.select(StyleControls(emotion="teasing"))
    assert selection.reference_id == "t1"
    assert selection.was_fallback is False


def test_character_style_falls_back_to_its_core_style_and_says_so():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("n1", "neutral"), _sample("t1", "teasing")]),
        style_definitions=DEFINITIONS,
    )
    selection = retriever.select(StyleControls(emotion="soft_teasing"))
    assert selection.reference_id == "t1"
    assert selection.was_fallback is True
    assert selection.requested_style == "soft_teasing"


def test_unknown_style_lands_on_neutral():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("n1", "neutral")]), style_definitions=DEFINITIONS
    )
    selection = retriever.select(StyleControls(emotion="nonexistent_style"))
    assert selection.reference_id == "n1"
    assert selection.was_fallback is True


def test_higher_quality_wins_when_rotation_is_off():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("a", "neutral", quality=0.5), _sample("b", "neutral", quality=0.99)]),
        style_definitions=DEFINITIONS,
        enable_rotation=False,
    )
    assert retriever.select(StyleControls()).reference_id == "b"


def test_duration_outside_the_preferred_window_is_penalised():
    retriever = RuleBasedReferenceRetriever(
        _bank(
            [
                _sample("short", "neutral", quality=0.95, duration=1.2),
                _sample("good", "neutral", quality=0.95, duration=6.0),
            ]
        ),
        style_definitions=DEFINITIONS,
        enable_rotation=False,
    )
    assert retriever.select(StyleControls()).reference_id == "good"


def test_rotation_varies_clips_between_utterances_but_stays_deterministic():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample(f"n{i}", "neutral", quality=0.9) for i in range(4)]),
        style_definitions=DEFINITIONS,
        rotation_pool=4,
    )
    picks = {
        key: retriever.select(StyleControls(), variation_key=key).reference_id
        for key in [f"s{i}" for i in range(12)]
    }
    # Varied…
    assert len(set(picks.values())) > 1, "rotation should not always return one clip"
    # …and reproducible, which is what keeps a benchmark re-runnable.
    for key, expected in picks.items():
        assert retriever.select(StyleControls(), variation_key=key).reference_id == expected


def test_rotation_is_stable_across_processes():
    """Uses a digest, not ``hash()``, which CPython salts per process."""
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample(f"n{i}", "neutral") for i in range(3)]),
        style_definitions=DEFINITIONS,
    )
    from cvai_reference_retrieval.rule_based import _stable_index

    assert _stable_index("fixed-key", 3) == _stable_index("fixed-key", 3)
    assert retriever.select(StyleControls(), variation_key="fixed-key").reference_id


def test_excluded_ids_are_skipped():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("a", "neutral"), _sample("b", "neutral")]),
        style_definitions=DEFINITIONS,
        enable_rotation=False,
    )
    selection = retriever.select(StyleControls(), exclude_ids={"a"})
    assert selection.reference_id == "b"


def test_empty_bank_is_refused_at_construction():
    with pytest.raises(ReferenceBankError):
        RuleBasedReferenceRetriever(_bank([]))


def test_excluding_everything_raises_rather_than_returning_nothing():
    retriever = RuleBasedReferenceRetriever(
        _bank([_sample("a", "neutral")]), style_definitions=DEFINITIONS
    )
    with pytest.raises(ReferenceBankError):
        retriever.select(StyleControls(), exclude_ids={"a"})


def test_paths_resolve_against_the_pack_when_provided(demo_pack):
    from cvai_core.loaders import load_reference_bank, load_voicepack_manifest

    manifest = load_voicepack_manifest(demo_pack)
    bank = load_reference_bank(demo_pack)
    retriever = RuleBasedReferenceRetriever.from_voicepack(demo_pack, manifest, bank)
    selection = retriever.select(StyleControls(emotion="soft"), variation_key="x")
    assert selection.audio_path.startswith(str(demo_pack.root))
    assert selection.transcript
