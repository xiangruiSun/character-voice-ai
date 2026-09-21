"""Schema behaviour that the rest of the system relies on."""

from __future__ import annotations

import pytest
from cvai_types import (
    AdaptationMode,
    AudioProperties,
    CharacterSpeechPlan,
    CoreStyle,
    DatasetEntry,
    DatasetManifest,
    DatasetSplit,
    EmotionVector,
    ProviderCapabilities,
    ReferenceBank,
    ReferenceSample,
    ReferenceSelection,
    RejectionReason,
    ReviewStatus,
    SpeakingRate,
    StyleControls,
    StyleDefinition,
    TrainingSample,
    TTSRequest,
    VolumeStyle,
    resolve_style_chain,
)
from pydantic import ValidationError


def _audio(duration: float = 4.0) -> AudioProperties:
    return AudioProperties(sample_rate=24000, channels=1, duration_s=duration)


# --------------------------------------------------------------------------------------
# Language and paths
# --------------------------------------------------------------------------------------


def test_language_is_locked_to_chinese():
    """V1 is zh-CN only (spec §25). Enforced in the schema, not by convention."""
    with pytest.raises(ValidationError):
        TrainingSample(
            sample_id="x",
            audio_path="clean/x.wav",
            transcript="你好",
            language="ja-JP",
            audio=_audio(),
        )


@pytest.mark.parametrize("bad", ["/abs/x.wav", "../escape.wav", "clean\\x.wav"])
def test_manifest_paths_must_stay_inside_the_pack(bad: str):
    with pytest.raises(ValidationError):
        TrainingSample(
            sample_id="x", audio_path=bad, transcript="你好", audio=_audio()
        )


def test_training_sample_may_not_point_into_raw():
    """raw/ is immutable input; anything approved has at least been segmented."""
    with pytest.raises(ValidationError) as exc:
        TrainingSample(
            sample_id="x", audio_path="raw/original.wav", transcript="你好", audio=_audio()
        )
    assert "raw" in str(exc.value)


# --------------------------------------------------------------------------------------
# Review status consistency
# --------------------------------------------------------------------------------------


def test_rejected_sample_requires_a_reason():
    with pytest.raises(ValidationError):
        TrainingSample(
            sample_id="x",
            audio_path="clean/x.wav",
            transcript="你好",
            audio=_audio(),
            review_status=ReviewStatus.REJECTED,
        )


def test_reason_without_rejection_is_rejected():
    with pytest.raises(ValidationError):
        TrainingSample(
            sample_id="x",
            audio_path="clean/x.wav",
            transcript="你好",
            audio=_audio(),
            review_status=ReviewStatus.APPROVED,
            rejection_reason=RejectionReason.OTHER_SPEAKER,
        )


# --------------------------------------------------------------------------------------
# Style controls
# --------------------------------------------------------------------------------------


def test_reference_style_defaults_to_emotion_but_can_be_decoupled():
    following = StyleControls(emotion="soft_teasing")
    assert following.effective_reference_style == "soft_teasing"

    decoupled = StyleControls(emotion="embarrassed", reference_style="soft_teasing")
    assert decoupled.effective_reference_style == "soft_teasing"


def test_speed_factor_prefers_explicit_duration_factor():
    controls = StyleControls(speaking_rate=SpeakingRate.FAST)
    assert controls.speed_factor() == pytest.approx(1.15)
    assert StyleControls(
        speaking_rate=SpeakingRate.FAST, duration_factor=0.9
    ).speed_factor() == pytest.approx(0.9)


def test_style_fallback_chain_walks_to_core_then_neutral():
    definitions = {
        "soft_teasing": StyleDefinition(
            name="soft_teasing", core_style=CoreStyle.TEASING
        )
    }
    assert resolve_style_chain("soft_teasing", definitions) == [
        "soft_teasing",
        "teasing",
        "neutral",
    ]
    assert resolve_style_chain("teasing", {}) == ["teasing", "neutral"]
    assert resolve_style_chain("neutral", {}) == ["neutral"]


def test_emotion_vector_order_matches_the_engine_contract():
    vector = EmotionVector.from_core_style(CoreStyle.ANGRY, 1.0)
    # happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
    assert vector.as_list() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert EmotionVector().is_empty()


def test_speech_plan_projects_onto_controls():
    plan = CharacterSpeechPlan(
        text="谁在等你了……只是刚好没什么事情而已。",
        emotion="embarrassed_teasing",
        emotion_intensity=0.4,
        speaking_rate=SpeakingRate.SLIGHTLY_SLOW,
        volume_style=VolumeStyle.SOFT,
        reference_style="soft_teasing",
    )
    controls = plan.to_style_controls()
    assert controls.emotion == "embarrassed_teasing"
    assert controls.effective_reference_style == "soft_teasing"
    assert controls.speed_factor() == pytest.approx(0.93)


# --------------------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------------------


def test_capabilities_report_controls_the_engine_will_ignore():
    caps = ProviderCapabilities(
        engine="e", supports_instruct=False, supports_emotion_vector=False
    )
    controls = StyleControls(
        instruct="轻声调侃", emotion_vector=EmotionVector(happy=0.5)
    )
    assert set(caps.unsupported_controls(controls)) == {"instruct", "emotion_vector"}

    permissive = ProviderCapabilities(
        engine="e", supports_instruct=True, supports_emotion_vector=True
    )
    assert permissive.unsupported_controls(controls) == []


def test_zero_shot_request_requires_a_reference():
    with pytest.raises(ValidationError):
        TTSRequest(request_id="r", text="你好", adaptation_mode=AdaptationMode.ZERO_SHOT)

    # A fine-tuned voice carries the speaker in the weights, so no reference is needed.
    TTSRequest(
        request_id="r",
        text="你好",
        adaptation_mode=AdaptationMode.FINETUNED,
        checkpoint_id="ckpt",
    )


# --------------------------------------------------------------------------------------
# Reference bank
# --------------------------------------------------------------------------------------


def _reference(reference_id: str, style: str, quality: float = 0.9, duration: float = 4.0):
    return ReferenceSample(
        reference_id=reference_id,
        audio_path=f"references/{style}/{reference_id}.wav",
        transcript="示例台词。",
        style=style,
        core_style=CoreStyle(style) if style in CoreStyle._value2member_map_ else CoreStyle.NEUTRAL,
        audio=_audio(duration),
        quality_score=quality,
    )


def test_reference_bank_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        ReferenceBank(
            voicepack_id="p",
            voicepack_version="1.0.0",
            samples=[_reference("a", "neutral"), _reference("a", "soft")],
        )


def test_reference_bank_resolves_through_the_fallback_chain():
    bank = ReferenceBank(
        voicepack_id="p",
        voicepack_version="1.0.0",
        samples=[_reference("n1", "neutral"), _reference("t1", "teasing")],
    )
    definitions = {
        "soft_teasing": StyleDefinition(name="soft_teasing", core_style=CoreStyle.TEASING)
    }
    resolved = bank.resolve("soft_teasing", definitions)
    assert [r.reference_id for r in resolved] == ["t1"]
    assert bank.coverage() == {"neutral": 1, "teasing": 1}


@pytest.mark.parametrize("duration", [0.4, 45.0])
def test_reference_duration_bounds_are_enforced(duration: float):
    with pytest.raises(ValidationError):
        _reference("x", "neutral", duration=duration)


# --------------------------------------------------------------------------------------
# Dataset manifest
# --------------------------------------------------------------------------------------


def test_dataset_reports_minutes_per_style():
    samples = [
        TrainingSample(
            sample_id=f"s{i}",
            audio_path=f"clean/s{i}.wav",
            transcript="你好",
            audio=_audio(60.0),
            voice_style=style,
            review_status=ReviewStatus.APPROVED,
        )
        for i, style in enumerate(["neutral", "neutral", "teasing"])
    ]
    dataset = DatasetManifest(
        voicepack_id="p", voicepack_version="1.0.0", samples=samples
    )
    assert dataset.minutes_by_style() == {"neutral": 2.0, "teasing": 1.0}
    assert dataset.total_seconds() == pytest.approx(180.0)


def test_dataset_splits_must_reference_known_samples():
    with pytest.raises(ValidationError):
        DatasetManifest(
            voicepack_id="p",
            voicepack_version="1.0.0",
            samples=[],
            splits=[DatasetEntry(sample_id="ghost", split=DatasetSplit.TRAIN)],
        )


def test_reference_selection_carries_the_transcript():
    """Decision D3: several engines require it and cannot go back to the bank."""
    selection = ReferenceSelection(
        reference_id="t1",
        audio_path="/tmp/t1.wav",
        transcript="你猜呢？",
        style="teasing",
        core_style=CoreStyle.TEASING,
    )
    assert selection.transcript
