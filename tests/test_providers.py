"""Provider adapters: the mock engine, and the payload mapping of the real ones.

The real adapters cannot be run here (no GPU, no sidecars), but the interesting part of
each one *is* testable: how an engine-neutral request becomes that engine's own
parameters. Getting that mapping wrong is silent — the audio still renders, it just is
not what was asked for — so it is worth pinning down.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from cvai_core.audio import read_wav_properties
from cvai_core.errors import SynthesisError, UnsupportedFeatureError
from cvai_types import (
    AdaptationMode,
    CoreStyle,
    EmotionVector,
    ReferenceSelection,
    SpeakingRate,
    StyleControls,
    TTSRequest,
    VolumeStyle,
)
from cvai_tts_providers import (
    CosyVoiceProvider,
    FishSpeechProvider,
    GPTSoVITSProvider,
    IndexTTSProvider,
    MockTTSProvider,
    QwenTTSProvider,
    VoxCPMProvider,
)
from cvai_tts_providers.fish_speech import PROMPT_TOKEN_KEY


def _reference(**overrides) -> ReferenceSelection:
    data = {
        "reference_id": "teasing_01",
        "audio_path": "/packs/demo/references/teasing/teasing_01.wav",
        "transcript": "你猜呢？",
        "style": "teasing",
        "core_style": CoreStyle.TEASING,
    }
    data.update(overrides)
    return ReferenceSelection(**data)


def _request(**overrides) -> TTSRequest:
    data = {
        "request_id": "req-1",
        "text": "今天外面风有点大。",
        "controls": StyleControls(emotion="teasing", speaking_rate=SpeakingRate.SLOW),
        "reference": _reference(),
    }
    data.update(overrides)
    return TTSRequest(**data)


# --------------------------------------------------------------------------------------
# Mock engine
# --------------------------------------------------------------------------------------


def test_mock_writes_real_audio_with_plausible_duration(tmp_path: Path):
    provider = MockTTSProvider(sample_rate=16000)
    out = tmp_path / "a.wav"
    result = asyncio.run(provider.synthesize(_request(), out))

    assert out.is_file()
    probed = read_wav_properties(out)
    assert probed.sample_rate == 16000
    assert probed.duration_s == pytest.approx(result.duration_s, rel=0.02)
    assert 0.5 < result.duration_s < 20.0
    assert result.engine == "mock"
    assert result.reference_id == "teasing_01"
    assert result.latency_ms >= 0.0


def test_mock_is_deterministic_for_the_same_request(tmp_path: Path):
    provider = MockTTSProvider()
    first = tmp_path / "1.wav"
    second = tmp_path / "2.wav"
    asyncio.run(provider.synthesize(_request(), first))
    asyncio.run(provider.synthesize(_request(), second))
    assert first.read_bytes() == second.read_bytes()


def test_mock_output_changes_with_the_reference_clip(tmp_path: Path):
    """Reference rotation must be audible, otherwise it is untestable end to end."""
    provider = MockTTSProvider()
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    asyncio.run(provider.synthesize(_request(), a))
    asyncio.run(
        provider.synthesize(
            _request(reference=_reference(reference_id="teasing_02")), b
        )
    )
    assert a.read_bytes() != b.read_bytes()


def test_mock_speaking_rate_changes_duration(tmp_path: Path):
    provider = MockTTSProvider()
    slow = asyncio.run(
        provider.synthesize(
            _request(controls=StyleControls(speaking_rate=SpeakingRate.VERY_SLOW)),
            tmp_path / "slow.wav",
        )
    )
    fast = asyncio.run(
        provider.synthesize(
            _request(controls=StyleControls(speaking_rate=SpeakingRate.VERY_FAST)),
            tmp_path / "fast.wav",
        )
    )
    assert slow.duration_s > fast.duration_s


def test_mock_reports_controls_it_cannot_honour(tmp_path: Path):
    provider = MockTTSProvider(supports_instruct=False, supports_emotion_vector=False)
    request = _request(
        controls=StyleControls(
            emotion="teasing",
            instruct="轻声调侃",
            emotion_vector=EmotionVector(happy=0.6),
        )
    )
    result = asyncio.run(provider.synthesize(request, tmp_path / "a.wav"))
    assert set(result.dropped_controls) == {"instruct", "emotion_vector"}


def test_mock_failure_injection_is_deterministic(tmp_path: Path):
    provider = MockTTSProvider(failure_rate=1.0)
    with pytest.raises(SynthesisError):
        asyncio.run(provider.synthesize(_request(), tmp_path / "a.wav"))


def test_streaming_is_refused_rather_than_silently_unsupported(tmp_path: Path):
    provider = MockTTSProvider()

    async def drain():
        async for _ in provider.synthesize_stream(_request()):
            pass

    with pytest.raises(UnsupportedFeatureError):
        asyncio.run(drain())


# --------------------------------------------------------------------------------------
# GPT-SoVITS payload
# --------------------------------------------------------------------------------------


def test_gpt_sovits_payload_matches_the_documented_api():
    provider = GPTSoVITSProvider(base_url="http://localhost:9880")
    payload = provider._build_payload(_request(), seed=99)

    assert payload["text"] == "今天外面风有点大。"
    assert payload["text_lang"] == "zh" and payload["prompt_lang"] == "zh"
    # The reference transcript is how this engine gets any style at all.
    assert payload["prompt_text"] == "你猜呢？"
    assert payload["ref_audio_path"].endswith("teasing_01.wav")
    assert payload["seed"] == 99
    # SLOW == 0.85 in the shared rate table.
    assert payload["speed_factor"] == pytest.approx(0.85)
    for key in ("top_k", "top_p", "temperature", "text_split_method", "sample_steps"):
        assert key in payload


def test_gpt_sovits_request_params_override_adapter_defaults():
    provider = GPTSoVITSProvider()
    payload = provider._build_payload(
        _request(engine_params={"top_k": 5, "temperature": 0.6}), seed=1
    )
    assert payload["top_k"] == 5
    assert payload["temperature"] == 0.6


def test_gpt_sovits_checkpoint_id_must_name_both_stages():
    provider = GPTSoVITSProvider()
    with pytest.raises(SynthesisError):
        asyncio.run(provider.load_checkpoint("only-one-path"))


def test_gpt_sovits_refuses_to_synthesize_without_a_reference(tmp_path: Path):
    provider = GPTSoVITSProvider()
    request = TTSRequest(
        request_id="r",
        text="你好",
        adaptation_mode=AdaptationMode.FINETUNED,
        checkpoint_id="a|b",
    )
    with pytest.raises(SynthesisError):
        asyncio.run(provider.synthesize(request, tmp_path / "x.wav"))


# --------------------------------------------------------------------------------------
# Qwen3-TTS / CosyVoice / VoxCPM instruct path
# --------------------------------------------------------------------------------------


def test_qwen_sends_reference_text_and_a_reusable_prompt_key():
    provider = QwenTTSProvider()
    kwargs = provider.native_kwargs(_request(), seed=5)
    assert kwargs["ref_text"] == "你猜呢？"
    assert kwargs["voice_clone_prompt_key"] == "teasing_01"
    assert kwargs["language"] == "Chinese"
    assert "instruct" in kwargs


def test_qwen_requires_reference_text_unless_x_vector_only():
    assert QwenTTSProvider().capabilities().requires_reference_text is True
    assert (
        QwenTTSProvider(x_vector_only_mode=True).capabilities().requires_reference_text
        is False
    )


def test_instruct_is_built_in_chinese_from_the_neutral_controls():
    from cvai_tts_providers.qwen3_tts import build_instruct

    instruct = build_instruct(
        _request(
            controls=StyleControls(
                emotion="soft_teasing",
                emotion_intensity=0.8,
                speaking_rate=SpeakingRate.SLIGHTLY_SLOW,
                volume_style=VolumeStyle.SOFT,
            )
        )
    )
    assert "语速略慢" in instruct
    assert "音量轻柔" in instruct
    assert "明显" in instruct  # high intensity


def test_planner_instruct_wins_over_the_generated_one():
    provider = QwenTTSProvider()
    kwargs = provider.native_kwargs(
        _request(controls=StyleControls(emotion="teasing", instruct="压低声音，慢慢说")),
        seed=1,
    )
    assert kwargs["instruct"] == "压低声音，慢慢说"


def test_cosyvoice_maps_onto_prompt_wav_and_speed():
    kwargs = CosyVoiceProvider().native_kwargs(_request(), seed=3)
    assert kwargs["prompt_wav"].endswith("teasing_01.wav")
    assert kwargs["prompt_text"] == "你猜呢？"
    assert kwargs["speed"] == pytest.approx(0.85)


def test_voxcpm_declares_48k_and_requires_the_transcript():
    caps = VoxCPMProvider().capabilities()
    assert caps.native_sample_rate == 48000
    assert caps.requires_reference_text is True
    assert AdaptationMode.LORA in caps.supported_adaptation_modes


# --------------------------------------------------------------------------------------
# IndexTTS emotion vector
# --------------------------------------------------------------------------------------


def test_index_tts_derives_an_emotion_vector_and_inverts_the_speed_factor():
    provider = IndexTTSProvider()
    kwargs = provider.native_kwargs(_request(), seed=1)
    assert len(kwargs["emo_vector"]) == 8
    # SLOW == 0.85 speed factor, and duration_factor runs the other way.
    assert kwargs["duration_factor"] == pytest.approx(1 / 0.85, rel=1e-3)
    assert kwargs["lang"] == "ZH"


def test_index_tts_passes_an_explicit_vector_through_unchanged():
    provider = IndexTTSProvider()
    vector = EmotionVector(sad=0.9, calm=0.2)
    kwargs = provider.native_kwargs(
        _request(controls=StyleControls(emotion="sad", emotion_vector=vector)), seed=1
    )
    assert kwargs["emo_vector"] == vector.as_list()


def test_index_tts_is_zero_shot_only():
    caps = IndexTTSProvider().capabilities()
    assert caps.supported_adaptation_modes == [AdaptationMode.ZERO_SHOT]
    # Licence is not OSI, so commercial use is unknown rather than assumed.
    assert caps.commercial_use is None


# --------------------------------------------------------------------------------------
# Fish Speech precomputed tokens
# --------------------------------------------------------------------------------------


def test_fish_speech_demands_precomputed_prompt_tokens():
    provider = FishSpeechProvider(require_precomputed_tokens=True)
    with pytest.raises(SynthesisError) as exc:
        provider.native_kwargs(_request(), seed=1)
    assert PROMPT_TOKEN_KEY in str(exc.value)


def test_fish_speech_uses_tokens_when_the_bank_has_them():
    provider = FishSpeechProvider()
    reference = _reference(precomputed={PROMPT_TOKEN_KEY: "references/teasing/t1.npy"})
    kwargs = provider.native_kwargs(_request(reference=reference), seed=1)
    assert kwargs["prompt_tokens"] == "references/teasing/t1.npy"
    assert kwargs["prompt_text"] == "你猜呢？"


def test_fish_speech_can_fall_back_to_raw_audio_when_explicitly_allowed():
    provider = FishSpeechProvider(require_precomputed_tokens=False)
    kwargs = provider.native_kwargs(_request(), seed=1)
    assert "reference_audio" in kwargs


# --------------------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------------------


def test_derived_seeds_are_stable_and_content_sensitive():
    from cvai_tts_providers import derive_seed

    a = derive_seed(_request(), salt="v1")
    b = derive_seed(_request(), salt="v1")
    c = derive_seed(_request(text="另一句话。"), salt="v1")
    assert a == b
    assert a != c
    assert derive_seed(_request(seed=1234)) == 1234
