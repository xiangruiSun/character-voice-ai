"""Engine sidecar request handling.

No FastAPI, no engine, no GPU — all the logic that can be wrong lives in
``cvai_sidecar.handlers`` precisely so it can be tested here. The behaviour worth
pinning down is mostly about refusing to produce something that *looks* like a result:
empty audio, a silently wrong sample rate, or an arbitrary method name reflected onto a
loaded model object.
"""

from __future__ import annotations

import io
import math
import wave

import pytest
from cvai_sidecar import (
    ENGINE_VERSION_HEADER,
    SAMPLE_RATE_HEADER,
    Sidecar,
    SidecarError,
    SynthesizeRequest,
    encode_wav,
    resample_linear,
    to_float_samples,
)


class FakeEngine:
    """Stands in for a real engine; records what it was asked to do."""

    name = "fake"
    version = "0.1"

    def __init__(self, *, rate: int = 24000, samples: list[float] | None = None) -> None:
        self.rate = rate
        self.samples = samples if samples is not None else [0.1, -0.1] * 1200
        self.loaded = False
        self.calls: list[tuple[str, dict, int | None]] = []
        self.checkpoints: list[str] = []
        self.supports_checkpoints = True

    def load(self) -> None:
        self.loaded = True

    def ready(self) -> bool:
        return self.loaded

    def call(self, method, kwargs, seed):
        self.calls.append((method, dict(kwargs), seed))
        if kwargs.get("explode"):
            raise RuntimeError("engine blew up")
        if kwargs.get("bad_args"):
            raise TypeError("unexpected keyword argument 'nonsense'")
        return self.samples, self.rate

    def load_checkpoint(self, checkpoint_id: str) -> None:
        if not self.supports_checkpoints:
            raise NotImplementedError("zero-shot only")
        self.checkpoints.append(checkpoint_id)


def make(**kwargs) -> tuple[Sidecar, FakeEngine]:
    engine = FakeEngine(**kwargs)
    sidecar = Sidecar(engine, allowed_calls=["synthesize", "generate"])
    sidecar.start()
    return sidecar, engine


# --------------------------------------------------------------------------------------
# Request parsing
# --------------------------------------------------------------------------------------


def test_request_requires_a_call():
    with pytest.raises(SidecarError) as exc:
        SynthesizeRequest.parse({"kwargs": {}})
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        "not a dict",
        {"call": "x", "kwargs": "not an object"},
        {"call": "x", "seed": "not an int"},
        {"call": "x", "sample_rate": 999999},
    ],
)
def test_malformed_requests_are_client_errors(payload):
    with pytest.raises(SidecarError) as exc:
        SynthesizeRequest.parse(payload)
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def test_call_is_restricted_to_an_allow_list():
    """`call` arrives over HTTP; free dispatch onto a model object is a security hole."""
    sidecar, engine = make()
    with pytest.raises(SidecarError) as exc:
        sidecar.synthesize({"call": "__class__", "kwargs": {}})
    assert exc.value.status_code == 400
    assert not engine.calls


def test_arguments_and_seed_reach_the_engine_untouched():
    sidecar, engine = make()
    sidecar.synthesize(
        {"call": "synthesize", "kwargs": {"text": "你好", "top_p": 0.7}, "seed": 42}
    )
    method, kwargs, seed = engine.calls[0]
    assert method == "synthesize"
    assert kwargs == {"text": "你好", "top_p": 0.7}
    assert seed == 42


def test_requests_before_loading_are_refused_with_503():
    engine = FakeEngine()
    sidecar = Sidecar(engine, allowed_calls=["synthesize"])
    with pytest.raises(SidecarError) as exc:
        sidecar.synthesize({"call": "synthesize", "kwargs": {}})
    assert exc.value.status_code == 503


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def test_response_is_real_wav_with_the_rate_in_a_header():
    sidecar, _ = make(rate=32000)
    result = sidecar.synthesize({"call": "synthesize", "kwargs": {}})

    with wave.open(io.BytesIO(result.audio_wav), "rb") as handle:
        assert handle.getframerate() == 32000
        assert handle.getsampwidth() == 2
        assert handle.getnframes() > 0
    assert result.headers()[SAMPLE_RATE_HEADER] == "32000"
    assert result.headers()[ENGINE_VERSION_HEADER] == "0.1"


def test_empty_audio_is_an_error_not_a_silent_success():
    """Silence in a listening test looks like a result, which is worse than a failure."""
    sidecar, _ = make(samples=[])
    with pytest.raises(SidecarError) as exc:
        sidecar.synthesize({"call": "synthesize", "kwargs": {}})
    assert "no audio" in str(exc.value)


def test_engine_failures_become_500s_with_the_reason():
    sidecar, _ = make()
    with pytest.raises(SidecarError) as exc:
        sidecar.synthesize({"call": "synthesize", "kwargs": {"explode": True}})
    assert exc.value.status_code == 500
    assert "blew up" in str(exc.value)


def test_argument_mismatches_are_reported_as_a_version_skew():
    sidecar, _ = make()
    with pytest.raises(SidecarError) as exc:
        sidecar.synthesize({"call": "synthesize", "kwargs": {"bad_args": True}})
    assert exc.value.status_code == 400
    assert "out of sync" in str(exc.value)


def test_explicit_sample_rate_triggers_resampling():
    sidecar, _ = make(rate=24000, samples=[0.2] * 24000)
    result = sidecar.synthesize(
        {"call": "synthesize", "kwargs": {}, "sample_rate": 16000}
    )
    with wave.open(io.BytesIO(result.audio_wav), "rb") as handle:
        assert handle.getframerate() == 16000
        assert abs(handle.getnframes() - 16000) < 50


def test_no_resampling_when_the_caller_does_not_ask():
    """The benchmark leaves it unset so a 48 kHz engine is judged at 48 kHz."""
    sidecar, _ = make(rate=48000)
    result = sidecar.synthesize({"call": "synthesize", "kwargs": {}})
    assert result.sample_rate == 48000


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------


def test_checkpoint_switching_is_idempotent():
    sidecar, engine = make()
    first = sidecar.set_checkpoint({"checkpoint_id": "denia_v2"})
    second = sidecar.set_checkpoint({"checkpoint_id": "denia_v2"})
    assert first["changed"] is True
    assert second["changed"] is False
    assert engine.checkpoints == ["denia_v2"]


def test_zero_shot_engines_report_that_they_cannot_switch():
    sidecar, engine = make()
    engine.supports_checkpoints = False
    with pytest.raises(SidecarError) as exc:
        sidecar.set_checkpoint({"checkpoint_id": "x"})
    assert exc.value.status_code == 400
    assert "cannot switch checkpoints" in str(exc.value)


def test_checkpoint_requires_an_id():
    sidecar, _ = make()
    with pytest.raises(SidecarError) as exc:
        sidecar.set_checkpoint({})
    assert exc.value.status_code == 400


def test_health_reports_readiness_and_checkpoint():
    sidecar, _ = make()
    sidecar.set_checkpoint({"checkpoint_id": "denia_v3"})
    health = sidecar.health()
    assert health == {
        "engine": "fake",
        "version": "0.1",
        "ready": True,
        "checkpoint": "denia_v3",
    }


# --------------------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------------------


def test_wav_round_trip_preserves_the_signal():
    from cvai_sidecar import read_wav_file

    original = [math.sin(i / 20.0) * 0.5 for i in range(4000)]
    data = encode_wav(original, 16000)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "a.wav"
        path.write_bytes(data)
        restored, rate = read_wav_file(str(path))

    assert rate == 16000
    assert len(restored) == len(original)
    assert max(abs(a - b) for a, b in zip(original, restored)) < 1e-3


def test_samples_are_clamped_rather_than_wrapping():
    """An out-of-range sample that wraps turns a loud line into white noise."""
    data = encode_wav([5.0, -5.0], 16000)
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.readframes(2)
    import struct

    values = struct.unpack("<2h", frames)
    assert values == (32767, -32767)


def test_channel_major_output_is_downmixed():
    assert to_float_samples([[1.0, 1.0], [0.0, 0.0]]) == [0.5, 0.5]


def test_resampling_preserves_duration():
    samples = [0.0] * 1000
    assert len(resample_linear(samples, 10000, 5000)) == 500
    assert len(resample_linear(samples, 10000, 20000)) == 2000
    assert resample_linear(samples, 16000, 16000) == samples


def test_unknown_engine_output_is_rejected_clearly():
    with pytest.raises(SidecarError):
        to_float_samples(object())
