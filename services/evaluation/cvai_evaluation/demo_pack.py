"""Build a synthetic voice pack so the pipeline can be demonstrated offline.

**Everything this produces is synthetic.** The "reference clips" are generated tones, not
a person, and the "held-out recordings" used as the listening-test anchor are the same.
The pack is called ``demo_zh`` and says so in its manifest, notes and README, so it can
never be mistaken for a real character pack. It exists for exactly one reason: to let
`make demo` and the tests exercise reference retrieval, benchmarking, blind-test
assembly and reporting on a machine with no audio data and no GPU.

A real pack is built by the Milestone 2 preprocessing pipeline from real recordings.
"""

from __future__ import annotations

import math
from pathlib import Path

from cvai_core.audio import write_wav
from cvai_core.loaders import save_model_json
from cvai_core.paths import VoicePackPaths
from cvai_core.voicepack import scaffold_voicepack
from cvai_types import (
    AudioProperties,
    CoreStyle,
    DatasetEntry,
    DatasetManifest,
    DatasetSplit,
    ProcessingStage,
    ProcessingStep,
    ReferenceBank,
    ReferenceSample,
    ReviewStatus,
    SourceInfo,
    StyleDefinition,
    TrainingSample,
    VoicePackManifest,
)

SAMPLE_RATE = 24000

#: Lead-in room tone in the synthetic "original recordings", and therefore the offset at
#: which each reference sits inside its source. Gives the null-processing control (D7)
#: something real to cut.
_RAW_LEAD_S = 0.4

#: style -> (base F0, clips, transcripts). Different pitches per style so retrieval and
#: rotation are audible in the demo rather than merely logged.
_STYLE_SPEC: dict[str, tuple[float, list[str]]] = {
    "neutral": (220.0, ["今天天气不错。", "我知道了，先这样吧。"]),
    "soft": (200.0, ["别担心，我在这里。", "慢慢来，不着急。"]),
    "teasing": (250.0, ["你猜呢？", "哦——原来是这样啊。"]),
    "serious": (190.0, ["这件事没有那么简单。", "听好了，只说一遍。"]),
    "sad": (185.0, ["……算了。", "有些话说不出口。"]),
    "happy": (265.0, ["太好了！", "我就知道你会来。"]),
    "surprised": (280.0, ["诶？真的吗？", "你怎么会在这里？"]),
    "whisper": (170.0, ["嘘，小声点。", "别让他们听见。"]),
}

_HELDOUT_LINES = [
    "这条是留作对照的真实录音占位。",
    "盲测里需要一条没人训练过的原始录音。",
    "如果生成结果和这条分不出来，就达标了。",
    "否则差距就在这里。",
]


def build_demo_voicepack(root: Path, *, overwrite: bool = True) -> VoicePackPaths:
    """Create ``voicepacks/demo_zh`` with synthetic references and a dataset."""
    paths = VoicePackPaths(Path(root))

    manifest = VoicePackManifest(
        voicepack_id=paths.root.name,
        version="0.1.0",
        character_id="demo_zh",
        display_name="Demo (synthetic)",
        language="zh-CN",
        target_sample_rate=SAMPLE_RATE,
        source=SourceInfo(
            description="SYNTHETIC. Generated tones, not recordings of any person.",
            game_or_media="none",
            extraction_tool="cvai_evaluation.demo_pack",
            license_note="No rights involved: nothing here was recorded from anyone.",
            consent_confirmed=True,
        ),
        styles=[
            StyleDefinition(name=name, core_style=CoreStyle(name))
            for name in _STYLE_SPEC
        ],
        notes=(
            "Synthetic demo pack for exercising the pipeline offline. Not a character "
            "voice, not evidence about any engine."
        ),
    )

    scaffold_voicepack(paths, manifest, overwrite=overwrite)

    references: list[ReferenceSample] = []
    for style, (base_f0, transcripts) in _STYLE_SPEC.items():
        for index, transcript in enumerate(transcripts, start=1):
            reference_id = f"{style}_{index:02d}"
            relative = f"references/{style}/{reference_id}.wav"
            duration = 3.5 + 0.7 * index
            _write_tone(paths.root / relative, base_f0 + index * 6.0, duration)

            # The "original recording" this reference was supposedly cut from: the same
            # line with lead-in room noise, a lower level and hiss over it. The pack
            # therefore has a real null-processing control (D7) to exercise — the
            # unprocessed candidate hears this, the normal candidate hears the clip
            # above, and everything else about them is identical.
            raw_relative = f"raw/{reference_id}.wav"
            _write_raw_source(
                paths.root / raw_relative,
                base_f0 + index * 6.0,
                duration,
                lead_s=_RAW_LEAD_S,
            )

            references.append(
                ReferenceSample(
                    reference_id=reference_id,
                    audio_path=relative,
                    transcript=transcript,
                    style=style,
                    core_style=CoreStyle(style),
                    audio=AudioProperties(
                        sample_rate=SAMPLE_RATE, channels=1, duration_s=duration
                    ),
                    quality_score=0.95 - 0.03 * index,
                    source_clip=raw_relative,
                    source_offset_s=_RAW_LEAD_S,
                    tags=["synthetic"],
                )
            )

    bank = ReferenceBank(
        voicepack_id=manifest.voicepack_id,
        voicepack_version=manifest.version,
        samples=references,
    )
    save_model_json(bank, paths.references_file)

    samples: list[TrainingSample] = []
    splits: list[DatasetEntry] = []
    chain = [
        ProcessingStep(
            stage=ProcessingStage.INGEST, tool="cvai_evaluation.demo_pack", tool_version="1"
        )
    ]

    # A handful of "training" clips, plus held-out ones that become the blind-test anchor.
    for style, (base_f0, transcripts) in _STYLE_SPEC.items():
        for index, transcript in enumerate(transcripts, start=1):
            sample_id = f"train_{style}_{index:02d}"
            relative = f"clean/{sample_id}.wav"
            duration = 2.8 + 0.4 * index
            _write_tone(paths.root / relative, base_f0, duration)
            samples.append(
                TrainingSample(
                    sample_id=sample_id,
                    audio_path=relative,
                    transcript=transcript,
                    audio=AudioProperties(
                        sample_rate=SAMPLE_RATE, channels=1, duration_s=duration
                    ),
                    emotion=style,
                    voice_style=style,
                    quality_score=0.9,
                    review_status=ReviewStatus.APPROVED,
                    processing_chain=chain,
                )
            )
            splits.append(DatasetEntry(sample_id=sample_id, split=DatasetSplit.TRAIN))

    for index, transcript in enumerate(_HELDOUT_LINES, start=1):
        sample_id = f"heldout_{index:02d}"
        relative = f"clean/{sample_id}.wav"
        duration = 3.0 + 0.3 * index
        _write_tone(paths.root / relative, 215.0 + index * 3.0, duration)
        samples.append(
            TrainingSample(
                sample_id=sample_id,
                audio_path=relative,
                transcript=transcript,
                audio=AudioProperties(
                    sample_rate=SAMPLE_RATE, channels=1, duration_s=duration
                ),
                emotion="neutral",
                voice_style="neutral",
                quality_score=0.97,
                review_status=ReviewStatus.APPROVED,
                processing_chain=chain,
            )
        )
        splits.append(DatasetEntry(sample_id=sample_id, split=DatasetSplit.HELDOUT))

    dataset = DatasetManifest(
        voicepack_id=manifest.voicepack_id,
        voicepack_version=manifest.version,
        samples=samples,
        splits=splits,
    )
    save_model_json(dataset, paths.dataset_file)
    return paths


def _write_raw_source(path: Path, f0: float, duration_s: float, *, lead_s: float) -> None:
    """The same line as it would have arrived: quieter, hissy, with lead-in room tone.

    Deterministic noise rather than `random`, so two runs of the demo produce byte-equal
    packs and a diff in a benchmark result means something changed in the code.
    """
    tone = _tone_samples(f0, duration_s)
    lead = int(lead_s * SAMPLE_RATE)
    tail = int(0.25 * SAMPLE_RATE)
    total = lead + len(tone) + tail

    samples: list[float] = []
    for n in range(total):
        hiss = 0.012 * (((n * 7919) % 2003) / 1001.5 - 1.0)
        voice = tone[n - lead] * 0.72 if lead <= n < lead + len(tone) else 0.0
        samples.append(voice + hiss)
    write_wav(path, samples, SAMPLE_RATE)


def _tone_samples(f0: float, duration_s: float) -> list[float]:
    """A short vowel-ish tone with a fade, so players do not click."""
    total = int(duration_s * SAMPLE_RATE)
    fade = int(0.02 * SAMPLE_RATE)
    two_pi = 2.0 * math.pi
    samples: list[float] = []
    for n in range(total):
        t = n / SAMPLE_RATE
        value = (
            math.sin(two_pi * f0 * t)
            + 0.45 * math.sin(two_pi * 2 * f0 * t)
            + 0.22 * math.sin(two_pi * 3 * f0 * t)
        ) / 1.67
        # Syllable-rate amplitude modulation, so it reads as speech-like rather than a
        # test tone when someone opens the demo rating page.
        value *= 0.55 + 0.45 * abs(math.sin(two_pi * 2.6 * t))
        if n < fade:
            value *= n / fade
        elif n > total - fade:
            value *= max(0.0, (total - n) / fade)
        samples.append(value * 0.5)
    return samples


def _write_tone(path: Path, f0: float, duration_s: float) -> None:
    write_wav(path, _tone_samples(f0, duration_s), SAMPLE_RATE)
