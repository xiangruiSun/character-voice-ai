"""Build a voice pack's Reference Bank from a folder of ``.wav`` + ``.lab`` pairs.

The AI Hobbyist game datasets ship every voice line as ``<name>.wav`` with its exact
transcript in ``<name>.lab``. Those transcripts are the game's own script, so the
ASR/review stages of ``cvai-prep`` have nothing to add for reference selection; this
script goes straight to a usable bank for zero-shot synthesis.

Styles are assigned from punctuation only — ``……`` → soft, ``！`` → excited, otherwise
neutral. That is a coarse first pass meant to stop every line being performed from one
clip (spec §27); relabel by ear in ``metadata/references.json`` when it matters.

    python scripts/import_lab_pack.py C:/datasets/cartethyia cartethyia_cn \
        --display-name "卡提希娅"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path

from cvai_core.paths import VoicePackPaths, repo_root
from cvai_core.voicepack import scaffold_voicepack
from cvai_types import (
    AudioProperties,
    CoreStyle,
    ReferenceBank,
    ReferenceSample,
    SourceInfo,
    StyleDefinition,
    VoicePackManifest,
)

#: GPT-SoVITS rejects prompts outside 3-10 s; keep a margin on both ends.
MIN_S, MAX_S, IDEAL_S = 3.5, 9.5, 6.0

STYLES = {
    "neutral": ("neutral", "平常语气，叙述与回答"),
    "soft": ("soft", "迟疑、轻声，带停顿"),
    "excited": ("excited", "急促、提高音量"),
}


def style_of(text: str) -> str:
    if text.endswith("！") or text.endswith("!"):
        return "excited"
    if "……" in text:
        return "soft"
    return "neutral"


def duration(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        return handle.getnframes() / rate, rate, handle.getnchannels()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="folder of .wav + .lab pairs")
    parser.add_argument("voicepack_id")
    parser.add_argument("--character", help="character_id (default: voicepack_id)")
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--per-style", type=int, default=8)
    args = parser.parse_args(argv)

    paths = VoicePackPaths(repo_root() / "voicepacks" / args.voicepack_id)
    manifest = VoicePackManifest(
        voicepack_id=args.voicepack_id,
        character_id=args.character or args.voicepack_id,
        display_name=args.display_name,
        target_sample_rate=44100,
        styles=[
            StyleDefinition(name=name, core_style=CoreStyle(core), description=desc)
            for name, (core, desc) in STYLES.items()
        ],
        source=SourceInfo(
            description=f"AI Hobbyist dataset, imported from {args.source.name}",
            game_or_media="鸣潮 (Wuthering Waves)",
            extraction_tool="AI Hobbyist dataset (wav + lab)",
            license_note="Game voice lines; personal, non-commercial use only.",
        ),
    )
    if not paths.manifest_file.is_file():
        scaffold_voicepack(paths, manifest)

    candidates: dict[str, list[tuple[float, Path, str, int, int]]] = {s: [] for s in STYLES}
    for wav in sorted(args.source.glob("*.wav")):
        lab = wav.with_suffix(".lab")
        if not lab.is_file():
            continue
        text = lab.read_text(encoding="utf-8").strip()
        length, rate, channels = duration(wav)
        if text and MIN_S <= length <= MAX_S:
            candidates[style_of(text)].append((length, wav, text, rate, channels))

    samples: list[ReferenceSample] = []
    for style, items in candidates.items():
        items.sort(key=lambda item: abs(item[0] - IDEAL_S))
        target_dir = paths.references / style
        target_dir.mkdir(parents=True, exist_ok=True)
        for length, wav, text, rate, channels in items[: args.per_style]:
            shutil.copy2(wav, target_dir / wav.name)
            samples.append(
                ReferenceSample(
                    reference_id=wav.stem,
                    audio_path=f"references/{style}/{wav.name}",
                    transcript=text,
                    style=style,
                    core_style=CoreStyle(STYLES[style][0]),
                    audio=AudioProperties(
                        sample_rate=rate, channels=channels, duration_s=round(length, 3)
                    ),
                    quality_score=round(max(0.0, 1.0 - abs(length - IDEAL_S) / 10), 3),
                    tags=["game_line"],
                )
            )
        print(f"{style:8} {min(len(items), args.per_style):2d} of {len(items)} candidates")

    bank = ReferenceBank(
        voicepack_id=args.voicepack_id,
        voicepack_version=manifest.version,
        samples=samples,
    )
    paths.references_file.write_text(
        bank.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"wrote {len(samples)} references to {paths.references_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
