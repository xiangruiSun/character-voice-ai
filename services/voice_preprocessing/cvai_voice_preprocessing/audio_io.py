"""Audio decoding and I/O for the preprocessing pipeline.

Decoding goes through ffmpeg rather than a Python library: the inputs are whatever a game
shipped (``.wem``, ``.ogg``, ``.mp3``, odd sample rates, occasionally 8-bit), and ffmpeg
handles all of it with one code path. Reading samples prefers ``soundfile`` when it is
installed and falls back to the standard library, so the pipeline still runs — and stays
testable — in an environment with nothing but pydantic and PyYAML.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final, Sequence

from cvai_core.audio import read_wav_properties, read_wav_samples, write_wav
from cvai_core.errors import CVAIError
from cvai_types import AudioProperties

FFMPEG: Final = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE: Final = shutil.which("ffprobe") or "ffprobe"

#: Formats worth trying. Anything ffmpeg can open works; this list only drives directory
#: scanning. ``.wem`` needs vgmstream first — see docs/VOICEPACK.md.
AUDIO_EXTENSIONS: Final[tuple[str, ...]] = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
)


class AudioError(CVAIError):
    """Decoding or reading failed."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_to_wav(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 44100,
    mono: bool = True,
    overwrite: bool = True,
) -> Path:
    """Decode anything ffmpeg understands to 16-bit PCM WAV.

    Deliberately *only* decodes and resamples. No loudness change, no filtering, no
    trimming: the point of this step is to make the audio readable without altering the
    performance, so that ``raw/`` and the first processed copy differ only in container.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    if not ffmpeg_available():
        raise AudioError(
            "ffmpeg is not installed; it is required to decode source audio "
            "(apt install ffmpeg / brew install ffmpeg)"
        )

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vn",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        "-acodec",
        "pcm_s16le",
    ]
    if mono:
        command += ["-ac", "1"]
    command.append(str(destination))

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise AudioError(
            f"ffmpeg failed to decode {source}:\n{result.stderr.strip()[:800]}"
        )
    return destination


def probe(path: Path) -> AudioProperties:
    """Audio properties, from the WAV header when possible, otherwise ffprobe."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return read_wav_properties(path)
        except Exception:  # noqa: BLE001 - fall through to ffprobe for odd WAVs
            pass

    if shutil.which("ffprobe") is None:
        raise AudioError(f"cannot probe {path}: not a readable WAV and ffprobe is missing")

    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels:format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioError(f"ffprobe failed on {path}: {result.stderr.strip()[:400]}")

    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    try:
        return AudioProperties(
            sample_rate=int(fields["sample_rate"]),
            channels=int(fields.get("channels", 1)),
            duration_s=float(fields["duration"]),
        )
    except (KeyError, ValueError) as exc:
        raise AudioError(f"ffprobe gave no usable stream info for {path}") from exc


def read_samples(path: Path) -> tuple[list[float], int]:
    """Read a WAV as mono floats in [-1, 1].

    Uses ``soundfile`` when available (faster, handles 24/32-bit and float WAVs) and the
    standard library otherwise.
    """
    path = Path(path)
    try:
        import soundfile  # noqa: PLC0415 - optional acceleration
    except ImportError:
        return read_wav_samples(path)

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return [float(v) for v in mono], int(rate)


def write_samples(path: Path, samples: Sequence[float], sample_rate: int) -> Path:
    write_wav(Path(path), samples, sample_rate)
    return Path(path)


def slice_samples(
    samples: Sequence[float], sample_rate: int, start_s: float, end_s: float
) -> list[float]:
    start = max(0, int(start_s * sample_rate))
    end = min(len(samples), int(end_s * sample_rate))
    return list(samples[start:end])


def iter_audio_files(root: Path, recursive: bool = True) -> list[Path]:
    """All audio files under ``root``, sorted, hidden files skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and not path.name.startswith(".")
    ]
    return found
