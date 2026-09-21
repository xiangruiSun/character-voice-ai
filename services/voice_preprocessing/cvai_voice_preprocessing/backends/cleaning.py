"""Source separation and denoising — the two stages most likely to do harm.

Both are off by default (decision D7). Spec §7 is explicit that the objective is not
maximum spectral cleanliness, and both of these remove part of the performance along
with the interference:

* **Separation** is for audio with music or effects mixed under the dialogue. On dry
  dialogue it costs high-frequency detail and adds phase artefacts for nothing.
* **Denoising** is for genuine broadband noise. Every denoiser attenuates breaths,
  aspiration and vocal fry — precisely the "low-level expressive details" spec §7 says
  to preserve.

So both are opt-in, both record themselves in ``processing_chain``, and the pipeline
keeps a null-processing control set so their effect is measurable rather than assumed.

Generative restoration (resemble-enhance and similar) is **not** offered. It resynthesizes
the voice, which changes timbre — fatal for a dataset whose entire purpose is preserving
one specific timbre.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .base import DenoiseBackend, SeparationBackend

_SEPARATOR_HINT = (
    "audio-separator is not installed; run `make install-preprocess` "
    "(pip install -e '.[preprocess]')"
)
_DF_HINT = (
    "DeepFilterNet is not installed; `pip install deepfilternet` if this pack genuinely "
    "needs denoising — most do not"
)

#: Top-SDR vocal isolation model from the UVR zoo. Chosen over the MDX-Net defaults
#: because it leaves fewer artefacts in the residual, which matters when the residual is
#: a character's breath rather than a guitar.
DEFAULT_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class UVRSeparationBackend(SeparationBackend):
    """Vocal isolation through ``audio-separator`` (the UVR model zoo)."""

    name = "audio-separator"

    def __init__(
        self,
        *,
        model_filename: str = DEFAULT_VOCAL_MODEL,
        model_dir: str | None = None,
        use_gpu: bool = True,
    ) -> None:
        self.model_filename = model_filename
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.version = model_filename
        self._separator: Any = None

    def available(self) -> bool:
        try:
            import audio_separator  # noqa: F401,PLC0415

            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return _SEPARATOR_HINT

    def _ensure(self, output_dir: Path) -> Any:
        if self._separator is None:
            try:
                from audio_separator.separator import Separator  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(_SEPARATOR_HINT) from exc
            kwargs: dict[str, Any] = {"output_dir": str(output_dir)}
            if self.model_dir:
                kwargs["model_file_dir"] = self.model_dir
            self._separator = Separator(**kwargs)
            self._separator.load_model(model_filename=self.model_filename)
        else:
            # Reusing one loaded model across a whole pack is the difference between
            # minutes and hours; only the output directory changes per clip.
            self._separator.output_dir = str(output_dir)
        return self._separator

    def isolate_vocals(self, source: Path, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._ensure(output_dir)
        produced = separator.separate(str(source)) or []

        candidates = [output_dir / name for name in produced]
        vocals = _pick_vocal_stem(candidates)
        if vocals is None:
            raise RuntimeError(
                f"{self.name} produced no vocal stem for {source}; got {produced}"
            )
        destination = output_dir / f"{Path(source).stem}.wav"
        if vocals != destination:
            shutil.move(str(vocals), destination)
        for leftover in candidates:
            if leftover.exists() and leftover != destination:
                leftover.unlink(missing_ok=True)
        return destination


def _pick_vocal_stem(paths: list[Path]) -> Path | None:
    for path in paths:
        if "vocal" in path.name.lower():
            return path
    return paths[0] if paths else None


class DeepFilterNetBackend(DenoiseBackend):
    """Conservative broadband denoising.

    ``attenuation_db`` is capped low on purpose. Full-strength suppression is what
    removes the breath before a line; 10-12 dB takes the hiss off without hollowing out
    the voice. If a clip needs more than that to be usable, it is usually better to
    reject the clip.
    """

    name = "deepfilternet"

    def __init__(self, *, attenuation_db: float = 12.0, device: str = "cpu") -> None:
        self.attenuation_db = attenuation_db
        self.device = device
        self.version = "3"
        self._state: Any = None
        self._model: Any = None
        self._df: Any = None

    def available(self) -> bool:
        try:
            import df  # noqa: F401,PLC0415

            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return _DF_HINT

    def _ensure(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            try:
                from df.enhance import init_df  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise RuntimeError(_DF_HINT) from exc
            self._model, self._state, _ = init_df()
        return self._model, self._state, self._df

    def denoise(self, source: Path, destination: Path) -> Path:
        from df.enhance import enhance, load_audio, save_audio  # noqa: PLC0415

        model, state, _ = self._ensure()
        audio, _ = load_audio(str(source), sr=state.sr())
        enhanced = enhance(model, state, audio, atten_lim_db=self.attenuation_db)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_audio(str(destination), enhanced, sr=state.sr())
        return destination


class CopyOnlyCleaningBackend(SeparationBackend, DenoiseBackend):
    """Pass-through used when a cleaning stage is enabled but has no backend.

    Copies the file and records that nothing was done, so the ``processing_chain`` is
    honest about it. Better than a silent skip: "separation was on but unavailable" and
    "separation ran" must not look the same in the record.
    """

    name = "passthrough"
    version = "1"
    destructive = False

    def isolate_vocals(self, source: Path, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / Path(source).name
        shutil.copy2(source, destination)
        return destination

    def denoise(self, source: Path, destination: Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
