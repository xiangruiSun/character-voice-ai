"""FunASR-backed stages: Chinese ASR, speaker embeddings, emotion.

FunASR is the right base for a Chinese pipeline: one MIT-licensed toolkit covers
``paraformer-zh`` (ASR with character-level timestamps), ``ct-punc`` (punctuation
restoration), ``fsmn-vad``, ``cam++`` (speaker embeddings) and ``emotion2vec`` — so the
pack is annotated by one consistent family of models rather than five unrelated ones.

Punctuation restoration matters more here than it looks. A transcript without 。？！……
trains the model on text that never matches what the LLM will later send it, and the
prosody it learns to attach to a sentence ending goes with it.

**Result-shape handling is deliberately defensive.** FunASR's return dictionaries vary by
model and version. Each extractor tries the documented keys and falls back rather than
raising, because a pipeline that dies on an unexpected key three hours into a run is
worse than one that records a missing field and carries on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .base import (
    ASRBackend,
    ASRResult,
    CharTiming,
    EmotionBackend,
    EmotionResult,
    SpeakerBackend,
)

_IMPORT_HINT = (
    "FunASR is not installed; run `make install-preprocess` "
    "(pip install -e '.[preprocess]')"
)

#: emotion2vec's nine classes mapped onto the project's core style taxonomy (spec §9).
#: ``other`` and ``unknown`` map to nothing on purpose — a shrug is more useful than a
#: wrong label a human then has to notice and undo.
EMOTION2VEC_TO_CORE_STYLE: dict[str, str | None] = {
    "angry": "angry",
    "生气": "angry",
    "disgusted": "angry",
    "厌恶": "angry",
    "fearful": "surprised",
    "恐惧": "surprised",
    "happy": "happy",
    "开心": "happy",
    "neutral": "neutral",
    "中立": "neutral",
    "sad": "sad",
    "难过": "sad",
    "surprised": "surprised",
    "吃惊": "surprised",
    "other": None,
    "其他": None,
    "unknown": None,
    "未知": None,
}


def _load_funasr() -> Any:
    try:
        from funasr import AutoModel  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(_IMPORT_HINT) from exc
    return AutoModel


def _funasr_installed() -> bool:
    try:
        import funasr  # noqa: F401,PLC0415

        return True
    except ImportError:
        return False


def _first(result: Any) -> dict[str, Any]:
    if isinstance(result, list) and result:
        first = result[0]
        return first if isinstance(first, dict) else {}
    return result if isinstance(result, dict) else {}


# --------------------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------------------


class FunASRTranscriber(ASRBackend):
    """``paraformer-zh`` + ``ct-punc``, with character timestamps."""

    name = "funasr-paraformer-zh"

    def __init__(
        self,
        *,
        model: str = "paraformer-zh",
        punc_model: str | None = "ct-punc",
        vad_model: str | None = None,
        device: str = "cpu",
        batch_size_s: int = 300,
        disable_update: bool = True,
        hotword_weight: int = 20,
    ) -> None:
        self.model_id = model
        self.punc_model = punc_model
        self.hotword_weight = hotword_weight
        # VAD is off by default here: the pipeline has already segmented, and a second
        # pass would re-split a clip that a human is about to review as one unit.
        self.vad_model = vad_model
        self.device = device
        self.batch_size_s = batch_size_s
        self.disable_update = disable_update
        self.version = model
        self._model: Any = None

    def available(self) -> bool:
        return _funasr_installed()

    def unavailable_reason(self) -> str:
        return _IMPORT_HINT

    def _ensure(self) -> Any:
        if self._model is None:
            AutoModel = _load_funasr()
            kwargs: dict[str, Any] = {
                "model": self.model_id,
                "device": self.device,
                "disable_update": self.disable_update,
            }
            if self.punc_model:
                kwargs["punc_model"] = self.punc_model
            if self.vad_model:
                kwargs["vad_model"] = self.vad_model
            self._model = AutoModel(**kwargs)
        return self._model

    def transcribe(self, path: Path, *, hotwords: Sequence[str] = ()) -> ASRResult:
        model = self._ensure()
        kwargs: dict[str, Any] = {"input": str(path), "batch_size_s": self.batch_size_s}
        if hotwords:
            # paraformer takes "word weight word weight …"; 20 is the usual weight.
            kwargs["hotword"] = " ".join(
                f"{word} {self.hotword_weight}" for word in hotwords
            )

        raw = model.generate(**kwargs)
        payload = _first(raw)
        text = str(payload.get("text", "")).strip()

        return ASRResult(
            text=text,
            confidence=_extract_confidence(payload),
            char_timings=_extract_char_timings(text, payload),
            sentences=list(payload.get("sentence_info", []) or []),
            model=self.model_id,
        )


def _extract_confidence(payload: dict[str, Any]) -> float | None:
    for key in ("confidence", "score", "avg_logprob"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            # Log-probabilities are negative; squash into 0-1 so the field means one
            # thing regardless of which model filled it.
            if value < 0:
                return max(0.0, min(1.0, 1.0 + float(value) / 5.0))
            return max(0.0, min(1.0, float(value)))
    return None


def _extract_char_timings(text: str, payload: dict[str, Any]) -> list[CharTiming]:
    """Pair paraformer's ``timestamp`` list with the transcript's characters.

    The timestamp list covers spoken tokens only, so punctuation inserted by ``ct-punc``
    has to be skipped when zipping. Mismatched lengths yield no timings rather than
    wrong ones — a wrong alignment is worse than none, because it silently corrupts any
    QC that depends on it.
    """
    stamps = payload.get("timestamp")
    if not isinstance(stamps, list) or not stamps or not text:
        return []

    spoken = [ch for ch in text if not _is_punctuation(ch) and not ch.isspace()]
    if len(spoken) != len(stamps):
        return []

    timings: list[CharTiming] = []
    for char, stamp in zip(spoken, stamps):
        try:
            start, end = int(stamp[0]), int(stamp[1])
        except (TypeError, ValueError, IndexError):
            return []
        timings.append(CharTiming(char=char, start_ms=max(0, start), end_ms=max(0, end)))
    return timings


_PUNCTUATION = set("，。！？；：、…,.!?;:\"'“”‘’()（）《》—-　")


def _is_punctuation(char: str) -> bool:
    return char in _PUNCTUATION


# --------------------------------------------------------------------------------------
# Speaker embeddings
# --------------------------------------------------------------------------------------


class CamPlusPlusSpeakerBackend(SpeakerBackend):
    """CAM++ speaker embeddings, used to keep other characters out of the dataset.

    This is the defence against spec §27's "training on other characters": build a
    centroid from clips a human confirmed, score every segment against it, and let the
    reviewer see the number.
    """

    name = "funasr-campplus"

    def __init__(self, *, model: str = "cam++", device: str = "cpu") -> None:
        self.model_id = model
        self.device = device
        self.version = model
        self._model: Any = None

    def available(self) -> bool:
        return _funasr_installed()

    def unavailable_reason(self) -> str:
        return _IMPORT_HINT

    def _ensure(self) -> Any:
        if self._model is None:
            AutoModel = _load_funasr()
            self._model = AutoModel(
                model=self.model_id, device=self.device, disable_update=True
            )
        return self._model

    def embed(self, path: Path) -> list[float]:
        model = self._ensure()
        raw = model.generate(input=str(path), extract_embedding=True)
        payload = _first(raw)
        vector = _extract_embedding(payload)
        if vector is None:
            raise RuntimeError(
                f"{self.name} returned no embedding for {path}; "
                f"keys present: {sorted(payload)[:10]}"
            )
        return vector


def _extract_embedding(payload: dict[str, Any]) -> list[float] | None:
    for key in ("spk_embedding", "embedding", "feats", "spk_emb"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            flattened = _flatten_numeric(value)
        except TypeError:
            continue
        if flattened:
            return flattened
    return None


def _flatten_numeric(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    raise TypeError(type(value))


# --------------------------------------------------------------------------------------
# Emotion
# --------------------------------------------------------------------------------------


class Emotion2VecBackend(EmotionBackend):
    """emotion2vec+ utterance-level emotion classification.

    A first pass only. The labels that carry character identity — ``soft_teasing``,
    ``embarrassed``, ``cold`` — are not in any pretrained taxonomy, so this bootstraps
    the review queue and a human assigns the real style (spec §9).
    """

    name = "emotion2vec"

    def __init__(
        self,
        *,
        model: str = "iic/emotion2vec_plus_large",
        device: str = "cpu",
        granularity: str = "utterance",
    ) -> None:
        self.model_id = model
        self.device = device
        self.granularity = granularity
        self.version = model
        self._model: Any = None

    def available(self) -> bool:
        return _funasr_installed()

    def unavailable_reason(self) -> str:
        return _IMPORT_HINT

    def _ensure(self) -> Any:
        if self._model is None:
            AutoModel = _load_funasr()
            self._model = AutoModel(
                model=self.model_id, device=self.device, disable_update=True
            )
        return self._model

    def classify(self, path: Path) -> EmotionResult:
        model = self._ensure()
        raw = model.generate(
            str(path), granularity=self.granularity, extract_embedding=False
        )
        payload = _first(raw)
        labels = [str(x) for x in (payload.get("labels") or [])]
        scores = [float(x) for x in (payload.get("scores") or [])]

        pairs = dict(zip(labels, scores)) if len(labels) == len(scores) else {}
        best = max(pairs, key=pairs.get) if pairs else "unknown"
        return EmotionResult(
            label=normalize_emotion_label(best),
            scores={normalize_emotion_label(k): v for k, v in pairs.items()},
            model=self.model_id,
        )


def normalize_emotion_label(label: str) -> str:
    """Map a model label onto a core style, or ``unknown``.

    emotion2vec returns labels like ``"生气/angry"``; both halves are checked.
    """
    cleaned = label.strip().lower()
    for part in cleaned.replace("|", "/").split("/"):
        part = part.strip()
        if part in EMOTION2VEC_TO_CORE_STYLE:
            mapped = EMOTION2VEC_TO_CORE_STYLE[part]
            return mapped or "unknown"
    return "unknown"
