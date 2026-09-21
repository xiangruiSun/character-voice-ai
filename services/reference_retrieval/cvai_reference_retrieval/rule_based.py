"""Metadata-rule reference retriever.

Three behaviours matter here, all of them because of failure modes the spec calls out:

1. **Style fallback** (spec §10) — ``embarrassed_teasing`` → ``teasing`` → ``neutral``,
   so a planner can ask for a nuanced character style before the bank has clips for it
   without the request failing.
2. **Rotation** (spec §27, "using one reference clip for all emotions") — among the
   clips that are good enough, which one is returned varies with the utterance, so a
   whole conversation is not conditioned on a single recording. The variation is derived
   from a key rather than an RNG, so a benchmark run is still bit-for-bit reproducible.
3. **Honest fallback reporting** — the selection records whether the requested style was
   actually available. A benchmark where most lines quietly fell back to neutral is
   measuring neutral delivery for every candidate, and the fallback rate is how that
   becomes visible instead of invisible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cvai_core.errors import ReferenceBankError
from cvai_core.interfaces.reference import ReferenceRetriever
from cvai_core.paths import VoicePackPaths
from cvai_types import (
    CoreStyle,
    CVAIModel,
    ReferenceBank,
    ReferenceSample,
    ReferenceSelection,
    StyleControls,
    StyleDefinition,
    VoicePackManifest,
    resolve_style_chain,
)
from pydantic import Field


class ReferenceScore(CVAIModel):
    """Why a clip was ranked where it was. Exposed for debugging and reports."""

    reference_id: str
    total: float
    quality: float
    duration_fit: float
    tag_bonus: float = 0.0
    notes: str = Field(default="", max_length=200)


class RuleBasedReferenceRetriever(ReferenceRetriever):
    """Select a reference clip from a :class:`~cvai_types.voicepack.ReferenceBank`."""

    def __init__(
        self,
        bank: ReferenceBank,
        *,
        pack_paths: VoicePackPaths | None = None,
        style_definitions: dict[str, StyleDefinition] | None = None,
        preferred_duration_s: tuple[float, float] = (3.0, 10.0),
        rotation_pool: int = 3,
        enable_rotation: bool = True,
    ) -> None:
        if not bank.samples:
            raise ReferenceBankError(
                f"reference bank for {bank.voicepack_id!r} is empty; "
                "build it in Milestone 2 before synthesizing anything"
            )
        self.bank = bank
        self.pack_paths = pack_paths
        self.style_definitions = style_definitions or {}
        self.preferred_duration_s = preferred_duration_s
        self.rotation_pool = max(1, rotation_pool)
        self.enable_rotation = enable_rotation

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_voicepack(
        cls,
        paths: VoicePackPaths,
        manifest: VoicePackManifest,
        bank: ReferenceBank,
        **kwargs: object,
    ) -> "RuleBasedReferenceRetriever":
        return cls(
            bank,
            pack_paths=paths,
            style_definitions=manifest.style_map(),
            **kwargs,  # type: ignore[arg-type]
        )

    # -- interface ----------------------------------------------------------------

    def select(
        self,
        controls: StyleControls,
        *,
        variation_key: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> ReferenceSelection:
        requested = controls.effective_reference_style
        excluded = exclude_ids or set()

        matched_style, candidates = self._candidates_for(requested, excluded)
        if not candidates:
            raise ReferenceBankError(
                f"no reference clip available for style {requested!r} "
                f"(excluded {len(excluded)} ids); the bank has {self.bank.coverage()}"
            )

        ranked = self.rank(candidates)
        chosen = self._rotate(ranked, candidates, variation_key)

        return ReferenceSelection(
            reference_id=chosen.reference_id,
            audio_path=self._resolve_audio(chosen),
            transcript=chosen.transcript,
            language=chosen.language,
            style=chosen.style,
            core_style=chosen.core_style,
            was_fallback=matched_style != requested,
            requested_style=requested,
            precomputed=dict(chosen.precomputed),
        )

    def coverage(self) -> dict[str, int]:
        return self.bank.coverage()

    # -- scoring ------------------------------------------------------------------

    def rank(self, candidates: list[ReferenceSample]) -> list[ReferenceScore]:
        """Score clips; higher is better. Ties break on ``reference_id`` for stability."""
        scores = [self._score(sample) for sample in candidates]
        return sorted(scores, key=lambda s: (-s.total, s.reference_id))

    def _score(self, sample: ReferenceSample) -> ReferenceScore:
        low, high = self.preferred_duration_s
        duration = sample.audio.duration_s
        if low <= duration <= high:
            duration_fit = 1.0
        elif duration < low:
            # Short clips carry too little prosody for the engine to imitate.
            duration_fit = max(0.0, duration / low)
        else:
            # Long clips get truncated by most engines and drift on the rest.
            duration_fit = max(0.0, 1.0 - (duration - high) / (high * 2.0))

        quality = sample.quality_score
        total = 0.65 * quality + 0.35 * duration_fit
        return ReferenceScore(
            reference_id=sample.reference_id,
            total=round(total, 6),
            quality=quality,
            duration_fit=round(duration_fit, 6),
        )

    # -- internals ----------------------------------------------------------------

    def _candidates_for(
        self, requested: str, excluded: set[str]
    ) -> tuple[str, list[ReferenceSample]]:
        for style in resolve_style_chain(requested, self.style_definitions):
            pool = [s for s in self.bank.by_style(style) if s.reference_id not in excluded]
            if pool:
                return style, pool
        # Nothing in the chain: use whatever the bank has rather than failing a live turn.
        remainder = [s for s in self.bank.samples if s.reference_id not in excluded]
        fallback_style = remainder[0].style if remainder else requested
        return fallback_style, remainder

    def _rotate(
        self,
        ranked: list[ReferenceScore],
        candidates: list[ReferenceSample],
        variation_key: str | None,
    ) -> ReferenceSample:
        by_id = {s.reference_id: s for s in candidates}
        if not self.enable_rotation or variation_key is None:
            return by_id[ranked[0].reference_id]

        pool = ranked[: min(self.rotation_pool, len(ranked))]
        index = _stable_index(variation_key, len(pool))
        return by_id[pool[index].reference_id]

    def _resolve_audio(self, sample: ReferenceSample) -> str:
        if self.pack_paths is None:
            return sample.audio_path
        return str(self.pack_paths.resolve(sample.audio_path))


def _stable_index(key: str, modulus: int) -> int:
    """Deterministic index from a string.

    ``hash()`` is salted per process in CPython, which would make benchmark runs
    irreproducible across invocations — hence an explicit digest.
    """
    if modulus <= 1:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def core_style_of(style: str, definitions: dict[str, StyleDefinition]) -> CoreStyle:
    definition = definitions.get(style)
    if definition is not None:
        return definition.core_style
    try:
        return CoreStyle(style)
    except ValueError:
        return CoreStyle.NEUTRAL


def reference_audio_path(paths: VoicePackPaths, sample: ReferenceSample) -> Path:
    return paths.resolve(sample.audio_path)
