"""``ReferenceRetriever`` — which reference clip to condition on (spec §10).

Its own component because spec §27 names "using one reference clip for all emotions" as a
failure mode, and because the swap from metadata rules (V1) to embedding retrieval (later)
must not touch the TTS adapters.

The retriever returns a :class:`~cvai_types.tts.ReferenceSelection` that carries the
transcript with the clip (decision D3), so an adapter never has to reach back into the
bank, and records ``was_fallback`` so evaluation can tell whether the test actually
exercised the requested styles.
"""

from __future__ import annotations

import abc

from cvai_types import ReferenceSelection, StyleControls


class ReferenceRetriever(abc.ABC):
    @abc.abstractmethod
    def select(
        self,
        controls: StyleControls,
        *,
        variation_key: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> ReferenceSelection:
        """Choose a reference clip for one utterance.

        ``variation_key`` makes selection deterministic *and* varied: the same key always
        yields the same clip (reproducible benchmarks), while different keys rotate
        through the available clips for a style instead of always returning the first
        one. Passing the sentence id, or the utterance text, is the usual choice.
        """

    @abc.abstractmethod
    def coverage(self) -> dict[str, int]:
        """Clips available per style. Used by the voice pack validator and reports."""

    def available_styles(self) -> list[str]:
        return sorted(self.coverage())
