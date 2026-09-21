"""``CharacterProvider`` — where character definitions come from.

Filesystem-backed in V1 (spec §3: local filesystem only, no database). The interface
exists so that a later database or CMS is a new implementation rather than a rewrite, and
so tests can supply an in-memory provider without touching disk.
"""

from __future__ import annotations

import abc

from cvai_types import CharacterProfile, DialogueExample


class CharacterProvider(abc.ABC):
    @abc.abstractmethod
    def get(self, character_id: str) -> CharacterProfile:
        """Load one profile. Raises ``VoicePackError``/``ConfigError`` if unknown."""

    @abc.abstractmethod
    def list_ids(self) -> list[str]:
        ...

    def retrieve_examples(
        self,
        character_id: str,
        query: str,
        *,
        limit: int = 6,
        style: str | None = None,
    ) -> list[DialogueExample]:
        """Pick dialogue examples to include in the prompt.

        Spec §11 wants "system prompt + original dialogue examples + lore + history", and
        the examples are the part that actually carries voice. The default implementation
        filters by style and truncates; Milestone 8 can replace it with embedding
        retrieval without changing any caller.
        """
        profile = self.get(character_id)
        pool = profile.dialogue_examples
        if style:
            styled = [e for e in pool if e.style == style]
            if styled:
                pool = styled
        return list(pool[:limit])
