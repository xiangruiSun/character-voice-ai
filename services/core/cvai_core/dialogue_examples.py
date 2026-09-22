"""Choosing which of a character's real lines go into the prompt (spec §11).

Spec §11 rules out fine-tuning the LLM for V1 and prescribes "system prompt + original
dialogue examples + character lore + conversation history". Of those four, the examples
are the part that carries voice: adjectives tell a model what to aim for, her actual
lines show it. Which examples get chosen is therefore not a detail — it is most of what
makes the character sound like herself in a given turn.

The first implementation took the first N after a style filter, which means a profile
with two hundred lines always showed the same six, whatever the user said. A question
about her past, a joke and a goodbye all got the same evidence, and the model had to
generalise from lines that had nothing to do with the situation.

This module picks by relevance instead, with no model, no embeddings and no network:

* **Character bigrams, IDF-weighted.** Chinese has no spaces, and a word segmenter is a
  dependency and a failure mode. Overlapping bigrams are the standard cheap substitute
  and work well at this scale. IDF is computed over the character's own examples, so
  common connectives she says constantly stop dominating the score and the distinctive
  words do the work.
* **Matched against what was said *to* her**, primarily. What predicts how she answers
  is the situation she was answering, not the words of her reply — so ``user`` is
  weighted far above ``character``. Standalone lines (battle cries, idle barks) have no
  ``user`` at all and are scored on their own text at a discount.
* **Style is a strong preference, not a filter.** Filtering hard means a "teasing" turn
  with two teasing examples in the profile shows those two and nothing else, and the
  model loses every other signal about how she speaks.
* **Diversity, so six examples are not six paraphrases.** Near-duplicates crowd out the
  range the prompt is supposed to demonstrate — the same failure as conditioning every
  line on one reference clip, one level up.

Ties break on the profile's own ordering, so the selection is deterministic: the same
turn produces the same prompt, which is the least a reproducible experiment can ask.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from cvai_types import DialogueExample

#: Weight on the line the user said, versus the character's own reply. The situation
#: predicts the answer far better than the answer does.
USER_WEIGHT = 1.0
CHARACTER_WEIGHT = 0.35
#: Multiplier for an example in the requested style. A preference, not a filter.
STYLE_BOOST = 1.6
#: How much an example is penalised for resembling one already chosen. 0 ignores
#: redundancy; 1 makes it nearly disqualifying.
DIVERSITY = 0.55
#: Above this similarity to a line already chosen, an example is dropped rather than
#: merely penalised. Profiles collected from a game are full of repeated barks, and a
#: second copy of a line the model has already seen teaches it nothing while costing one
#: of the few slots that could have shown her range.
NEAR_DUPLICATE = 0.85

_DROP = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】—…·~～\-]+")


def normalize(text: str) -> str:
    return _DROP.sub("", text or "")


def bigrams(text: str) -> set[str]:
    """Overlapping character bigrams, plus single characters for very short text.

    "在吗" has one bigram; "在" has none, and a one-character query that scored zero
    against everything would silently fall back to profile order.
    """
    cleaned = normalize(text)
    if len(cleaned) < 2:
        return set(cleaned)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _idf(pool: Sequence[DialogueExample]) -> dict[str, float]:
    """Inverse document frequency over this character's own lines.

    Over her lines rather than over Chinese in general: what matters is which words are
    distinctive *for her*. A character who says 大人 in half her lines should not have
    every turn retrieved by it.
    """
    total = max(1, len(pool))
    counts: dict[str, int] = {}
    for example in pool:
        for gram in bigrams(example.user) | bigrams(example.character):
            counts[gram] = counts.get(gram, 0) + 1
    return {
        gram: math.log(1.0 + total / (1.0 + count)) for gram, count in counts.items()
    }


def _weighted_overlap(
    query: set[str], target: set[str], idf: dict[str, float]
) -> float:
    if not query or not target:
        return 0.0
    shared = query & target
    if not shared:
        return 0.0
    score = sum(idf.get(gram, 1.0) for gram in shared)
    # Normalised by the query, not by the union: a long example is not worse evidence
    # than a short one just for being long.
    ceiling = sum(idf.get(gram, 1.0) for gram in query)
    return score / ceiling if ceiling else 0.0


def score_example(
    query: str,
    example: DialogueExample,
    idf: dict[str, float],
    *,
    style: str | None = None,
) -> float:
    grams = bigrams(query)
    user_grams = bigrams(example.user)
    character_grams = bigrams(example.character)

    score = USER_WEIGHT * _weighted_overlap(grams, user_grams, idf)
    score += CHARACTER_WEIGHT * _weighted_overlap(grams, character_grams, idf)
    if not user_grams:
        # A standalone line has no situation to match against, so its own text is all
        # there is. Discounted rather than excluded: an idle bark is still her voice.
        score *= 0.8

    if style and example.style == style:
        score *= STYLE_BOOST
    return score


def select_examples(
    query: str,
    pool: Sequence[DialogueExample],
    *,
    limit: int = 6,
    style: str | None = None,
) -> list[DialogueExample]:
    """The examples worth showing the model for this turn.

    Falls back to the profile's own order when nothing is relevant — early in a
    conversation, or for a character whose examples are all standalone lines, a prompt
    with her real lines in it beats a prompt with none.
    """
    if limit <= 0 or not pool:
        return []
    if len(pool) <= limit:
        return _distinct(pool, limit)

    idf = _idf(pool)
    scored = [
        (score_example(query, example, idf, style=style), index, example)
        for index, example in enumerate(pool)
    ]

    chosen: list[tuple[int, DialogueExample]] = []
    chosen_grams: list[set[str]] = []
    remaining = list(scored)

    while remaining and len(chosen) < limit:
        best: tuple[float, int, DialogueExample] | None = None
        best_value = -1.0
        for score, index, example in remaining:
            grams = bigrams(example.character)
            redundancy = max(
                (_jaccard(grams, seen) for seen in chosen_grams), default=0.0
            )
            if redundancy >= NEAR_DUPLICATE:
                continue
            value = score - DIVERSITY * redundancy * max(score, 0.2)
            # Strict > keeps the profile's own order as the tie-break, which is what
            # makes the same turn produce the same prompt every time.
            if value > best_value:
                best_value, best = value, (score, index, example)
        if best is None:
            # Everything left is a near-duplicate of something already chosen. Fewer,
            # distinct examples is the better prompt — repeating a line the model has
            # already read does not make it more her.
            break
        remaining.remove(best)
        chosen.append((best[1], best[2]))
        chosen_grams.append(bigrams(best[2].character))

    if all(score <= 0.0 for score, _, _ in scored):
        # Nothing matched. Profile order is a deliberate choice by whoever wrote the
        # character; an arbitrary relevance ordering of zeros is not. Duplicates are
        # still dropped — repetition is never the better prompt.
        return _distinct(pool, limit)

    # Present them in profile order. The prompt reads as a transcript of her, and a
    # relevance-sorted list quietly tells the model that the first one matters most.
    return [example for _, example in sorted(chosen, key=lambda pair: pair[0])]


def _distinct(pool: Sequence[DialogueExample], limit: int) -> list[DialogueExample]:
    """The first ``limit`` examples that are not repeats of each other."""
    chosen: list[DialogueExample] = []
    seen: list[set[str]] = []
    for example in pool:
        grams = bigrams(example.character)
        if any(_jaccard(grams, other) >= NEAR_DUPLICATE for other in seen):
            continue
        chosen.append(example)
        seen.append(grams)
        if len(chosen) >= limit:
            break
    return chosen


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0
