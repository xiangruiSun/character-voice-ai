"""Checking a character profile before it costs anything (spec §11, §27).

The profile is the whole "what she says" half of the system, and it is written by hand.
Nothing about it fails loudly: a style declared with no examples still produces replies,
a profile with six lines still answers, an empty `world_knowledge` list still talks
confidently about a world it is inventing. Every one of those produces a character that
is subtly not her, which is the failure this project exists to avoid — and it is much
cheaper to catch here than in a listening test.

So this is a linter, not a schema. The schema already rejects what is malformed; these
are the things that are *well-formed and wrong*:

* **A style with no examples.** The Speech Planner may ask for any style in
  ``available_styles``. With no dialogue in that style, the model has to guess what she
  sounds like teasing — and the voice pack will faithfully perform the guess.
* **Too few examples, or examples that are all the same line.** Game rips repeat barks;
  a profile of two hundred lines that dedupes to nine is a profile of nine.
* **No ``source`` on the examples.** Spec §11 wants *original* dialogue. An example with
  no provenance cannot be audited, and a profile quietly seeded with LLM-written lines
  teaches the model to imitate an LLM imitating her.
* **No world knowledge and no forbidden behaviour.** Both act as boundaries; without
  them the model invents lore, which breaks the character fastest for exactly the people
  who know her best.

Severity is chosen by what the finding costs. Errors are things that will produce wrong
output; warnings are things that will produce weaker output. A profile early in
collection should be full of warnings and no errors, and the linter says so rather than
refusing to let anyone work.
"""

from __future__ import annotations

import re
from collections import Counter

from cvai_types import CharacterProfile, CVAIModel
from pydantic import Field

from .dialogue_examples import bigrams, normalize
from .voicepack import Issue, Severity

#: Below this, the prompt is carried by adjectives rather than by her actual voice.
MIN_EXAMPLES = 12
#: Per style in ``available_styles``. One example is a coincidence, not a register.
MIN_EXAMPLES_PER_STYLE = 2
#: Two examples this similar are one example, however the profile counts them.
DUPLICATE_SIMILARITY = 0.85
#: Above this share of examples without a ``source``, provenance cannot be audited.
MAX_UNSOURCED_RATIO = 0.5

_PINYIN = re.compile(r"^[a-zü]+[1-5]?$", re.IGNORECASE)
_LATIN = re.compile(r"[A-Za-z]")


class ProfileReport(CVAIModel):
    character_id: str
    issues: list[Issue] = Field(default_factory=list)
    stats: dict[str, object] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def render(self) -> str:
        lines = [f"Character: {self.character_id}"]
        if self.stats:
            lines.append("")
            lines.append("Stats")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
        symbol = {Severity.ERROR: "✗", Severity.WARNING: "!", Severity.INFO: "·"}
        if self.issues:
            lines.append("")
            lines.append("Findings")
            for issue in self.issues:
                lines.append(f"  {symbol[issue.severity]} [{issue.code}] {issue.message}")
        lines.append("")
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)


def lint_profile(profile: CharacterProfile) -> ProfileReport:
    """Everything worth saying about a profile before it is used."""
    issues: list[Issue] = []

    def add(severity: Severity, code: str, message: str, **context: object) -> None:
        issues.append(
            Issue(severity=severity, code=code, message=message, context=context)
        )

    examples = profile.dialogue_examples
    distinct = _distinct_count(examples)

    _check_examples(profile, distinct, add)
    _check_style_coverage(profile, add)
    _check_boundaries(profile, add)
    _check_speaking_habits(profile, add)
    _check_voice_binding(profile, add)

    by_style = Counter(example.style for example in examples)
    return ProfileReport(
        character_id=profile.character_id,
        issues=issues,
        stats={
            "dialogue_examples": len(examples),
            "distinct_examples": distinct,
            "examples_by_style": dict(sorted(by_style.items())),
            "available_styles": list(profile.available_styles),
            "world_knowledge_items": len(profile.world_knowledge),
            "frequent_expressions": len(profile.speaking_habits.frequent_expressions),
        },
    )


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def _check_examples(profile: CharacterProfile, distinct: int, add) -> None:
    examples = profile.dialogue_examples
    if not examples:
        add(
            Severity.ERROR,
            "profile.no_examples",
            "no dialogue examples. Spec §11 builds the character from her original "
            "lines rather than from fine-tuning, so without them the prompt is "
            "adjectives and the reply will be a generic anime character.",
        )
        return

    if len(examples) < MIN_EXAMPLES:
        add(
            Severity.WARNING,
            "profile.few_examples",
            f"{len(examples)} dialogue examples; aim for at least {MIN_EXAMPLES} "
            "spread across her styles. Below that the model generalises from too "
            "little and falls back on its own register.",
            count=len(examples),
        )

    if distinct < len(examples):
        add(
            Severity.WARNING,
            "profile.duplicate_examples",
            f"{len(examples) - distinct} of {len(examples)} examples repeat a line "
            "already present. Repeats are dropped when the prompt is built, so the "
            "profile is effectively smaller than it looks.",
            distinct=distinct,
        )

    unsourced = [e for e in examples if not e.source.strip()]
    if len(unsourced) > MAX_UNSOURCED_RATIO * len(examples):
        add(
            Severity.WARNING,
            "profile.unsourced_examples",
            f"{len(unsourced)} of {len(examples)} examples record no source. Spec §11 "
            "wants original dialogue; without provenance there is no way to audit that "
            "these are her lines rather than an LLM's impression of them.",
            unsourced=len(unsourced),
        )

    latin = [e for e in examples if _LATIN.search(e.character)]
    if latin:
        add(
            Severity.INFO,
            "profile.latin_in_examples",
            f"{len(latin)} example(s) contain Latin letters. V1 is zh-CN only "
            "(spec §25); check the text normalizer reads them the way she would say "
            "them.",
            count=len(latin),
        )


def _check_style_coverage(profile: CharacterProfile, add) -> None:
    """The planner may ask for any available style; each one needs evidence."""
    by_style = Counter(example.style for example in profile.dialogue_examples)

    # With no examples at all, ``profile.no_examples`` has already said it once. Saying
    # it again per style buries the findings that are still worth reading.
    per_style = profile.available_styles if profile.dialogue_examples else []

    for style in per_style:
        count = by_style.get(style, 0)
        if count == 0:
            add(
                Severity.ERROR,
                "profile.style_without_examples",
                f"style {style!r} is declared available but no dialogue example uses "
                "it. The planner will ask for it, the model will guess what she sounds "
                "like, and the voice pack will perform the guess faithfully. Add "
                "examples or remove the style.",
                style=style,
            )
        elif count < MIN_EXAMPLES_PER_STYLE:
            add(
                Severity.WARNING,
                "profile.thin_style",
                f"style {style!r} has {count} example; one line is a coincidence, not "
                "a register.",
                style=style,
                count=count,
            )

    undeclared = sorted(set(by_style) - set(profile.available_styles))
    if undeclared:
        add(
            Severity.WARNING,
            "profile.examples_in_undeclared_styles",
            f"examples use styles that are not available: {', '.join(undeclared)}. "
            "They will still be retrieved, but the planner can never request them, so "
            "that part of her range is unreachable.",
            styles=undeclared,
        )

    if profile.voice.default_reference_style not in profile.available_styles:
        add(
            Severity.ERROR,
            "profile.default_style_unavailable",
            f"default_reference_style {profile.voice.default_reference_style!r} is not "
            "in available_styles, so every fallback lands on a style this character "
            "does not declare.",
        )


def _check_boundaries(profile: CharacterProfile, add) -> None:
    if not profile.world_knowledge:
        add(
            Severity.WARNING,
            "profile.no_world_knowledge",
            "no world_knowledge. The list doubles as a boundary — the prompt tells her "
            "not to assert anything outside it — so without one she invents lore, "
            "which breaks the character fastest for the people who know her best.",
        )
    if not profile.personality:
        add(
            Severity.WARNING,
            "profile.no_personality",
            "no personality traits. The examples carry most of the voice, but with "
            "nothing to aim at the model drifts between them.",
        )
    if not profile.background.strip():
        add(Severity.INFO, "profile.no_background", "no background written.")
    if not profile.forbidden_behavior:
        add(
            Severity.INFO,
            "profile.no_forbidden_behavior",
            "nothing in forbidden_behavior. The universal rules still apply; this list "
            "is for what would break *this* character specifically.",
        )


def _check_speaking_habits(profile: CharacterProfile, add) -> None:
    habits = profile.speaking_habits
    if not habits.frequent_expressions:
        add(
            Severity.WARNING,
            "profile.no_frequent_expressions",
            "no frequent_expressions. They go into the prompt, and they are also what "
            "the microphone path feeds the recogniser as hotwords — without them her "
            "own name comes back from ASR mangled.",
        )

    for word, reading in habits.pronunciation_overrides.items():
        if not _PINYIN.match(reading.strip()):
            add(
                Severity.ERROR,
                "profile.bad_pronunciation",
                f"pronunciation override {word!r} → {reading!r} is not pinyin with an "
                "optional tone digit (e.g. 'chong2'). The text front-end cannot apply "
                "it and will read the character the standard way.",
                word=word,
            )
        if len(word) != 1:
            add(
                Severity.WARNING,
                "profile.multi_char_pronunciation",
                f"pronunciation override {word!r} covers more than one character; the "
                "lexicon is per character.",
                word=word,
            )


def _check_voice_binding(profile: CharacterProfile, add) -> None:
    if profile.llm.max_chars_per_reply > 200:
        add(
            Severity.WARNING,
            "profile.long_replies",
            f"max_chars_per_reply is {profile.llm.max_chars_per_reply}. Spec §27 lists "
            "over-long TTS chunks as a failure mode: long replies break the illusion "
            "and make chunking worse.",
        )
    if profile.memory.enabled and profile.memory.max_turns == 0:
        add(
            Severity.WARNING,
            "profile.memory_without_turns",
            "memory is enabled but max_turns is 0, so she remembers nothing within a "
            "conversation.",
        )


def _distinct_count(examples) -> int:
    seen: list[set[str]] = []
    for example in examples:
        grams = bigrams(example.character) or {normalize(example.character)}
        if any(_similar(grams, other) for other in seen):
            continue
        seen.append(grams)
    return len(seen)


def _similar(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return left == right
    union = len(left | right)
    return bool(union) and len(left & right) / union >= DUPLICATE_SIMILARITY
