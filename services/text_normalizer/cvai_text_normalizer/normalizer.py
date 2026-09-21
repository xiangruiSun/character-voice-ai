"""The Chinese text normalizer (spec §13).

    raw LLM text → clean → Latin/acronyms → numbers, dates, units → punctuation
                 → per-character pronunciation hints → TTS-ready text

Spec §13's point is that "raw LLM text is not always appropriate TTS input", and the
failure is quiet: the model says 二千零二十六年 instead of 二零二六年, or reads a
product code letter by letter in the middle of a sentence, and it sounds like the voice
model is bad when the front end is at fault.

Library first, rules second. When `wetext` (the pynini-free WeTextProcessing runtime) or
`cn2an` is installed, they do the heavy lifting; the built-in rules then run over
whatever is left as a safety net. With neither installed the built-in rules do the whole
job — the behaviour is pinned by tests either way, which is why they exist.

What stays ours, always:

* the per-character pronunciation lexicon from the :class:`CharacterProfile`, because no
  general tool knows that *this* character says a polyphone a particular way;
* Latin and acronym policy, because the right answer depends on the engine;
* punctuation mapping, because Chinese TTS keys its prosody off 。？！…… and an ASCII
  comma is a pause that does not happen.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from cvai_types import CharacterProfile, CVAIModel
from pydantic import Field

from . import rules as R
from .latin import LatinConfig, LatinPolicy, normalize_latin

#: Characters that survive into the final text. Anything else is flagged.
_ALLOWED_OTHER: Final = set(" \n，。！？；：、…—（）《》“”‘’·")


class NormalizerConfig(CVAIModel):
    """Everything tunable about the front end."""

    #: Prefer installed libraries over the built-in rules when available.
    use_wetext: bool = True
    use_cn2an: bool = True
    #: Remove the 儿 of erhua, which some engines mispronounce.
    remove_erhua: bool = False

    latin_policy: LatinPolicy = LatinPolicy.SPELL
    latin_lexicon: dict[str, str] = Field(default_factory=dict)
    spell_codes: bool = True
    version_as_decimal: bool = True

    map_punctuation: bool = True
    strip_emoji: bool = True
    strip_markdown: bool = True
    collapse_repeated_punctuation: bool = True
    #: Longest run of ellipsis to keep. Engines pause per ellipsis; five in a row is a
    #: three-second silence nobody asked for.
    max_ellipsis: int = Field(default=1, ge=1, le=4)

    #: Per-character pinyin overrides, e.g. ``{"重": "chong2"}``. Loaded from the
    #: character profile; see :func:`from_character`.
    pronunciation_overrides: dict[str, str] = Field(default_factory=dict)


class Replacement(CVAIModel):
    rule: str
    detail: str = ""


class NormalizedText(CVAIModel):
    """The result, with enough detail to debug a surprising reading."""

    original: str
    text: str
    replacements: list[Replacement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: ``{character: pinyin}`` for characters this character pronounces unusually.
    #: Engines that accept inline pinyin can consume these; the rest ignore them.
    pinyin_hints: dict[str, str] = Field(default_factory=dict)
    backend: str = "builtin"

    @property
    def changed(self) -> bool:
        return self.original != self.text


@dataclass
class _Backends:
    wetext: object | None = None
    cn2an: object | None = None
    names: list[str] = field(default_factory=list)


class ChineseTextNormalizer:
    """Normalize one utterance at a time. Cheap to construct, safe to reuse."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self.latin = LatinConfig(
            policy=self.config.latin_policy,
            lexicon={**LatinConfig().lexicon, **self.config.latin_lexicon},
            spell_codes=self.config.spell_codes,
            version_as_decimal=self.config.version_as_decimal,
        )
        self._backends = self._resolve_backends()

    # -- construction ---------------------------------------------------------------

    @classmethod
    def from_character(
        cls, profile: CharacterProfile, config: NormalizerConfig | None = None
    ) -> "ChineseTextNormalizer":
        """Build a normalizer carrying one character's pronunciation habits.

        Spec §11 puts ``pronunciation_overrides`` on the character, not the voice pack,
        and that is right: it is a property of how she speaks, and it should follow her
        across a change of TTS engine.
        """
        settings = (config or NormalizerConfig()).model_copy(
            update={
                "pronunciation_overrides": {
                    **(config.pronunciation_overrides if config else {}),
                    **profile.speaking_habits.pronunciation_overrides,
                }
            }
        )
        return cls(settings)

    def _resolve_backends(self) -> _Backends:
        backends = _Backends()
        if self.config.use_wetext:
            try:
                from wetext import Normalizer  # noqa: PLC0415

                backends.wetext = Normalizer(
                    lang="zh", operator="tn", remove_erhua=self.config.remove_erhua
                )
                backends.names.append("wetext")
            except Exception:  # noqa: BLE001 - optional, and its import can be heavy
                backends.wetext = None
        if self.config.use_cn2an:
            try:
                import cn2an  # noqa: PLC0415

                backends.cn2an = cn2an
                backends.names.append("cn2an")
            except ImportError:
                backends.cn2an = None
        return backends

    @property
    def backend_name(self) -> str:
        return "+".join(self._backends.names) or "builtin"

    # -- main entry point -----------------------------------------------------------

    def normalize(self, text: str) -> NormalizedText:
        original = text
        replacements: list[Replacement] = []
        warnings: list[str] = []

        working = self._preclean(text)
        working, latin_changes = normalize_latin(working, self.latin)
        replacements += [Replacement(rule="latin", detail=d) for d in latin_changes]

        working, library_note = self._library_pass(working)
        if library_note:
            replacements.append(Replacement(rule="library", detail=library_note))

        working, rule_hits = self._rule_pass(working)
        replacements += rule_hits

        working = self._punctuation(working)
        working = self._collapse_whitespace(working)

        hints = self._pronunciation_hints(working)
        warnings += self._validate(working)

        return NormalizedText(
            original=original,
            text=working,
            replacements=replacements,
            warnings=warnings,
            pinyin_hints=hints,
            backend=self.backend_name,
        )

    def __call__(self, text: str) -> str:
        """Convenience for callers that only want the string."""
        return self.normalize(text).text

    # -- stages ----------------------------------------------------------------------

    def _preclean(self, text: str) -> str:
        # NFKC folds full-width ASCII (ＧＰＴ, ３) onto the normal forms the rules match,
        # which is otherwise a silent source of "why didn't the number rule fire".
        text = unicodedata.normalize("NFKC", text)
        text = "".join(
            ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C"
        )
        if self.config.strip_emoji:
            text = R.EMOJI_PATTERN.sub("", text)
        if self.config.strip_markdown:
            text = R.MARKDOWN_PATTERN.sub("", text)
        text = "".join(ch for ch in text if ch not in R.DROPPED_CHARS)
        return text.strip()

    def _library_pass(self, text: str) -> tuple[str, str]:
        """Run an installed normalizer, if any. Failure is non-fatal."""
        if self._backends.wetext is not None:
            try:
                result = self._backends.wetext.normalize(text)  # type: ignore[attr-defined]
                if isinstance(result, str) and result.strip():
                    return result, "wetext"
            except Exception as exc:  # noqa: BLE001
                return text, f"wetext failed ({exc}); using built-in rules"
        if self._backends.cn2an is not None:
            try:
                result = self._backends.cn2an.transform(text, "an2cn")  # type: ignore[attr-defined]
                if isinstance(result, str) and result.strip():
                    return result, "cn2an"
            except Exception as exc:  # noqa: BLE001
                return text, f"cn2an failed ({exc}); using built-in rules"
        return text, ""

    def _rule_pass(self, text: str) -> tuple[str, list[Replacement]]:
        """Built-in rules, in order. Also the safety net after a library pass."""
        hits: list[Replacement] = []
        for rule in R.ALL_RULES:
            text, count = rule.apply(text)
            if count:
                hits.append(Replacement(rule=rule.name, detail=f"×{count}"))
        return text, hits

    def _punctuation(self, text: str) -> str:
        if self.config.map_punctuation:
            # Only ASCII punctuation between CJK becomes Chinese punctuation; a period
            # inside a number that survived the rules should not become 。
            text = re.sub(
                rf"(?<=[{R.CJK}])([,.?!;:])(?=[{R.CJK}]|\s|$)",
                lambda m: R.PUNCTUATION_MAP.get(m.group(1), m.group(1)),
                text,
            )
            text = re.sub(
                r"([?!,;:])(?=\s|$)",
                lambda m: R.PUNCTUATION_MAP.get(m.group(1), m.group(1)),
                text,
            )
            for source, target in R.SYMBOL_MAP.items():
                text = text.replace(source, target)

        text = R.ELLIPSIS_PATTERN.sub("……", text)
        if self.config.max_ellipsis:
            limit = self.config.max_ellipsis
            text = re.sub(r"(?:……){%d,}" % (limit + 1), "……" * limit, text)
        if self.config.collapse_repeated_punctuation:
            text = R.REPEATED_PUNCT.sub(r"\1", text)
        return text

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        # A space between two Chinese characters is an artefact of the Latin padding and
        # becomes an audible hesitation in some engines.
        text = re.sub(rf"(?<=[{R.CJK}]) (?=[{R.CJK}])", "", text)
        text = re.sub(rf"\s+(?=[，。！？；：、…）])", "", text)
        return text.strip()

    def _pronunciation_hints(self, text: str) -> dict[str, str]:
        overrides = self.config.pronunciation_overrides
        if not overrides:
            return {}
        return {char: pinyin for char, pinyin in overrides.items() if char in text}

    def _validate(self, text: str) -> list[str]:
        """Flag anything a TTS engine is likely to read badly."""
        warnings: list[str] = []
        leftover_digits = re.findall(r"\d+", text)
        if leftover_digits:
            warnings.append(
                f"digits survived normalization: {leftover_digits[:5]} — the engine "
                "will guess how to read them"
            )
        odd = {
            ch
            for ch in text
            if not _is_expected(ch)
        }
        if odd:
            warnings.append(f"unusual characters remain: {sorted(odd)[:8]}")
        if not text.strip():
            warnings.append("normalization produced empty text")
        return warnings


def _is_expected(char: str) -> bool:
    """Whether a character is something a Chinese TTS engine will handle sensibly."""
    if char.isspace() or char in _ALLOWED_OTHER:
        return True
    if "一" <= char <= "鿿":  # CJK
        return True
    return char.isalnum()  # Latin letters and digits are reported separately


def normalize(text: str, config: NormalizerConfig | None = None) -> str:
    """One-shot helper for callers that do not want to hold a normalizer."""
    return ChineseTextNormalizer(config).normalize(text).text
