"""Latin letters, acronyms, product names and alphanumeric codes.

Spec §13 asks for a *configurable* spoken form for things like ``GPT-5.6``, and that is
the right requirement: there is no correct universal answer. A Chinese speaker reading
``PDF`` says the three letters; reading ``NASA`` they may say the word; reading
``A7-3B`` they spell it out with 杠 for the hyphen. Which one a given TTS engine
produces depends on the engine, so this is policy plus a lexicon rather than a rule.

This pass runs **before** the number rules, and that ordering does real work: without
it, ``A7-3B`` reaches the number rules and comes back as ``A7-三B``, because ``3``
after a hyphen looks exactly like a bare number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Pattern

from . import numbers as N


class LatinPolicy(str, Enum):
    """What to do with Latin text that the lexicon does not cover."""

    #: Leave it alone. Most modern Chinese TTS reads common acronyms acceptably.
    KEEP = "keep"
    #: Space the letters out, which makes engines read them individually.
    SPELL = "spell"
    #: Drop it. For packs where any Latin at all breaks the character's register.
    DROP = "drop"


#: Acronyms common enough in this domain to be worth a default reading. Users add their
#: own; these only exist so the first run on a technical sentence is not surprising.
DEFAULT_LEXICON: Final[dict[str, str]] = {
    "AI": "A I",
    "API": "A P I",
    "CPU": "C P U",
    "GPU": "G P U",
    "PDF": "P D F",
    "URL": "U R L",
    "USB": "U S B",
    "ID": "I D",
    "OK": "OK",
    "GPT": "G P T",
    "CUDA": "CUDA",
    "TTS": "T T S",
}

_TOKEN: Final[Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9]*")

# NOTE: these use explicit lookarounds rather than ``\b``. Python treats CJK as word
# characters, so there is no word boundary between 是 and GPT — and with ``\b`` the
# version pattern silently never fires in exactly the sentences it exists for, leaving
# the ``-5.6`` to be read later as *negative* 5.6.
_NOT_ALNUM_BEFORE: Final = r"(?<![A-Za-z0-9])"
_NOT_ALNUM_AFTER: Final = r"(?![A-Za-z0-9])"

#: ``GPT-5.6``, ``CUDA 12.8`` — a name followed by a version number.
_VERSIONED: Final[Pattern[str]] = re.compile(
    _NOT_ALNUM_BEFORE
    + r"(?P<name>[A-Za-z][A-Za-z0-9]*)\s*-?\s*(?P<version>\d+(?:\.\d+)+)"
    + _NOT_ALNUM_AFTER
)

#: ``A7-3B``, ``X1``, ``SN-2024A`` — identifiers mixing letters and digits.
_CODE: Final[Pattern[str]] = re.compile(
    _NOT_ALNUM_BEFORE
    + r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
    + r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
    + _NOT_ALNUM_AFTER
)


@dataclass
class LatinConfig:
    policy: LatinPolicy = LatinPolicy.SPELL
    lexicon: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LEXICON))
    #: Read the number in a version string as a decimal (``5.6`` → 五点六) rather than
    #: digit by digit (五点六 vs 五点六 — identical here, but ``12.8`` differs:
    #: 十二点八 vs 一二点八).
    version_as_decimal: bool = True
    #: Spell alphanumeric codes character by character, with 杠 for the hyphen.
    spell_codes: bool = True

    def spoken(self, token: str) -> str | None:
        """Lexicon lookup, case-insensitive, exact match only."""
        for key, value in self.lexicon.items():
            if key.lower() == token.lower():
                return value
        return None


def normalize_latin(text: str, config: LatinConfig | None = None) -> tuple[str, list[str]]:
    """Rewrite Latin tokens. Returns the text and a list of what changed."""
    settings = config or LatinConfig()
    changes: list[str] = []

    def note(before: str, after: str) -> None:
        if before != after:
            changes.append(f"{before} → {after}")

    def version(match: re.Match[str]) -> str:
        name, number = match.group("name"), match.group("version")
        spoken_name = _render_token(name, settings)
        spoken_number = (
            N.decimal_to_chinese(number)
            if settings.version_as_decimal and number.count(".") == 1
            else N.digit_by_digit(number)
        )
        result = f"{spoken_name}{spoken_number}".strip()
        note(match.group(0), result)
        return result

    text = _VERSIONED.sub(version, text)

    if settings.spell_codes:

        def code(match: re.Match[str]) -> str:
            token = match.group(0)
            replacement = settings.spoken(token)
            if replacement is None:
                replacement = _spell_code(token)
            note(token, replacement)
            return replacement

        text = _CODE.sub(code, text)

    def token(match: re.Match[str]) -> str:
        word = match.group(0)
        result = _render_token(word, settings)
        note(word, result)
        return result

    text = _TOKEN.sub(token, text)
    return text, changes


def _render_token(token: str, settings: LatinConfig) -> str:
    replacement = settings.spoken(token)
    if replacement is not None:
        return _pad(replacement)
    if settings.policy is LatinPolicy.DROP:
        return ""
    if settings.policy is LatinPolicy.SPELL and _looks_like_acronym(token):
        return _pad(" ".join(token.upper()))
    return _pad(token)


def _looks_like_acronym(token: str) -> bool:
    """All-caps and short. ``NASA`` counts; ``Hello`` does not.

    Lower-case words are left alone: spelling out ``hello`` letter by letter is much
    worse than letting the engine read it.
    """
    return token.isupper() and 2 <= len(token) <= 6


def _spell_code(token: str) -> str:
    """``A7-3B`` → ``A 七 杠 三 B``."""
    parts: list[str] = []
    for char in token:
        if char.isdigit():
            parts.append(N.DIGITS[int(char)])
        elif char == "-":
            parts.append("杠")
        else:
            parts.append(char.upper())
    return _pad(" ".join(parts))


def _pad(text: str) -> str:
    """Keep a space around Latin so it does not fuse with adjacent Chinese.

    Collapsed later by the whitespace pass; here it only has to prevent ``的PDF`` from
    becoming one unsegmentable token.
    """
    return f" {text} " if text else ""
