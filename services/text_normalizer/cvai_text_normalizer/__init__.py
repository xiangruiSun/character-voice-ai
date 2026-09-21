"""Chinese text front-end (Milestone 8, spec §13 and §14).

Two jobs, both between the LLM and the TTS engine:

* **Normalization** — turn written Chinese into spoken Chinese. Numbers, dates, times,
  percentages, currency, Latin acronyms, product codes, punctuation, emoji, and a
  per-character pronunciation lexicon.
* **Chunking** — split a streamed reply into pieces a TTS engine can speak naturally,
  avoiding both token-by-token synthesis and paragraph-sized requests (spec §27).

Libraries first: `wetext` (the pynini-free WeTextProcessing runtime) and `cn2an` are
used when installed. The built-in rules run either way — as the whole implementation on
a bare install, and as a safety net after a library pass. Both paths are covered by
tests, so the behaviour does not depend on what happens to be installed.
"""

from __future__ import annotations

from .chunking import (
    CONTINUATIONS,
    SOFT_BOUNDARIES,
    STRONG_BOUNDARIES,
    ChunkerConfig,
    SpeechChunker,
    ends_completely,
    estimate_chunks,
)
from .latin import DEFAULT_LEXICON, LatinConfig, LatinPolicy, normalize_latin
from .normalizer import (
    ChineseTextNormalizer,
    NormalizedText,
    NormalizerConfig,
    Replacement,
    normalize,
)
from .numbers import (
    decimal_to_chinese,
    digit_by_digit,
    integer_to_chinese,
    number_to_chinese,
    percent_to_chinese,
    time_to_chinese,
    year_to_chinese,
)

__all__ = [
    "CONTINUATIONS",
    "ChineseTextNormalizer",
    "ChunkerConfig",
    "DEFAULT_LEXICON",
    "LatinConfig",
    "LatinPolicy",
    "NormalizedText",
    "NormalizerConfig",
    "Replacement",
    "SOFT_BOUNDARIES",
    "STRONG_BOUNDARIES",
    "SpeechChunker",
    "decimal_to_chinese",
    "digit_by_digit",
    "ends_completely",
    "estimate_chunks",
    "integer_to_chinese",
    "normalize",
    "normalize_latin",
    "number_to_chinese",
    "percent_to_chinese",
    "time_to_chinese",
    "year_to_chinese",
]
