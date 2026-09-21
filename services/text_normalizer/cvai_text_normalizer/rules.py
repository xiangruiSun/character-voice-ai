"""Regex rules that turn written Chinese into spoken Chinese.

Order is the whole design. A date must be matched before the plain integers inside it,
a percentage before the decimal it contains, and a long digit run before anything tries
to read it as a cardinal. The rules are therefore an explicit ordered list, not a dict,
and each records why it sits where it does.

Every rule reports what it changed, so the normalizer can show a reviewer exactly which
transformation produced a surprising reading — which is how a front-end bug gets
distinguished from an acoustic-model bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Final, Pattern

from . import numbers as N

#: Digit runs at least this long are read digit by digit rather than as a cardinal.
#: Phone numbers, order ids and verification codes are never read as "eight billion…".
LONG_DIGIT_RUN: Final = 7

CJK: Final = r"\u4e00-\u9fff"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: Pattern[str]
    replace: Callable[[re.Match[str]], str]
    why: str = ""

    def apply(self, text: str) -> tuple[str, int]:
        result, count = self.pattern.subn(self.replace, text)
        return result, count


def _r(
    name: str, pattern: str, replace: Callable[[re.Match[str]], str], why: str = ""
) -> Rule:
    return Rule(name=name, pattern=re.compile(pattern), replace=replace, why=why)


# --------------------------------------------------------------------------------------
# Dates and times — first, because they contain numbers that must not be read as numbers
# --------------------------------------------------------------------------------------


def _full_date(match: re.Match[str]) -> str:
    year, month, day = match.group("y"), match.group("m"), match.group("d")
    return (
        f"{N.year_to_chinese(year)}年"
        f"{N.month_day_to_chinese(month)}月"
        f"{N.month_day_to_chinese(day)}日"
    )


def _year_only(match: re.Match[str]) -> str:
    return N.year_to_chinese(match.group("y")) + "年"


def _month_day(match: re.Match[str]) -> str:
    return f"{N.month_day_to_chinese(match.group('m'))}月{N.month_day_to_chinese(match.group('d'))}日"


def _slash_date(match: re.Match[str]) -> str:
    return (
        f"{N.year_to_chinese(match.group('y'))}年"
        f"{N.month_day_to_chinese(match.group('m'))}月"
        f"{N.month_day_to_chinese(match.group('d'))}日"
    )


def _clock(match: re.Match[str]) -> str:
    return N.time_to_chinese(
        match.group("h"), match.group("mi"), match.groupdict().get("s")
    )


DATE_RULES: Final[tuple[Rule, ...]] = (
    _r(
        "date_full",
        r"(?P<y>\d{4})\s*[年/\-\.]\s*(?P<m>\d{1,2})\s*[月/\-\.]\s*(?P<d>\d{1,2})\s*日?",
        _full_date,
        "2026年3月17日 — the year is digit-by-digit, month and day are cardinal",
    ),
    _r(
        "date_slash",
        r"(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})",
        _slash_date,
        "2026/03/17",
    ),
    _r(
        "date_month_day",
        r"(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日",
        _month_day,
        "3月17日 without a year",
    ),
    _r("date_year", r"(?P<y>\d{4})\s*年", _year_only, "2026年 alone"),
    _r(
        "time_hms",
        r"\b(?P<h>[01]?\d|2[0-3]):(?P<mi>[0-5]\d):(?P<s>[0-5]\d)\b",
        _clock,
        "14:30:05",
    ),
    _r(
        "time_hm",
        r"\b(?P<h>[01]?\d|2[0-3]):(?P<mi>[0-5]\d)\b",
        _clock,
        "14:30 — must beat the fraction and range rules",
    ),
)


# --------------------------------------------------------------------------------------
# Quantities carrying a unit
# --------------------------------------------------------------------------------------


def _percent(match: re.Match[str]) -> str:
    return N.percent_to_chinese(match.group("n"))


def _permille(match: re.Match[str]) -> str:
    return N.permille_to_chinese(match.group("n"))


def _fraction(match: re.Match[str]) -> str:
    return N.fraction_to_chinese(match.group("a"), match.group("b"))


def _currency(match: re.Match[str]) -> str:
    symbol = match.group("c")
    unit = {"¥": "元", "￥": "元", "$": "美元", "€": "欧元", "£": "英镑"}.get(symbol, "元")
    return N.number_to_chinese(match.group("n")) + unit


def _temperature(match: re.Match[str]) -> str:
    scale = "摄氏度" if (match.group("s") or "C").upper() == "C" else "华氏度"
    return N.number_to_chinese(match.group("n")) + scale


UNIT_RULES: Final[tuple[Rule, ...]] = (
    _r(
        "percent",
        r"(?P<n>-?\d+(?:\.\d+)?)\s*%",
        _percent,
        "20% → 百分之二十; the unit leads in Chinese",
    ),
    _r("permille", r"(?P<n>-?\d+(?:\.\d+)?)\s*‰", _permille),
    _r(
        "currency",
        r"(?P<c>[¥￥$€£])\s*(?P<n>\d+(?:\.\d+)?)",
        _currency,
        "¥50 → 五十元",
    ),
    _r(
        "temperature",
        r"(?P<n>-?\d+(?:\.\d+)?)\s*°\s*(?P<s>[CF])?",
        _temperature,
    ),
    _r(
        "fraction",
        rf"(?<![\d:/])(?P<a>\d{{1,4}})/(?P<b>\d{{1,4}})(?![\d:/])",
        _fraction,
        "1/2 → 二分之一; runs after the date and time rules so it cannot eat them",
    ),
)


# --------------------------------------------------------------------------------------
# Bare numbers
# --------------------------------------------------------------------------------------


def _ordinal(match: re.Match[str]) -> str:
    return "第" + N.ordinal_to_chinese(match.group("n"))


def _long_run(match: re.Match[str]) -> str:
    return N.digit_by_digit(match.group(0))


def _range(match: re.Match[str]) -> str:
    return N.range_to_chinese(match.group("a"), match.group("b"))


def _plain_number(match: re.Match[str]) -> str:
    reading = N.number_to_chinese(match.group("n"))
    return N.apply_liang(reading, match.group("after") or "")


NUMBER_RULES: Final[tuple[Rule, ...]] = (
    _r("ordinal", r"第\s*(?P<n>\d+)", _ordinal, "第3 → 第三"),
    _r(
        "long_digit_run",
        rf"(?<!\d)\d{{{LONG_DIGIT_RUN},}}(?!\d)",
        _long_run,
        "phone numbers and ids are read digit by digit, never as a cardinal",
    ),
    _r(
        "range",
        rf"(?<=[{CJK}])\s*(?P<a>\d+)\s*[-–~～]\s*(?P<b>\d+)\s*(?=[{CJK}]|$)",
        _range,
        "3-5 between Chinese characters is a range; A7-3B is not, hence the CJK guard",
    ),
    _r(
        "plain",
        r"(?<![\dA-Za-z.])(?P<n>-?\d+(?:[.,]\d+)*)(?![\d.])(?P<after>.?)",
        lambda m: _plain_number(m) + (m.group("after") or ""),
        "everything left over, with 二/两 decided by what follows",
    ),
)


ALL_RULES: Final[tuple[Rule, ...]] = DATE_RULES + UNIT_RULES + NUMBER_RULES


# --------------------------------------------------------------------------------------
# Punctuation and symbols
# --------------------------------------------------------------------------------------

#: ASCII punctuation to its Chinese equivalent. TTS front-ends key prosody off these, and
#: a comma the model does not recognise is a pause that does not happen.
PUNCTUATION_MAP: Final[dict[str, str]] = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
    "[": "（",
    "]": "）",
}

#: Symbols with a spoken form.
SYMBOL_MAP: Final[dict[str, str]] = {
    "&": "和",
    "@": "at",
    "+": "加",
    "=": "等于",
    "#": "井号",
    "°": "度",
    "×": "乘",
    "÷": "除以",
}

#: Dropped outright: they carry no sound and some engines read them literally.
DROPPED_CHARS: Final[str] = "*_`~^|\\<>{}"

#: Collapsed to the standard Chinese ellipsis, which engines treat as a pause.
ELLIPSIS_PATTERN: Final[Pattern[str]] = re.compile(r"(?:\.{3,}|。{3,}|…{2,})")

REPEATED_PUNCT: Final[Pattern[str]] = re.compile(r"([，。！？；：、])\1{1,}")

#: Emoji and pictographs. A TTS engine either ignores them or, worse, reads the name.
EMOJI_PATTERN: Final[Pattern[str]] = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f000-\U0001f2ff"
    "\U0000fe00-\U0000fe0f"
    "\U00002190-\U000021ff"
    "]+",
    flags=re.UNICODE,
)

#: Markdown emphasis and code fences that leak out of an LLM.
MARKDOWN_PATTERN: Final[Pattern[str]] = re.compile(r"(\*{1,3}|_{2,}|`{1,3})")
