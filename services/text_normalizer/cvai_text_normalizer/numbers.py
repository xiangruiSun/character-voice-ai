"""Reading numbers aloud in Mandarin.

``cn2an`` and WeTextProcessing do this well and the normalizer prefers them when they
are installed. This module is the fallback, and it is a real implementation rather than
a stub: number reading is the single most visible thing a Chinese TTS front-end gets
wrong, it is deterministic, and having it in-repo means the behaviour is pinned by tests
instead of varying with whatever version of a dependency is present.

The cases that actually bite, all handled here:

* ``一十`` vs ``十`` — 10 is 十, not 一十, but 110 is 一百一十.
* zero insertion across groups — 10001 is 一万零一, not 一万一.
* ``二`` vs ``两`` — 两个 and 两点, but 十二个 and 二月.
* years are read digit by digit (2026 → 二零二六) while day-of-month is cardinal
  (17日 → 十七日). Getting this backwards is the classic giveaway.
* decimals are cardinal before the point and digit-by-digit after: 78.5 → 七十八点五.
"""

from __future__ import annotations

from typing import Final

DIGITS: Final = "零一二三四五六七八九"
UNITS4: Final = ("", "十", "百", "千")
#: Section units, ascending. 万亿 rather than 兆 — 兆 is ambiguous in Mandarin.
SECTIONS: Final = ("", "万", "亿", "万亿", "亿亿")

NEGATIVE: Final = "负"
POINT: Final = "点"

#: Measure words and time units that take 两 rather than 二. Not exhaustive — it does not
#: need to be; an unlisted one simply reads 二, which is understandable rather than wrong.
LIANG_MEASURES: Final = frozenset(
    "个只件条本台次张份块位名家人天年月周秒分点时倍层双对杯瓶碗支把副群批轮圈步口句段"
)

#: Multi-character measure words that also take 两.
LIANG_MEASURE_WORDS: Final = (
    "小时",
    "分钟",
    "公里",
    "千米",
    "毫升",
    "公斤",
    "千克",
    "百万",
    "亿",
    "万",
    "千",
    "百",
)

MAX_CARDINAL: Final = 10**16


def digit_by_digit(text: str) -> str:
    """``2026`` → ``二零二六``. Used for years, IDs, phone numbers."""
    out: list[str] = []
    for char in text:
        if char.isdigit():
            out.append(DIGITS[int(char)])
        elif char in "-－—":
            out.append("杠")
        else:
            out.append(char)
    return "".join(out)


def _read_group(value: int) -> str:
    """Read 1..9999 with 千百十 and internal zeros."""
    if value == 0:
        return ""
    digits = [int(c) for c in str(value)]
    length = len(digits)
    out: list[str] = []
    zero_pending = False
    for index, digit in enumerate(digits):
        unit = UNITS4[length - 1 - index]
        if digit == 0:
            zero_pending = True
            continue
        if zero_pending and out:
            out.append(DIGITS[0])
        zero_pending = False
        out.append(DIGITS[digit] + unit)
    return "".join(out)


def integer_to_chinese(value: int) -> str:
    """Cardinal reading of an integer. ``1250`` → ``一千二百五十``."""
    if value < 0:
        return NEGATIVE + integer_to_chinese(-value)
    if value == 0:
        return DIGITS[0]
    if value >= MAX_CARDINAL:
        # Beyond 亿亿 the grouping gets exotic and nobody reads it as a cardinal anyway.
        return digit_by_digit(str(value))

    groups: list[int] = []
    remainder = value
    while remainder > 0:
        groups.append(remainder % 10000)
        remainder //= 10000

    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            # A zero group only needs a 零 if something non-zero follows it.
            if parts and any(groups[lower] for lower in range(index)):
                if not parts[-1].endswith(DIGITS[0]):
                    parts.append(DIGITS[0])
            continue
        # A group under 1000 that is not the leading one needs a 零 in front:
        # 10001 is 一万零一, not 一万一.
        if parts and group < 1000 and not parts[-1].endswith(DIGITS[0]):
            parts.append(DIGITS[0])
        parts.append(_read_group(group) + SECTIONS[index])

    text = "".join(parts)
    # 10 is 十, not 一十 — but only at the very start: 一百一十 keeps its 一十.
    if text.startswith("一十"):
        text = text[1:]
    return text


def decimal_to_chinese(text: str) -> str:
    """``78.5`` → ``七十八点五``; the fractional part is read digit by digit."""
    negative = text.startswith("-")
    body = text.lstrip("+-")
    if "." not in body:
        return (NEGATIVE if negative else "") + integer_to_chinese(int(body or "0"))

    whole, _, fraction = body.partition(".")
    whole_text = integer_to_chinese(int(whole)) if whole else DIGITS[0]
    fraction_text = digit_by_digit(fraction)
    return (NEGATIVE if negative else "") + whole_text + POINT + fraction_text


def number_to_chinese(text: str) -> str:
    """Read a bare numeric token, decimal or not."""
    cleaned = text.replace(",", "").replace("，", "")
    return decimal_to_chinese(cleaned)


def apply_liang(reading: str, following: str) -> str:
    """Turn a standalone ``二`` into ``两`` before a measure word.

    Only when the whole number is 2: 两个 and 两点, but 十二个 and 二十二个 keep 二.
    Speakers use 两 for the bare numeral, not for a 2 that happens to be the units
    digit of a larger number.
    """
    if reading != "二" or not following:
        return reading
    if any(following.startswith(word) for word in LIANG_MEASURE_WORDS):
        return "两"
    if following[0] in LIANG_MEASURES:
        return "两"
    return reading


def year_to_chinese(text: str) -> str:
    """Years are read digit by digit: 2026 → 二零二六."""
    return digit_by_digit(text)


def month_day_to_chinese(value: str) -> str:
    """Month and day are cardinal: 17 → 十七."""
    try:
        return integer_to_chinese(int(value))
    except ValueError:
        return digit_by_digit(value)


def time_to_chinese(hour: str, minute: str, second: str | None = None) -> str:
    """``14:30`` → ``十四点三十分``; ``2:05`` → ``两点零五分``; ``:00`` → ``整``."""
    hour_value = int(hour)
    minute_value = int(minute)

    hour_text = apply_liang(integer_to_chinese(hour_value), "点") + "点"

    if minute_value == 0:
        minute_text = "整"
    elif minute_value == 30 and second in (None, "", "00"):
        minute_text = "半"
    elif minute_value < 10:
        minute_text = DIGITS[0] + integer_to_chinese(minute_value) + "分"
    else:
        minute_text = integer_to_chinese(minute_value) + "分"

    text = hour_text + minute_text
    if second and int(second) > 0:
        text += integer_to_chinese(int(second)) + "秒"
    return text


def percent_to_chinese(number: str) -> str:
    """``20%`` → ``百分之二十``. The unit leads in Chinese, which trips up naive rules."""
    return "百分之" + number_to_chinese(number)


def permille_to_chinese(number: str) -> str:
    return "千分之" + number_to_chinese(number)


def fraction_to_chinese(numerator: str, denominator: str) -> str:
    """``1/2`` → ``二分之一``: denominator first."""
    return f"{number_to_chinese(denominator)}分之{number_to_chinese(numerator)}"


def ordinal_to_chinese(value: str) -> str:
    return integer_to_chinese(int(value))


def range_to_chinese(low: str, high: str) -> str:
    return f"{number_to_chinese(low)}到{number_to_chinese(high)}"
