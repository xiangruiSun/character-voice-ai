"""Base model and shared primitives for every typed schema in the system.

V1 is Chinese-only by decision D6 in ``docs/IMPLEMENTATION_PLAN.md``. That is enforced
here as a validation error rather than a convention, because the cheapest place to stop
multilingual scope creep is at the schema boundary.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------------------

#: The only language V1 supports. See spec §25 (non-goals) and decision D6.
LANGUAGE_ZH_CN: Final = "zh-CN"

Language = Literal["zh-CN"]

# --------------------------------------------------------------------------------------
# Constrained scalar aliases
# --------------------------------------------------------------------------------------

#: A 0.0-1.0 score. Used for emotion intensity, quality and confidence.
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]

#: Identifier slug: lowercase, digits and underscores; must start with a letter.
SLUG_PATTERN: Final = r"^[a-z][a-z0-9_]*$"
Slug = Annotated[str, Field(pattern=SLUG_PATTERN, min_length=1, max_length=64)]

#: A style tag. Same shape as a slug but semantically distinct: it may be one of the
#: core styles (spec §9) or a character-specific style such as ``soft_teasing``.
StyleTag = Annotated[str, Field(pattern=SLUG_PATTERN, min_length=1, max_length=64)]

#: Relative POSIX path inside a voice pack. Absolute paths and ``..`` are rejected so a
#: pack stays relocatable and a manifest can never reach outside its own directory.
RelPath = Annotated[str, Field(min_length=1, max_length=512)]


class CVAIModel(BaseModel):
    """Base for all project models.

    ``extra="forbid"`` is deliberate: these models are loaded from hand-edited YAML, and
    a silently ignored typo in a config key is one of the harder bugs to find in a voice
    pipeline where the symptom is "it sounds slightly worse".
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="float",
    )


def utcnow() -> _dt.datetime:
    """Timezone-aware UTC now. Every timestamp in a manifest uses this."""
    return _dt.datetime.now(_dt.timezone.utc)


def validate_relative_path(value: str) -> str:
    """Reject absolute paths, parent traversal and Windows separators."""
    if value.startswith(("/", "\\")):
        raise ValueError(f"path must be relative to the voice pack root, got {value!r}")
    if "\\" in value:
        raise ValueError(f"use POSIX separators in manifests, got {value!r}")
    parts = value.split("/")
    if ".." in parts:
        raise ValueError(f"path must not traverse upwards, got {value!r}")
    if any(p == "" for p in parts):
        raise ValueError(f"path must not contain empty segments, got {value!r}")
    return value


def sha256_text(text: str) -> str:
    """Stable content hash, used for config hashes and run identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(value: str) -> tuple[int, int, int]:
    """Parse ``1.2.3`` / ``v1.2.3``. Voice pack and checkpoint versions use this."""
    match = _VERSION_RE.match(value.strip())
    if not match:
        raise ValueError(f"expected a semantic version like '1.0.0', got {value!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


SemVer = Annotated[str, Field(pattern=r"^v?\d+\.\d+\.\d+$")]
