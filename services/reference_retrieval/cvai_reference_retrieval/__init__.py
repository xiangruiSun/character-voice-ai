"""Reference retrieval (spec §10).

V1 uses metadata rules, as the spec asks. The interface is written so that embedding
retrieval can replace the scoring function later without any caller noticing.
"""

from __future__ import annotations

from .rule_based import RuleBasedReferenceRetriever, ReferenceScore

__all__ = ["RuleBasedReferenceRetriever", "ReferenceScore"]
