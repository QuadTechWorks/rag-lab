"""Helpers shared by retriever providers."""
from __future__ import annotations
import re
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens — shared by every lexical retriever."""
    return _TOKEN_RE.findall(text.lower())


def matches_filters(metadata: dict[str, Any], filters: dict | None) -> bool:
    """Exact-match metadata filter (string-compared, like the vector stores store it)."""
    if not filters:
        return True
    return all(str(metadata.get(k)) == str(v) for k, v in filters.items())
