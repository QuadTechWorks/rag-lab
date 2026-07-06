from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class SearchResult(BaseModel):
    rank: int               # 1-based rank in the result list
    score: float            # similarity score (higher = more similar)
    chunk_id: str
    source_doc_id: str
    source: str
    content: str
    metadata: dict[str, Any]
    provider: str           # which vector store returned this
    store: str              # store name
