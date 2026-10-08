from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    rank: int               # 1-based rank in the result list
    score: float            # retriever-specific score (higher = more relevant)
    chunk_id: str
    source_doc_id: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider: str           # retriever provider (e.g. "bm25")
    retriever: str          # retriever name (e.g. "okapi")
