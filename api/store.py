"""In-memory document store shared across API routes.

Persists for the lifetime of the API process. Each phase reads/writes here:
  Phase 1 (ingestion)  → writes LoadedDocument
  Phase 2 (chunking)   → reads LoadedDocument, writes Chunk
  Phase 3+ (embedding, retrieval, …) → reads/writes their own types
"""
from __future__ import annotations
from core.models.documents import LoadedDocument


class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, LoadedDocument] = {}

    def add(self, docs: list[LoadedDocument]) -> None:
        for d in docs:
            self._docs[d.id] = d

    def get(self, doc_id: str) -> LoadedDocument | None:
        return self._docs.get(doc_id)

    def all(self) -> list[LoadedDocument]:
        return list(self._docs.values())

    def delete(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None

    def clear(self) -> None:
        self._docs.clear()

    def count(self) -> int:
        return len(self._docs)


document_store = DocumentStore()
