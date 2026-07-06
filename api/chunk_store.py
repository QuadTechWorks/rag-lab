"""In-memory chunk store — Phase 2 (Chunking).

Persists for the lifetime of the API process.
Phase 3 (Embedding) reads from here and writes vectors to the vector store.
"""
from __future__ import annotations
from core.models.chunks import DocumentChunk


class ChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def add(self, chunks: list[DocumentChunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def get(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)

    def all(self) -> list[DocumentChunk]:
        return list(self._chunks.values())

    def by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return sorted(
            [c for c in self._chunks.values() if c.source_doc_id == doc_id],
            key=lambda c: c.chunk_index,
        )

    def delete_by_doc(self, doc_id: str) -> int:
        keys = [k for k, v in self._chunks.items() if v.source_doc_id == doc_id]
        for k in keys:
            del self._chunks[k]
        return len(keys)

    def clear(self) -> None:
        self._chunks.clear()

    def count(self) -> int:
        return len(self._chunks)

    def stats(self) -> dict:
        chunks = self.all()
        if not chunks:
            return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}
        sizes = [c.char_count for c in chunks]
        return {
            "count": len(chunks),
            "avg_chars": round(sum(sizes) / len(sizes)),
            "min_chars": min(sizes),
            "max_chars": max(sizes),
        }


chunk_store = ChunkStore()
