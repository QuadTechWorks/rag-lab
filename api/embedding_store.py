"""In-memory store for EmbeddedChunk objects (Phase 3)."""
from __future__ import annotations
from core.models.embeddings import EmbeddedChunk


class EmbeddingStore:
    def __init__(self) -> None:
        self._store: dict[str, EmbeddedChunk] = {}  # keyed by chunk_id

    def add(self, embeddings: list[EmbeddedChunk]) -> None:
        for ec in embeddings:
            self._store[ec.chunk_id] = ec

    def all(self) -> list[EmbeddedChunk]:
        return list(self._store.values())

    def by_chunk(self, chunk_id: str) -> EmbeddedChunk | None:
        return self._store.get(chunk_id)

    def by_doc(self, doc_id: str) -> list[EmbeddedChunk]:
        return [ec for ec in self._store.values() if ec.source_doc_id == doc_id]

    def get_page(self, doc_id: str | None, limit: int, offset: int) -> list[EmbeddedChunk]:
        items = self.by_doc(doc_id) if doc_id else self.all()
        return items[offset: offset + limit]

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        items = self.all()
        if not items:
            return {"count": 0, "vector_dim": 0, "avg_chars": 0,
                    "providers": [], "models": []}
        dims = [ec.vector_dim for ec in items]
        return {
            "count": len(items),
            "vector_dim": dims[0] if dims else 0,
            "avg_chars": int(sum(ec.char_count for ec in items) / len(items)),
            "providers": list({ec.provider for ec in items}),
            "embedders": list({ec.embedder for ec in items}),
            "models": list({ec.model for ec in items}),
        }


embedding_store = EmbeddingStore()
