from __future__ import annotations
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult


@VECTOR_STORES.register(provider="faiss", name="flat")
class FAISSFlatStore(BaseVectorStore):
    """FAISS IndexFlatIP — exact cosine search, in-memory.

    No training required. Best for small-to-medium datasets (<1M vectors).
    Data is in-memory only — cleared on API restart.
    Requires: pip install faiss-cpu
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index = None
        self._dim: int | None = None
        self._meta: list[dict] = []         # parallel metadata list
        self._id_to_idx: dict[str, int] = {}  # chunk_id → list index

    def _ensure_index(self, dim: int) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            self._dim = dim

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": len(self._meta)}
        import numpy as np
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")

        dim = embedded_chunks[0].vector_dim
        self._ensure_index(dim)

        new_chunks = []
        for ec in embedded_chunks:
            if ec.chunk_id in self._id_to_idx:
                idx = self._id_to_idx[ec.chunk_id]
                self._meta[idx] = _to_meta(ec)
            else:
                self._id_to_idx[ec.chunk_id] = len(self._meta)
                self._meta.append(_to_meta(ec))
                new_chunks.append(ec)

        if new_chunks:
            vectors = np.array([ec.vector for ec in new_chunks], dtype=np.float32)
            # L2-normalize for cosine similarity via inner product
            faiss.normalize_L2(vectors)
            self._index.add(vectors)

        return {"indexed": len(embedded_chunks), "total": self._index.ntotal}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import numpy as np
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            results.append(SearchResult(
                rank=rank + 1,
                score=float(score),
                chunk_id=m["chunk_id"],
                source_doc_id=m["source_doc_id"],
                source=m["source"],
                content=m["content"],
                metadata=m.get("metadata", {}),
                provider="faiss",
                store="flat",
            ))
        return results

    def delete(self, ids: list[str]) -> int:
        deleted = 0
        for cid in ids:
            if cid in self._id_to_idx:
                idx = self._id_to_idx.pop(cid)
                if idx < len(self._meta):
                    self._meta[idx] = {}  # tombstone
                deleted += 1
        return deleted

    def clear(self) -> None:
        self._index = None
        self._dim = None
        self._meta = []
        self._id_to_idx = {}

    def stats(self) -> dict:
        return {
            "count": self._index.ntotal if self._index else 0,
            "store": "faiss/flat",
            "vector_dim": self._dim,
            "in_memory": True,
        }

    @property
    def is_ready(self) -> bool:
        return True  # FAISS needs no external connection

    @property
    def store_name(self) -> str:
        return "FAISS IndexFlatIP (exact cosine, in-memory)"


def _to_meta(ec: EmbeddedChunk) -> dict:
    return {
        "chunk_id": ec.chunk_id,
        "source_doc_id": ec.source_doc_id,
        "source": ec.source,
        "content": ec.content,
        "metadata": ec.metadata,
    }
