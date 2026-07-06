from __future__ import annotations
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult


@VECTOR_STORES.register(provider="faiss", name="ivf")
class FAISSIVFStore(BaseVectorStore):
    """FAISS IndexIVFFlat — approximate cosine search, in-memory.

    Faster than flat for large datasets (>10K vectors). Requires training.
    Buffered: collects vectors, trains on first search when ≥ nlist vectors.
    Data is in-memory only — cleared on API restart.
    Requires: pip install faiss-cpu
    """

    def __init__(self, nlist: int = 100, nprobe: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._nlist = nlist
        self._nprobe = nprobe
        self._index = None
        self._dim: int | None = None
        self._buffer: list[list[float]] = []     # pending vectors before training
        self._meta: list[dict] = []
        self._id_to_idx: dict[str, int] = {}
        self._trained = False

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": len(self._meta)}
        import numpy as np
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")

        dim = embedded_chunks[0].vector_dim
        if self._index is None:
            self._dim = dim
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, self._nlist,
                                             faiss.METRIC_INNER_PRODUCT)

        for ec in embedded_chunks:
            if ec.chunk_id not in self._id_to_idx:
                self._id_to_idx[ec.chunk_id] = len(self._meta)
                self._meta.append(_to_meta(ec))
                self._buffer.append(ec.vector)
            else:
                self._meta[self._id_to_idx[ec.chunk_id]] = _to_meta(ec)

        # Train + add when we have enough vectors
        if not self._trained and len(self._buffer) >= self._nlist:
            vecs = np.array(self._buffer, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self._index.train(vecs)
            self._index.add(vecs)
            self._index.nprobe = min(self._nprobe, self._nlist)
            self._trained = True
            self._buffer = []

        return {"indexed": len(embedded_chunks), "total": len(self._meta),
                "trained": self._trained,
                "pending": len(self._buffer),
                "needs": max(0, self._nlist - len(self._meta)) if not self._trained else 0}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        if not self._trained:
            # Fall back to linear scan over buffer
            return _linear_search(self._buffer, self._meta, query_vector, top_k,
                                  provider="faiss", store="ivf")
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
                rank=rank + 1, score=float(score),
                chunk_id=m["chunk_id"], source_doc_id=m["source_doc_id"],
                source=m["source"], content=m["content"],
                metadata=m.get("metadata", {}), provider="faiss", store="ivf",
            ))
        return results

    def delete(self, ids: list[str]) -> int:
        deleted = 0
        for cid in ids:
            if cid in self._id_to_idx:
                self._id_to_idx.pop(cid)
                deleted += 1
        return deleted

    def clear(self) -> None:
        self._index = None
        self._buffer = []
        self._meta = []
        self._id_to_idx = {}
        self._trained = False

    def stats(self) -> dict:
        return {
            "count": len(self._meta),
            "indexed": self._index.ntotal if self._trained and self._index else 0,
            "store": "faiss/ivf",
            "vector_dim": self._dim,
            "trained": self._trained,
            "pending_buffer": len(self._buffer),
            "in_memory": True,
        }

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def store_name(self) -> str:
        return "FAISS IndexIVFFlat (approximate cosine, in-memory)"


def _to_meta(ec: EmbeddedChunk) -> dict:
    return {"chunk_id": ec.chunk_id, "source_doc_id": ec.source_doc_id,
            "source": ec.source, "content": ec.content, "metadata": ec.metadata}


def _linear_search(buffer: list[list[float]], meta: list[dict],
                   query: list[float], top_k: int,
                   provider: str, store: str) -> list[SearchResult]:
    if not buffer:
        return []
    import math
    def dot(a, b):
        n = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
        return sum(x*y for x, y in zip(a, b)) / n if n else 0.0

    scored = sorted(enumerate(buffer), key=lambda t: dot(t[1], query), reverse=True)
    results = []
    for rank, (idx, _) in enumerate(scored[:top_k]):
        if idx >= len(meta):
            continue
        m = meta[idx]
        results.append(SearchResult(
            rank=rank + 1, score=dot(buffer[idx], query),
            chunk_id=m["chunk_id"], source_doc_id=m["source_doc_id"],
            source=m["source"], content=m["content"],
            metadata=m.get("metadata", {}), provider=provider, store=store,
        ))
    return results
