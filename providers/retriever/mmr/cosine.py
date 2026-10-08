from __future__ import annotations
import numpy as np
from core.registry import RETRIEVERS
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult
from providers.retriever.vector.dense import embed_query


@RETRIEVERS.register(provider="mmr", name="cosine")
class MMRCosineRetriever(BaseRetriever):
    """Maximal Marginal Relevance — relevant yet diverse results.

    Fetches `fetch_k` dense candidates, then greedily picks the chunk maximising
        lambda_mult * sim(query, d) - (1 - lambda_mult) * max sim(d, selected)
    Config: lambda_mult (default 0.5; 1.0 = pure relevance), fetch_k (default 20)
    Kwargs (injected by the API): vector_store, embedder
    Candidate vectors come from the indexed corpus; any missing are re-embedded.
    """

    def __init__(self, vector_store=None, embedder=None, lambda_mult: float = 0.5,
                 fetch_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._store = vector_store
        self._embedder = embedder
        self._lambda = lambda_mult
        self._fetch_k = fetch_k
        self._vectors: dict[str, list[float]] = {}

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        self._vectors = {c.chunk_id: c.vector for c in corpus}

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self._store is None or self._embedder is None:
            raise RuntimeError("mmr/cosine needs a connected vector store and an embedder")
        qvec = np.asarray(embed_query(self._embedder, query), dtype=float)
        hits = self._store.search(qvec.tolist(), top_k=max(self._fetch_k, top_k),
                                  filters=filters)
        if not hits:
            return []

        cand = np.asarray([self._vector_for(h) for h in hits], dtype=float)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True).clip(min=1e-12)
        qn = qvec / max(np.linalg.norm(qvec), 1e-12)
        rel = cand @ qn
        pair = cand @ cand.T

        selected: list[int] = []
        remaining = list(range(len(hits)))
        while remaining and len(selected) < top_k:
            if selected:
                redundancy = pair[np.ix_(remaining, selected)].max(axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            mmr = self._lambda * rel[remaining] - (1 - self._lambda) * redundancy
            pick = remaining[int(np.argmax(mmr))]
            selected.append(pick)
            remaining.remove(pick)

        return [
            RetrievalResult(
                rank=r + 1, score=float(rel[i]), chunk_id=hits[i].chunk_id,
                source_doc_id=hits[i].source_doc_id, source=hits[i].source,
                content=hits[i].content, metadata=hits[i].metadata,
                provider="mmr", retriever="cosine",
            )
            for r, i in enumerate(selected)
        ]

    def _vector_for(self, hit) -> list[float]:
        vec = self._vectors.get(hit.chunk_id)
        return vec if vec is not None else embed_query(self._embedder, hit.content)

    @property
    def retriever_name(self) -> str:
        return f"MMR (lambda={self._lambda}, fetch_k={self._fetch_k})"
