from __future__ import annotations
from core.registry import RETRIEVERS
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult
from core.models.search import SearchResult
from core.models.chunks import DocumentChunk


def embed_query(embedder, query: str) -> list[float]:
    """Embed a query string through the standard chunk-based embedder interface."""
    dummy = DocumentChunk(content=query, source_doc_id="query", source="query",
                          chunk_index=0, provider="_query", chunker="_query",
                          chunk_size=len(query), chunk_overlap=0)
    return embedder.embed([dummy])[0].vector


def to_result(r: SearchResult, rank: int, retriever: str, provider: str = "vector"
              ) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=r.score, chunk_id=r.chunk_id, source_doc_id=r.source_doc_id,
        source=r.source, content=r.content, metadata=r.metadata,
        provider=provider, retriever=retriever,
    )


@RETRIEVERS.register(provider="vector", name="dense")
class DenseRetriever(BaseRetriever):
    """Dense semantic retrieval — embeds the query and searches the active vector store.

    Kwargs (injected by the API): vector_store (BaseVectorStore), embedder (BaseEmbedder)
    """

    def __init__(self, vector_store=None, embedder=None, **kwargs):
        super().__init__(**kwargs)
        self._store = vector_store
        self._embedder = embedder

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        pass  # the vector store is populated in Phase 4

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self._store is None or self._embedder is None:
            raise RuntimeError("vector/dense needs a connected vector store and an embedder")
        hits = self._store.search(embed_query(self._embedder, query), top_k=top_k,
                                  filters=filters)
        return [to_result(h, i + 1, "dense") for i, h in enumerate(hits)]

    @property
    def retriever_name(self) -> str:
        return "Dense (vector store similarity)"
