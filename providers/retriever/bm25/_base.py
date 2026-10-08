from __future__ import annotations
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult
from core.retrieval_utils import tokenize, matches_filters


class BM25Base(BaseRetriever):
    """Shared BM25 logic; subclasses set `variant` and `_bm25_cls_name`.

    Requires: pip install rank_bm25
    Config: k1 (default 1.5), b (default 0.75)
    Chunks sharing no token with the query are never returned (BM25Plus would
    otherwise give them a positive score via its delta term).
    """
    variant: str = ""
    _bm25_cls_name: str = ""

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self._k1, self._b = k1, b
        self._corpus: list[EmbeddedChunk] = []
        self._bm25 = None
        self._token_sets: list[set[str]] = []

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        try:
            import rank_bm25
        except ImportError:
            raise ImportError("pip install rank_bm25")
        self._corpus = list(corpus)
        tokens = [tokenize(c.content) for c in self._corpus]
        self._token_sets = [set(t) for t in tokens]
        self._bm25 = (getattr(rank_bm25, self._bm25_cls_name)(tokens, k1=self._k1, b=self._b)
                      if any(tokens) else None)

    def retrieve(self, query: str, top_k: int = 5,
                 filters: dict | None = None) -> list[RetrievalResult]:
        q = tokenize(query)
        if self._bm25 is None or not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(
            (i for i in range(len(self._corpus))
             if scores[i] > 0 and self._token_sets[i].intersection(q)
             and matches_filters(self._corpus[i].metadata, filters)),
            key=lambda i: scores[i], reverse=True,
        )[:top_k]
        return [
            RetrievalResult(
                rank=r + 1, score=float(scores[i]),
                chunk_id=self._corpus[i].chunk_id,
                source_doc_id=self._corpus[i].source_doc_id,
                source=self._corpus[i].source,
                content=self._corpus[i].content,
                metadata=dict(self._corpus[i].metadata),
                provider="bm25", retriever=self.variant,
            )
            for r, i in enumerate(ranked)
        ]

    @property
    def retriever_name(self) -> str:
        return f"BM25 {self.variant} (k1={self._k1}, b={self._b})"
