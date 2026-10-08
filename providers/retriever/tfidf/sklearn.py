from __future__ import annotations
from core.registry import RETRIEVERS
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult
from core.retrieval_utils import matches_filters


@RETRIEVERS.register(provider="tfidf", name="sklearn")
class TfidfSklearnRetriever(BaseRetriever):
    """TF-IDF cosine similarity via scikit-learn.

    Requires: pip install scikit-learn
    Config: ngram_max (default 1; 2 adds bigrams), stop_words (default None; "english")
    """

    def __init__(self, ngram_max: int = 1, stop_words: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._ngram_max = ngram_max
        self._stop_words = stop_words
        self._corpus: list[EmbeddedChunk] = []
        self._vectorizer = None
        self._matrix = None

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("pip install scikit-learn")
        self._corpus = list(corpus)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, self._ngram_max),
                                           stop_words=self._stop_words,
                                           sublinear_tf=True)
        try:
            self._matrix = self._vectorizer.fit_transform([c.content for c in self._corpus])
        except ValueError:  # empty vocabulary
            self._matrix = None

    def retrieve(self, query: str, top_k: int = 5,
                 filters: dict | None = None) -> list[RetrievalResult]:
        if self._matrix is None or not query.strip():
            return []
        # TfidfVectorizer L2-normalises rows, so dot product == cosine similarity.
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        ranked = sorted(
            (i for i in range(len(self._corpus))
             if scores[i] > 0 and matches_filters(self._corpus[i].metadata, filters)),
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
                provider="tfidf", retriever="sklearn",
            )
            for r, i in enumerate(ranked)
        ]

    @property
    def retriever_name(self) -> str:
        return f"TF-IDF (scikit-learn, ngram<= {self._ngram_max})"
