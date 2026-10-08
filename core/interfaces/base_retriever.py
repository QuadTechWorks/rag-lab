from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult


class BaseRetriever(ABC):
    """A retriever turns a text query into ranked chunks.

    Lifecycle: construct with config -> index(corpus) -> retrieve(query).
    Dense retrievers receive `vector_store` and `embedder` via kwargs and
    ignore the corpus; sparse retrievers build their own index from it.
    """

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def index(self, corpus: list[EmbeddedChunk]) -> None:
        """Prepare the retriever over the given chunks (no-op for store-backed retrievers)."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Return up to top_k chunks ranked by relevance to the query."""
        ...

    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """Human-readable name for this retriever."""
        ...
