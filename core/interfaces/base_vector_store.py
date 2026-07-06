from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult


class BaseVectorStore(ABC):

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        """Upsert EmbeddedChunks into the store. Returns {indexed: N, total: N}."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar chunks."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete by chunk_id. Returns count deleted."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the store."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return store statistics: count, vector_dim, store_name, etc."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if the store is connected and ready."""
        ...

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Human-readable name for this store instance."""
        ...
