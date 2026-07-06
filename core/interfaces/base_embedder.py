from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk


class BaseEmbedder(ABC):

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def embed(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """Embed a list of DocumentChunks; returns one EmbeddedChunk per input."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier used by this embedder."""
        ...

    @property
    @abstractmethod
    def vector_dim(self) -> int:
        """Dimensionality of the output vectors."""
        ...
