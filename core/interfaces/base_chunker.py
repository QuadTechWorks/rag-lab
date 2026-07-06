from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


class BaseChunker(ABC):

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        """Split a list of LoadedDocuments into DocumentChunks."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Human-readable description of what this chunker supports."""
        ...
