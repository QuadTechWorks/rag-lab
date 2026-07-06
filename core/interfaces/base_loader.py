from __future__ import annotations
from abc import ABC, abstractmethod
from core.models.documents import LoadedDocument


class BaseLoader(ABC):

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def load(self, source: str) -> list[LoadedDocument]:
        """Load documents from a file path or URL."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Human-readable list of supported input types (e.g. ['.pdf', '.txt'])."""
        ...
