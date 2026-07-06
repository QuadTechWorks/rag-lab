from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="pdf")
class HaystackPDFLoader(BaseLoader):
    """Load PDF files using Haystack PyPDFToDocument converter."""

    @property
    def supported_types(self) -> list[str]:
        return [".pdf"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import PyPDFToDocument
        except ImportError:
            raise ImportError("pip install haystack-ai pypdf")

        converter = PyPDFToDocument()
        result = converter.run(sources=[source])
        docs = result.get("documents", [])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata={**(d.meta or {}), "page": i + 1},
                source=source,
                provider="haystack",
                loader="pdf",
            )
            for i, d in enumerate(docs)
        ]
