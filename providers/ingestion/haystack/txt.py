from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="txt")
class HaystackTextLoader(BaseLoader):
    """Load plain text/Markdown files using Haystack TextFileToDocument."""

    @property
    def supported_types(self) -> list[str]:
        return [".txt", ".md"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import TextFileToDocument
            from haystack.dataclasses import ByteStream
        except ImportError:
            raise ImportError("pip install haystack-ai")

        converter = TextFileToDocument()
        sources = [source]
        result = converter.run(sources=sources)
        docs = result.get("documents", [])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata=d.meta or {},
                source=source,
                provider="haystack",
                loader="txt",
            )
            for d in docs
        ]
