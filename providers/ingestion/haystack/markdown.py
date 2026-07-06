from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="markdown")
class HaystackMarkdownLoader(BaseLoader):
    """Load Markdown files using Haystack TextFileToDocument with markdown metadata tag."""

    @property
    def supported_types(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import TextFileToDocument
        except ImportError:
            raise ImportError("pip install haystack-ai")

        result = TextFileToDocument().run(sources=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata={**(d.meta or {}), "format": "markdown"},
                source=source,
                provider="haystack",
                loader="markdown",
            )
            for d in result.get("documents", [])
        ]
