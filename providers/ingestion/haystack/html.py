from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="html")
class HaystackHTMLLoader(BaseLoader):
    """Load HTML files using Haystack HTMLToDocument converter (BeautifulSoup-based)."""

    @property
    def supported_types(self) -> list[str]:
        return [".html", ".htm"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import HTMLToDocument
        except ImportError:
            raise ImportError("pip install 'haystack-ai>=2.3.0' beautifulsoup4")

        result = HTMLToDocument().run(sources=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata=d.meta or {},
                source=source,
                provider="haystack",
                loader="html",
            )
            for d in result.get("documents", [])
        ]
