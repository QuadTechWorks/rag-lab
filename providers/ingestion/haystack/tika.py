from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="tika")
class HaystackTikaLoader(BaseLoader):
    """Universal loader via Haystack TikaDocumentConverter — 100+ formats via Apache Tika.

    Requires:
      1. Java installed: brew install openjdk
      2. pip install tika haystack-ai
      Tika server auto-starts on first use (downloads tika-server.jar).
    """

    @property
    def supported_types(self) -> list[str]:
        return [".pdf", ".doc", ".docx", ".pptx", ".ppt", ".xlsx", ".xls",
                ".odt", ".ods", ".odp", ".rtf", ".eml", ".msg", ".html",
                ".epub", ".txt", ".xml", ".csv"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters.tika import TikaDocumentConverter
        except ImportError:
            raise ImportError("pip install 'haystack-ai' tika")

        result = TikaDocumentConverter().run(sources=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata=d.meta or {},
                source=source,
                provider="haystack",
                loader="tika",
            )
            for d in result.get("documents", [])
        ]
