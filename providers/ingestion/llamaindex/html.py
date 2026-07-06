from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="html")
class LlamaIndexHTMLLoader(BaseLoader):
    """Load local HTML files using LlamaIndex HTMLTagReader — tag-aware text extraction."""

    @property
    def supported_types(self) -> list[str]:
        return [".html", ".htm"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import HTMLTagReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file beautifulsoup4")

        nodes = HTMLTagReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="html",
            )
            for n in nodes
        ]
