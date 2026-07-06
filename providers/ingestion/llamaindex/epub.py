from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="epub")
class LlamaIndexEpubLoader(BaseLoader):
    """Load EPUB files using LlamaIndex EpubReader — one doc per chapter."""

    @property
    def supported_types(self) -> list[str]:
        return [".epub"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import EpubReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file ebooklib html2text")

        nodes = EpubReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "chapter": i + 1},
                source=source,
                provider="llamaindex",
                loader="epub",
            )
            for i, n in enumerate(nodes)
        ]
