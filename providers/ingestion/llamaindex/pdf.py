from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="pdf")
class LlamaIndexPDFLoader(BaseLoader):
    """Load PDFs using LlamaIndex PDFReader — page-per-doc via pypdf."""

    @property
    def supported_types(self) -> list[str]:
        return [".pdf"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import PDFReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file pypdf")

        nodes = PDFReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "page": i + 1},
                source=source,
                provider="llamaindex",
                loader="pdf",
            )
            for i, n in enumerate(nodes)
        ]
