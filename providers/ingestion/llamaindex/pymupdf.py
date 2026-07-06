from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="pymupdf")
class LlamaIndexPyMuPDFLoader(BaseLoader):
    """Load PDFs using LlamaIndex PyMuPDFReader — superior table and image text extraction."""

    @property
    def supported_types(self) -> list[str]:
        return [".pdf"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import PyMuPDFReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file pymupdf")

        nodes = PyMuPDFReader().load(file_path=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "page": i + 1},
                source=source,
                provider="llamaindex",
                loader="pymupdf",
            )
            for i, n in enumerate(nodes)
        ]
