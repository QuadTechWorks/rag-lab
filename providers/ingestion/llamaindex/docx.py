from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="docx")
class LlamaIndexDocxLoader(BaseLoader):
    """Load DOCX files using LlamaIndex DocxReader."""

    @property
    def supported_types(self) -> list[str]:
        return [".docx"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import DocxReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file python-docx")

        nodes = DocxReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="docx",
            )
            for n in nodes
        ]
