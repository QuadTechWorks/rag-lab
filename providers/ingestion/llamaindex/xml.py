from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="xml")
class LlamaIndexXMLLoader(BaseLoader):
    """Load XML files using LlamaIndex XMLReader — element-per-doc extraction."""

    @property
    def supported_types(self) -> list[str]:
        return [".xml"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import XMLReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file lxml")

        nodes = XMLReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="xml",
            )
            for n in nodes
        ]
