from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="txt")
class LlamaIndexFlatLoader(BaseLoader):
    """Load plain text files using LlamaIndex FlatReader — whole file as single doc."""

    @property
    def supported_types(self) -> list[str]:
        return [".txt", ".rst", ".text"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import FlatReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file")

        nodes = FlatReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="txt",
            )
            for n in nodes
        ]
