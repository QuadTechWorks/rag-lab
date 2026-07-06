from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="csv")
class LlamaIndexCSVLoader(BaseLoader):
    """Load CSV files using LlamaIndex PandasCSVReader — one doc per row with rich metadata."""

    @property
    def supported_types(self) -> list[str]:
        return [".csv"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import PandasCSVReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file pandas")

        nodes = PandasCSVReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "row": i},
                source=source,
                provider="llamaindex",
                loader="csv",
            )
            for i, n in enumerate(nodes)
        ]
