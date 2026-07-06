from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="json")
class LlamaIndexJSONLoader(BaseLoader):
    """Load JSON files using LlamaIndex JSONReader — levels-of-depth aware extraction."""

    @property
    def supported_types(self) -> list[str]:
        return [".json"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.core.readers.json import JSONReader
        except ImportError:
            raise ImportError("pip install llama-index-core")

        nodes = JSONReader().load_data(input_file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="json",
            )
            for n in nodes
        ]
