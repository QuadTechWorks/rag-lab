from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="ipynb")
class LlamaIndexNotebookLoader(BaseLoader):
    """Load Jupyter notebooks using LlamaIndex IPYNBReader — one doc per cell."""

    @property
    def supported_types(self) -> list[str]:
        return [".ipynb"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import IPYNBReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file nbformat")

        nodes = IPYNBReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "cell": i + 1},
                source=source,
                provider="llamaindex",
                loader="ipynb",
            )
            for i, n in enumerate(nodes)
        ]
