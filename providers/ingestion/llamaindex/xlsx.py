from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="xlsx")
class LlamaIndexExcelLoader(BaseLoader):
    """Load Excel files using LlamaIndex PandasExcelReader — one doc per sheet."""

    @property
    def supported_types(self) -> list[str]:
        return [".xlsx", ".xls"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import PandasExcelReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file pandas openpyxl")

        nodes = PandasExcelReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="xlsx",
            )
            for n in nodes
        ]
