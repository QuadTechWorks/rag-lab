from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="markdown")
class LlamaIndexMarkdownLoader(BaseLoader):
    """Load Markdown files using LlamaIndex MarkdownReader — heading-aware section splitting."""

    @property
    def supported_types(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import MarkdownReader
        except ImportError:
            raise ImportError("pip install llama-index-readers-file")

        nodes = MarkdownReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="markdown",
            )
            for n in nodes
        ]
