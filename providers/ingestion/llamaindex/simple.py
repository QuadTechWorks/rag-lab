from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="simple")
class LlamaIndexSimpleLoader(BaseLoader):
    """Load a single file using LlamaIndex SimpleDirectoryReader.

    Handles PDF, TXT, DOCX, HTML, PPTX automatically based on extension.
    """

    @property
    def supported_types(self) -> list[str]:
        return [".pdf", ".txt", ".md", ".docx", ".html", ".pptx"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError:
            raise ImportError("pip install llama-index-core llama-index-readers-file")

        import os
        from pathlib import Path as P

        path = P(source)
        reader = SimpleDirectoryReader(
            input_files=[str(path)],
            filename_as_id=True,
        )
        nodes = reader.load_data()
        return [
            LoadedDocument(
                content=n.text,
                metadata={**n.metadata, "node_id": n.id_},
                source=source,
                provider="llamaindex",
                loader="simple",
            )
            for n in nodes
        ]
