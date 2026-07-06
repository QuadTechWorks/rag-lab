from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="unstructured")
class LlamaIndexUnstructuredLoader(BaseLoader):
    """Universal loader via LlamaIndex UnstructuredReader — wraps unstructured.io (30+ formats).

    Requires: brew install libmagic poppler tesseract (macOS)
              pip install "unstructured[pdf,docx,pptx]"
    """

    @property
    def supported_types(self) -> list[str]:
        return [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx",
                ".html", ".eml", ".msg", ".odt", ".rtf", ".epub"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.file import UnstructuredReader
        except ImportError:
            raise ImportError(
                "pip install llama-index-readers-file "
                "\"unstructured[pdf,docx,pptx]\""
            )

        nodes = UnstructuredReader().load_data(file=Path(source))
        return [
            LoadedDocument(
                content=n.text,
                metadata=n.metadata,
                source=source,
                provider="llamaindex",
                loader="unstructured",
            )
            for n in nodes
        ]
