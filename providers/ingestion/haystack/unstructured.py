from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="unstructured")
class HaystackUnstructuredLoader(BaseLoader):
    """Universal loader via Haystack UnstructuredFileConverter — 30+ formats.

    Requires:
      brew install libmagic poppler tesseract (macOS)
      pip install "haystack-ai" "unstructured[pdf,docx,pptx]"
    """

    @property
    def supported_types(self) -> list[str]:
        return [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx",
                ".html", ".eml", ".msg", ".odt", ".rtf", ".epub"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters.unstructured import UnstructuredFileConverter
        except ImportError:
            raise ImportError(
                "pip install 'haystack-ai' "
                "\"unstructured[pdf,docx,pptx]\""
            )

        result = UnstructuredFileConverter().run(paths=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata=d.meta or {},
                source=source,
                provider="haystack",
                loader="unstructured",
            )
            for d in result.get("documents", [])
        ]
