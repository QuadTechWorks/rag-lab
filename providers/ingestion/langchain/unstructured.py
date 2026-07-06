"""Unstructured fallback loader — handles any file type it recognizes.

This is the last-resort loader in the auto-dispatch chain. Install extras:
  pip install "unstructured[pdf,docx,pptx,xlsx]"
On macOS also:  brew install libmagic poppler tesseract
"""
from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="unstructured")
class LangChainUnstructuredLoader(BaseLoader):
    """Universal loader via the unstructured library (PDF, DOC, PPT, EML, RTF, ODT, …)."""

    @property
    def supported_types(self) -> list[str]:
        return ["any (universal fallback)"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            raise ImportError(
                'pip install "unstructured[local-inference]"  '
                '# macOS: brew install libmagic poppler tesseract'
            )

        elements = partition(filename=source)
        text = "\n\n".join(str(e) for e in elements if str(e).strip())

        return [LoadedDocument(
            content=text,
            metadata={"source": source, "element_count": len(elements)},
            source=source,
            provider="langchain",
            loader="unstructured",
        )]
