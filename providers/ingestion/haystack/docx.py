from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="docx")
class HaystackDocxLoader(BaseLoader):
    """Load DOCX files using Haystack DOCXToDocument converter."""

    @property
    def supported_types(self) -> list[str]:
        return [".docx"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import DOCXToDocument
        except ImportError:
            raise ImportError("pip install 'haystack-ai>=2.3.0' python-docx")

        result = DOCXToDocument().run(sources=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata=d.meta or {},
                source=source,
                provider="haystack",
                loader="docx",
            )
            for d in result.get("documents", [])
        ]
