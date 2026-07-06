from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="pptx")
class HaystackPPTXLoader(BaseLoader):
    """Load PPTX files using Haystack PPTXToDocument converter (slide-per-doc)."""

    @property
    def supported_types(self) -> list[str]:
        return [".pptx"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from haystack.components.converters import PPTXToDocument
        except ImportError:
            raise ImportError("pip install 'haystack-ai>=2.5.0' python-pptx")

        result = PPTXToDocument().run(sources=[source])
        return [
            LoadedDocument(
                content=d.content or "",
                metadata={**(d.meta or {}), "slide": i + 1},
                source=source,
                provider="haystack",
                loader="pptx",
            )
            for i, d in enumerate(result.get("documents", []))
        ]
