from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="rtf")
class CustomRTFLoader(BaseLoader):
    """Load Rich Text Format (.rtf) files using striprtf."""

    @property
    def supported_types(self) -> list[str]:
        return [".rtf"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            raise ImportError("pip install striprtf")

        with open(source, encoding="utf-8", errors="replace") as f:
            rtf_content = f.read()

        text = rtf_to_text(rtf_content)
        return [LoadedDocument(
            content=text,
            metadata={"source": source},
            source=source,
            provider="custom",
            loader="rtf",
        )]
