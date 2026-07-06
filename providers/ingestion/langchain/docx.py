from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="docx")
class LangChainDocxLoader(BaseLoader):
    """Load Word documents (.docx) using LangChain Docx2txtLoader."""

    @property
    def supported_types(self) -> list[str]:
        return [".docx"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            raise ImportError("pip install langchain-community docx2txt")

        docs = Docx2txtLoader(source).load()
        return [
            LoadedDocument(
                content=d.page_content,
                metadata=d.metadata,
                source=source,
                provider="langchain",
                loader="docx",
            )
            for d in docs
        ]
