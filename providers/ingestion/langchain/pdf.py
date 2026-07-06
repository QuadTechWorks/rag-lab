from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="pdf")
class LangChainPDFLoader(BaseLoader):
    """Load PDF files using LangChain PyPDFLoader (one doc per page)."""

    @property
    def supported_types(self) -> list[str]:
        return [".pdf"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            raise ImportError("pip install langchain-community pypdf")

        docs = PyPDFLoader(source).load()
        return [
            LoadedDocument(
                content=d.page_content,
                metadata={**d.metadata, "page": i + 1},
                source=source,
                provider="langchain",
                loader="pdf",
            )
            for i, d in enumerate(docs)
        ]
