from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="txt")
class LangChainTextLoader(BaseLoader):
    """Load plain text files using LangChain TextLoader."""

    @property
    def supported_types(self) -> list[str]:
        return [".txt", ".md", ".rst", ".log"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from langchain_community.document_loaders import TextLoader
        except ImportError:
            raise ImportError("pip install langchain-community")

        docs = TextLoader(source, encoding="utf-8").load()
        return [
            LoadedDocument(
                content=d.page_content,
                metadata=d.metadata,
                source=source,
                provider="langchain",
                loader="txt",
            )
            for d in docs
        ]
