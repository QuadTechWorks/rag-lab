from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="web")
class LangChainWebLoader(BaseLoader):
    """Load a web page using LangChain WebBaseLoader (source = URL)."""

    @property
    def supported_types(self) -> list[str]:
        return ["url (http/https)"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from langchain_community.document_loaders import WebBaseLoader
        except ImportError:
            raise ImportError("pip install langchain-community beautifulsoup4")

        docs = WebBaseLoader(source).load()
        return [
            LoadedDocument(
                content=d.page_content,
                metadata=d.metadata,
                source=source,
                provider="langchain",
                loader="web",
            )
            for d in docs
        ]
