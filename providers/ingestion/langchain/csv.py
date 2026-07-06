from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="csv")
class LangChainCSVLoader(BaseLoader):
    """Load CSV files using LangChain CSVLoader (one doc per row)."""

    @property
    def supported_types(self) -> list[str]:
        return [".csv"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from langchain_community.document_loaders import CSVLoader
        except ImportError:
            raise ImportError("pip install langchain-community")

        docs = CSVLoader(file_path=source).load()
        return [
            LoadedDocument(
                content=d.page_content,
                metadata={**d.metadata, "row": i + 1},
                source=source,
                provider="langchain",
                loader="csv",
            )
            for i, d in enumerate(docs)
        ]
