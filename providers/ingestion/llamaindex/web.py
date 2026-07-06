from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="llamaindex", name="web")
class LlamaIndexWebLoader(BaseLoader):
    """Fetch and parse a web page using LlamaIndex BeautifulSoupWebReader."""

    @property
    def supported_types(self) -> list[str]:
        return ["url (http/https)"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from llama_index.readers.web import BeautifulSoupWebReader
        except ImportError:
            raise ImportError(
                "pip install llama-index-readers-web beautifulsoup4 html2text"
            )

        docs = BeautifulSoupWebReader().load_data(urls=[source])
        return [
            LoadedDocument(
                content=d.text,
                metadata={**d.metadata, "url": source},
                source=source,
                provider="llamaindex",
                loader="web",
            )
            for d in docs
        ]
