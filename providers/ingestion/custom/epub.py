from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="epub")
class CustomEPUBLoader(BaseLoader):
    """Load EPUB ebooks — one document per chapter."""

    @property
    def supported_types(self) -> list[str]:
        return [".epub"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install ebooklib beautifulsoup4")

        book = epub.read_epub(source)
        docs = []
        for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                docs.append(LoadedDocument(
                    content=text,
                    metadata={"chapter": i + 1, "item_name": item.get_name(), "source": source},
                    source=source,
                    provider="custom",
                    loader="epub",
                ))
        return docs
