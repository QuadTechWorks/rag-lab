from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="html")
class LangChainHTMLLoader(BaseLoader):
    """Load local HTML files using BeautifulSoup — strips tags, keeps text."""

    @property
    def supported_types(self) -> list[str]:
        return [".html", ".htm"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install beautifulsoup4")

        with open(source, encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style", "head", "nav", "footer"]):
            tag.decompose()

        content = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else ""

        return [LoadedDocument(
            content=content,
            metadata={"title": title, "source": source},
            source=source,
            provider="langchain",
            loader="html",
        )]
