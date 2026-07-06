from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="langchain", name="html")
class LangChainHTMLChunker(BaseChunker):
    """HTMLHeaderTextSplitter — splits HTML at heading tag boundaries (h1/h2/h3).

    chunk_size/overlap are NOT used — splits strictly at header tags.
    Best for HTML docs where headings define logical sections.
    Each chunk includes the header hierarchy in its metadata.
    """

    @property
    def supported_types(self) -> list[str]:
        return [".html", ".htm"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        try:
            from langchain_text_splitters import HTMLHeaderTextSplitter
        except ImportError:
            raise ImportError("pip install langchain beautifulsoup4")

        headers = [("h1", "h1"), ("h2", "h2"), ("h3", "h3"), ("h4", "h4")]
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)

        chunks = []
        for doc in docs:
            sections = splitter.split_text(doc.content)
            for i, section in enumerate(sections):
                chunks.append(DocumentChunk(
                    content=section.page_content,
                    metadata={
                        **doc.metadata,
                        **section.metadata,
                        "chunk_method": "html_header",
                    },
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="langchain",
                    chunker="html",
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                ))
        return chunks
