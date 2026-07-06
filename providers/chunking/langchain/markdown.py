from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="langchain", name="markdown")
class LangChainMarkdownChunker(BaseChunker):
    """MarkdownHeaderTextSplitter — splits at heading boundaries (H1/H2/H3).

    chunk_size/overlap are NOT used — splits strictly at header boundaries.
    Best for Markdown docs where headings define logical sections.
    Each chunk includes the section header in its metadata.
    """

    @property
    def supported_types(self) -> list[str]:
        return [".md", ".markdown"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
        except ImportError:
            raise ImportError("pip install langchain")

        headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        chunks = []
        for doc in docs:
            sections = splitter.split_text(doc.content)
            for i, section in enumerate(sections):
                chunks.append(DocumentChunk(
                    content=section.page_content,
                    metadata={
                        **doc.metadata,
                        **section.metadata,
                        "chunk_method": "markdown_header",
                    },
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="langchain",
                    chunker="markdown",
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                ))
        return chunks
