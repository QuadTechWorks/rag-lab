from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="haystack", name="word")
class HaystackWordChunker(BaseChunker):
    """Haystack DocumentSplitter — splits by word count.

    chunk_size = number of words per chunk. chunk_overlap = words of overlap.
    Fast and predictable. Does not respect sentence boundaries.
    """

    @property
    def supported_types(self) -> list[str]:
        return ["any"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 200,
        chunk_overlap: int = 20,
    ) -> list[DocumentChunk]:
        try:
            from haystack.components.preprocessors import DocumentSplitter
            from haystack.dataclasses import Document as HaystackDoc
        except ImportError:
            raise ImportError("pip install haystack-ai")

        splitter = DocumentSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
        )
        chunks = []
        for doc in docs:
            h_doc = HaystackDoc(content=doc.content, meta=doc.metadata)
            result = splitter.run(documents=[h_doc])
            for i, chunk in enumerate(result.get("documents", [])):
                chunks.append(DocumentChunk(
                    content=chunk.content or "",
                    metadata={**(chunk.meta or {}), "chunk_method": "word_splitter"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="haystack",
                    chunker="word",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
