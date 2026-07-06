from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="custom", name="sliding_window")
class SlidingWindowChunker(BaseChunker):
    """Pure character-level sliding window — no framework dependency.

    Slides a window of chunk_size characters across the text with chunk_overlap
    characters of overlap between consecutive chunks. Guarantees every chunk is
    exactly chunk_size chars except the last. Most predictable chunk sizes.
    """

    @property
    def supported_types(self) -> list[str]:
        return ["any"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        step = max(1, chunk_size - chunk_overlap)
        chunks = []
        for doc in docs:
            text = doc.content
            start = 0
            i = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(DocumentChunk(
                    content=text[start:end],
                    metadata={**doc.metadata, "chunk_method": "sliding_window",
                               "window_start": start, "window_end": end},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="custom",
                    chunker="sliding_window",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
                start += step
                i += 1
        return chunks
