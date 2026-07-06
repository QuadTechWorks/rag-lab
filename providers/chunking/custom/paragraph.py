from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="custom", name="paragraph")
class ParagraphChunker(BaseChunker):
    """Paragraph-aware chunker — splits on double-newlines, merges short paragraphs.

    Splits the text at \\n\\n boundaries. Merges consecutive paragraphs until
    chunk_size chars is reached, then starts a new chunk.
    chunk_overlap is in PARAGRAPHS (not characters) — the last N paragraphs of
    the previous chunk are prepended to the next one.
    """

    @property
    def supported_types(self) -> list[str]:
        return ["any"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 1000,
        chunk_overlap: int = 1,
    ) -> list[DocumentChunk]:
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
            current: list[str] = []
            current_len = 0
            chunk_idx = 0
            prev_tail: list[str] = []

            for para in paragraphs:
                if current_len + len(para) > chunk_size and current:
                    chunk_text = "\n\n".join(current)
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata={**doc.metadata, "chunk_method": "paragraph",
                                   "paragraphs": len(current)},
                        source_doc_id=doc.id,
                        source=doc.source,
                        chunk_index=chunk_idx,
                        provider="custom",
                        chunker="paragraph",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    ))
                    chunk_idx += 1
                    prev_tail = current[-chunk_overlap:] if chunk_overlap > 0 else []
                    current = list(prev_tail)
                    current_len = sum(len(p) for p in current)

                current.append(para)
                current_len += len(para)

            if current:
                chunks.append(DocumentChunk(
                    content="\n\n".join(current),
                    metadata={**doc.metadata, "chunk_method": "paragraph",
                               "paragraphs": len(current)},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=chunk_idx,
                    provider="custom",
                    chunker="paragraph",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
