from __future__ import annotations
import re
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk

# Sentence boundary: period/!/ followed by whitespace and uppercase, or end of string
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'])')


@CHUNKERS.register(provider="custom", name="sentence_boundary")
class SentenceBoundaryChunker(BaseChunker):
    """Sentence boundary chunker — regex sentence detection, no external dependencies.

    Detects sentence boundaries using punctuation patterns. Accumulates sentences
    until chunk_size chars is reached. chunk_overlap = number of sentences carried
    over to the next chunk. No NLTK or spaCy required.
    """

    @property
    def supported_types(self) -> list[str]:
        return ["any"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 2,
    ) -> list[DocumentChunk]:
        chunks = []
        for doc in docs:
            sentences = _SENT_RE.split(doc.content)
            sentences = [s.strip() for s in sentences if s.strip()]

            current: list[str] = []
            current_len = 0
            chunk_idx = 0

            for sent in sentences:
                if current_len + len(sent) > chunk_size and current:
                    chunk_text = " ".join(current)
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata={**doc.metadata, "chunk_method": "sentence_boundary",
                                   "sentence_count": len(current)},
                        source_doc_id=doc.id,
                        source=doc.source,
                        chunk_index=chunk_idx,
                        provider="custom",
                        chunker="sentence_boundary",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    ))
                    chunk_idx += 1
                    overlap_sents = current[-chunk_overlap:] if chunk_overlap > 0 else []
                    current = list(overlap_sents)
                    current_len = sum(len(s) for s in current)

                current.append(sent)
                current_len += len(sent)

            if current:
                chunks.append(DocumentChunk(
                    content=" ".join(current),
                    metadata={**doc.metadata, "chunk_method": "sentence_boundary",
                               "sentence_count": len(current)},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=chunk_idx,
                    provider="custom",
                    chunker="sentence_boundary",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
