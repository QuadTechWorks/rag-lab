from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="langchain", name="character")
class LangChainCharacterChunker(BaseChunker):
    """CharacterTextSplitter — hard split on a single separator (default: \\n\\n).

    Simple and fast. Splits strictly on the separator without trying alternatives.
    Use when documents have clear paragraph boundaries.
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
        try:
            from langchain_text_splitters import CharacterTextSplitter
        except ImportError:
            raise ImportError("pip install langchain")

        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = []
        for doc in docs:
            texts = splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(DocumentChunk(
                    content=text,
                    metadata={**doc.metadata, "chunk_method": "character"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="langchain",
                    chunker="character",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
