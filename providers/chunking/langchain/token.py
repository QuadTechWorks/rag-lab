from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="langchain", name="token")
class LangChainTokenChunker(BaseChunker):
    """TokenTextSplitter — splits by token count using tiktoken (cl100k_base encoding).

    chunk_size here means NUMBER OF TOKENS, not characters (~4 chars per token).
    Best when the downstream embedding/LLM has a strict token window.
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
            from langchain_text_splitters import TokenTextSplitter
        except ImportError:
            raise ImportError("pip install langchain tiktoken")

        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )
        chunks = []
        for doc in docs:
            texts = splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(DocumentChunk(
                    content=text,
                    metadata={**doc.metadata, "chunk_method": "token", "encoding": "cl100k_base"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="langchain",
                    chunker="token",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
