from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="llamaindex", name="sentence")
class LlamaIndexSentenceChunker(BaseChunker):
    """SentenceSplitter — sentence-aware chunking that avoids cutting mid-sentence.

    chunk_size is in TOKENS (~4 chars each). Respects sentence boundaries — will
    not split a sentence even if it exceeds chunk_size. Best for prose documents.
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
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core")

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = []
        for doc in docs:
            llama_doc = LlamaDoc(text=doc.content, metadata=doc.metadata, id_=doc.id)
            nodes = splitter.get_nodes_from_documents([llama_doc])
            for i, node in enumerate(nodes):
                chunks.append(DocumentChunk(
                    content=node.text,
                    metadata={**node.metadata, "chunk_method": "sentence_splitter"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="sentence",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
