from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="llamaindex", name="token")
class LlamaIndexTokenChunker(BaseChunker):
    """TokenTextSplitter — strict token-count splitting using tiktoken.

    chunk_size is NUMBER OF TOKENS. Hard cut at token boundaries — does not
    respect word/sentence boundaries. Most accurate for token-budget-constrained pipelines.
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
            from llama_index.core.node_parser import TokenTextSplitter
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core tiktoken")

        splitter = TokenTextSplitter(
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
                    metadata={**node.metadata, "chunk_method": "token_text_splitter"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="token",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
