from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="llamaindex", name="semantic")
class LlamaIndexSemanticChunker(BaseChunker):
    """SemanticSplitterNodeParser — groups sentences by embedding similarity.

    Does NOT use chunk_size/overlap — finds natural topic boundaries using cosine
    similarity between sentence embeddings. Uses BAAI/bge-small-en-v1.5 (local, free).

    Requires: pip install llama-index-embeddings-huggingface sentence-transformers
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
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core")

        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except ImportError:
            raise ImportError(
                "pip install llama-index-embeddings-huggingface sentence-transformers"
            )

        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,
        )
        chunks = []
        for doc in docs:
            llama_doc = LlamaDoc(text=doc.content, metadata=doc.metadata, id_=doc.id)
            nodes = splitter.get_nodes_from_documents([llama_doc])
            for i, node in enumerate(nodes):
                chunks.append(DocumentChunk(
                    content=node.text,
                    metadata={**node.metadata, "chunk_method": "semantic_splitter"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="semantic",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
