from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="llamaindex", name="hierarchical")
class LlamaIndexHierarchicalChunker(BaseChunker):
    """HierarchicalNodeParser — creates three-level parent/child chunk hierarchy.

    Produces chunks at 3 sizes: 2048 (large) → 512 (medium) → chunk_size (leaf).
    Returns LEAF nodes (smallest chunks) for retrieval, with parent context
    preserved in metadata for merged retrieval in Phase 5.
    Best for long documents where context window requires small retrieval units
    but accurate answer generation needs broader context.
    """

    @property
    def supported_types(self) -> list[str]:
        return ["any"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 128,
        chunk_overlap: int = 20,
    ) -> list[DocumentChunk]:
        try:
            from llama_index.core.node_parser import (
                HierarchicalNodeParser,
                get_leaf_nodes,
            )
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core")

        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, chunk_size],
        )
        chunks = []
        for doc in docs:
            llama_doc = LlamaDoc(text=doc.content, metadata=doc.metadata, id_=doc.id)
            all_nodes = parser.get_nodes_from_documents([llama_doc])
            leaf_nodes = get_leaf_nodes(all_nodes)
            for i, node in enumerate(leaf_nodes):
                chunks.append(DocumentChunk(
                    content=node.text,
                    metadata={
                        **node.metadata,
                        "chunk_method": "hierarchical",
                        "parent_id": node.parent_node.node_id if node.parent_node else None,
                    },
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="hierarchical",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
