from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk


@CHUNKERS.register(provider="llamaindex", name="markdown")
class LlamaIndexMarkdownChunker(BaseChunker):
    """MarkdownNodeParser — splits Markdown by heading hierarchy into section nodes.

    chunk_size/overlap are NOT used — splits at heading boundaries.
    Preserves heading context in node relationships (parent/child).
    Best for structured Markdown docs (README, wikis, API docs).
    """

    @property
    def supported_types(self) -> list[str]:
        return [".md", ".markdown"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        try:
            from llama_index.core.node_parser import MarkdownNodeParser
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core")

        parser = MarkdownNodeParser()
        chunks = []
        for doc in docs:
            llama_doc = LlamaDoc(text=doc.content, metadata=doc.metadata, id_=doc.id)
            nodes = parser.get_nodes_from_documents([llama_doc])
            for i, node in enumerate(nodes):
                chunks.append(DocumentChunk(
                    content=node.text,
                    metadata={**node.metadata, "chunk_method": "markdown_node_parser"},
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="markdown",
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                ))
        return chunks
