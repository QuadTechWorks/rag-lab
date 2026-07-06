from __future__ import annotations
from core.registry import CHUNKERS
from core.interfaces.base_chunker import BaseChunker
from core.models.documents import LoadedDocument
from core.models.chunks import DocumentChunk

# Extension → tree-sitter language name
_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php",
}


@CHUNKERS.register(provider="llamaindex", name="code")
class LlamaIndexCodeChunker(BaseChunker):
    """CodeSplitter — syntax-aware splitting using tree-sitter parse trees.

    chunk_size here is LINES (not characters). Splits along function/class boundaries.
    Language is auto-detected from file extension in metadata.

    Requires: pip install tree-sitter tree-sitter-languages
    """

    @property
    def supported_types(self) -> list[str]:
        return [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".cs"]

    def chunk(
        self,
        docs: list[LoadedDocument],
        chunk_size: int = 40,
        chunk_overlap: int = 10,
    ) -> list[DocumentChunk]:
        try:
            from llama_index.core.node_parser import CodeSplitter
            from llama_index.core import Document as LlamaDoc
        except ImportError:
            raise ImportError("pip install llama-index-core")

        chunks = []
        for doc in docs:
            ext = doc.metadata.get("file_path", doc.source)
            ext = "." + ext.rsplit(".", 1)[-1].lower() if "." in ext else ""
            language = _EXT_LANG.get(ext, "python")

            try:
                splitter = CodeSplitter(
                    language=language,
                    chunk_lines=chunk_size,
                    chunk_lines_overlap=chunk_overlap,
                )
                llama_doc = LlamaDoc(text=doc.content, metadata=doc.metadata, id_=doc.id)
                nodes = splitter.get_nodes_from_documents([llama_doc])
            except Exception as exc:
                raise ImportError(
                    f"CodeSplitter requires tree-sitter grammars: "
                    f"pip install tree-sitter tree-sitter-languages\n({exc})"
                )

            for i, node in enumerate(nodes):
                chunks.append(DocumentChunk(
                    content=node.text,
                    metadata={
                        **node.metadata,
                        "language": language,
                        "chunk_method": "code_splitter",
                    },
                    source_doc_id=doc.id,
                    source=doc.source,
                    chunk_index=i,
                    provider="llamaindex",
                    chunker="code",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ))
        return chunks
