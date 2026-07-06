from __future__ import annotations
import os
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_DEFAULT_PATH = "./ragx_chroma"


@VECTOR_STORES.register(provider="chroma", name="local")
class ChromaLocalStore(BaseVectorStore):
    """ChromaDB persistent local store — no server, data in ./ragx_chroma.

    Requires: pip install chromadb
    Config: persist_dir (default: ./ragx_chroma)
    """

    def __init__(self, persist_dir: str = _DEFAULT_PATH, **kwargs):
        super().__init__(**kwargs)
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_ready(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")
        self._client = chromadb.PersistentClient(path=self._persist_dir)

    def _get_or_create_collection(self, dim: int):
        self._ensure_ready()
        name = f"ragx_{dim}"
        try:
            self._collection = self._client.get_collection(name)
        except Exception:
            self._collection = self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        col = self._get_or_create_collection(embedded_chunks[0].vector_dim)
        col.upsert(
            ids=[ec.chunk_id for ec in embedded_chunks],
            embeddings=[ec.vector for ec in embedded_chunks],
            documents=[ec.content for ec in embedded_chunks],
            metadatas=[{
                "source_doc_id": ec.source_doc_id,
                "source": ec.source,
                "provider": ec.provider,
                "embedder": ec.embedder,
                "model": ec.model,
                **{k: str(v) for k, v in ec.metadata.items()},
            } for ec in embedded_chunks],
        )
        return {"indexed": len(embedded_chunks), "total": col.count()}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        if self._collection is None:
            dim = len(query_vector)
            self._get_or_create_collection(dim)
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self._collection.count())),
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        )):
            output.append(SearchResult(
                rank=rank + 1,
                score=float(1 - dist),  # cosine distance → similarity
                chunk_id=doc_id,
                source_doc_id=meta.get("source_doc_id", ""),
                source=meta.get("source", ""),
                content=doc,
                metadata={k: v for k, v in meta.items()
                          if k not in ("source_doc_id", "source", "provider", "embedder", "model")},
                provider="chroma",
                store="local",
            ))
        return output

    def delete(self, ids: list[str]) -> int:
        if self._collection is None:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        if self._client is None:
            return
        for col in self._client.list_collections():
            self._client.delete_collection(col.name)
        self._collection = None

    def stats(self) -> dict:
        if self._collection is None:
            return {"count": 0, "store": "chroma/local", "persist_dir": self._persist_dir}
        return {
            "count": self._collection.count(),
            "store": "chroma/local",
            "persist_dir": self._persist_dir,
            "collection": self._collection.name,
        }

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "ChromaDB (local persistent)"
