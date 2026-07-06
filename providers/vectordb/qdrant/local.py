from __future__ import annotations
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_COLLECTION = "ragx_chunks"


@VECTOR_STORES.register(provider="qdrant", name="local")
class QdrantLocalStore(BaseVectorStore):
    """Qdrant local persistent store — data in ./ragx_qdrant directory.

    Requires: pip install qdrant-client
    Config: path (default: ./ragx_qdrant)
    """

    def __init__(self, path: str = "./ragx_qdrant", **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._client = None
        self._dim: int | None = None

    def _ensure_ready(self, dim: int | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("pip install qdrant-client")

        if self._client is None:
            self._client = QdrantClient(path=self._path)

        if dim is not None and dim != self._dim:
            existing = [c.name for c in self._client.get_collections().collections]
            if _COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self._dim = dim

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        from qdrant_client.models import PointStruct
        dim = embedded_chunks[0].vector_dim
        self._ensure_ready(dim)
        points = [
            PointStruct(
                id=abs(hash(ec.chunk_id)) % (2**63),
                vector=ec.vector,
                payload={
                    "chunk_id": ec.chunk_id,
                    "source_doc_id": ec.source_doc_id,
                    "source": ec.source,
                    "content": ec.content,
                    "provider": ec.provider,
                    "embedder": ec.embedder,
                    "model": ec.model,
                    **{k: str(v) for k, v in ec.metadata.items()},
                },
            )
            for ec in embedded_chunks
        ]
        self._client.upsert(collection_name=_COLLECTION, points=points)
        info = self._client.get_collection(_COLLECTION)
        return {"indexed": len(embedded_chunks), "total": info.points_count}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        self._ensure_ready(len(query_vector))
        results = self._client.search(
            collection_name=_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                rank=i + 1,
                score=hit.score,
                chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                source_doc_id=hit.payload.get("source_doc_id", ""),
                source=hit.payload.get("source", ""),
                content=hit.payload.get("content", ""),
                metadata={k: v for k, v in hit.payload.items()
                          if k not in ("chunk_id", "source_doc_id", "source", "content",
                                       "provider", "embedder", "model")},
                provider="qdrant",
                store="local",
            )
            for i, hit in enumerate(results)
        ]

    def delete(self, ids: list[str]) -> int:
        if self._client is None:
            return 0
        from qdrant_client.models import PointIdsList
        int_ids = [abs(hash(cid)) % (2**63) for cid in ids]
        self._client.delete(collection_name=_COLLECTION,
                            points_selector=PointIdsList(points=int_ids))
        return len(ids)

    def clear(self) -> None:
        if self._client is None:
            return
        existing = [c.name for c in self._client.get_collections().collections]
        if _COLLECTION in existing:
            self._client.delete_collection(_COLLECTION)
        self._dim = None

    def stats(self) -> dict:
        if self._client is None:
            return {"count": 0, "store": "qdrant/local", "path": self._path}
        try:
            info = self._client.get_collection(_COLLECTION)
            return {"count": info.points_count, "store": "qdrant/local",
                    "path": self._path, "vector_dim": self._dim}
        except Exception:
            return {"count": 0, "store": "qdrant/local", "path": self._path}

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "Qdrant (local persistent)"
