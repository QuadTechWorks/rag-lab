from __future__ import annotations
import json
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_COLLECTION = "ragx_chunks"


@VECTOR_STORES.register(provider="milvus", name="local")
class MilvusLocalStore(BaseVectorStore):
    """Milvus Lite — local file-based store, no server required.

    Requires: pip install pymilvus>=2.4.0
    Config: path (default: ./ragx_milvus.db)
    For full Milvus server: set uri to http://localhost:19530
    """

    def __init__(self, path: str = "./ragx_milvus.db", uri: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._uri = uri or path
        self._client = None
        self._dim: int | None = None

    def _ensure_ready(self, dim: int | None = None) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("pip install 'pymilvus>=2.4.0'")

        if self._client is None:
            self._client = MilvusClient(self._uri)

        if dim is not None and dim != self._dim:
            if not self._client.has_collection(_COLLECTION):
                self._client.create_collection(
                    collection_name=_COLLECTION,
                    dimension=dim,
                    primary_field_name="pk",
                    vector_field_name="embedding",
                    metric_type="COSINE",
                    auto_id=True,
                )
            self._dim = dim

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        dim = embedded_chunks[0].vector_dim
        self._ensure_ready(dim)

        data = [
            {
                "embedding": ec.vector,
                "chunk_id": ec.chunk_id,
                "source_doc_id": ec.source_doc_id,
                "source": ec.source,
                "content": ec.content[:65535],  # Milvus VARCHAR limit
                "provider": ec.provider,
                "embedder": ec.embedder,
                "model": ec.model,
                "metadata_json": json.dumps(ec.metadata)[:65535],
            }
            for ec in embedded_chunks
        ]
        self._client.insert(collection_name=_COLLECTION, data=data)
        stats = self._client.get_collection_stats(_COLLECTION)
        total = int(stats.get("row_count", 0))
        return {"indexed": len(embedded_chunks), "total": total}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        self._ensure_ready(len(query_vector))
        results = self._client.search(
            collection_name=_COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "source_doc_id", "source", "content", "metadata_json"],
        )
        output = []
        for rank, hit in enumerate(results[0]):
            fields = hit.get("entity", {})
            output.append(SearchResult(
                rank=rank + 1,
                score=float(hit.get("distance", 0)),
                chunk_id=fields.get("chunk_id", ""),
                source_doc_id=fields.get("source_doc_id", ""),
                source=fields.get("source", ""),
                content=fields.get("content", ""),
                metadata=json.loads(fields.get("metadata_json", "{}")),
                provider="milvus",
                store="local",
            ))
        return output

    def delete(self, ids: list[str]) -> int:
        if self._client is None:
            return 0
        expr = f"chunk_id in {ids}"
        self._client.delete(collection_name=_COLLECTION, filter=expr)
        return len(ids)

    def clear(self) -> None:
        if self._client is None:
            return
        if self._client.has_collection(_COLLECTION):
            self._client.drop_collection(_COLLECTION)
        self._dim = None

    def stats(self) -> dict:
        if self._client is None:
            return {"count": 0, "store": "milvus/local", "uri": self._uri}
        try:
            s = self._client.get_collection_stats(_COLLECTION)
            return {"count": int(s.get("row_count", 0)), "store": "milvus/local",
                    "uri": self._uri, "vector_dim": self._dim}
        except Exception:
            return {"count": 0, "store": "milvus/local", "uri": self._uri}

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "Milvus Lite (local file)"
