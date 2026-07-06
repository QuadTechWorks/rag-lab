from __future__ import annotations
import json
import os
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_CLASS = "RagxChunk"


@VECTOR_STORES.register(provider="weaviate", name="cloud")
class WeaviateCloudStore(BaseVectorStore):
    """Weaviate Cloud sandbox — free 14-day sandbox at weaviate.io/cloud.

    Requires: pip install weaviate-client>=4.0.0
    Env vars: WEAVIATE_URL, WEAVIATE_API_KEY
    """

    def __init__(self, url: str | None = None, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._url = url or os.environ.get("WEAVIATE_URL", "")
        self._api_key = api_key or os.environ.get("WEAVIATE_API_KEY", "")
        self._client = None
        self._collection = None

    def _ensure_ready(self) -> None:
        try:
            import weaviate
            from weaviate.auth import Auth
        except ImportError:
            raise ImportError("pip install 'weaviate-client>=4.0.0'")
        if not self._url:
            raise RuntimeError("WEAVIATE_URL env var is required for weaviate/cloud")

        if self._client is None:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._url,
                auth_credentials=Auth.api_key(self._api_key),
            )

        if not self._client.collections.exists(_CLASS):
            import weaviate.classes.config as wvc
            self._client.collections.create(
                name=_CLASS,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
                properties=[
                    wvc.Property(name="chunk_id", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                    wvc.Property(name="source_doc_id", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                    wvc.Property(name="source", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                    wvc.Property(name="content", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="metadata_json", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                    wvc.Property(name="provider", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                    wvc.Property(name="embedder", data_type=wvc.DataType.TEXT,
                                 skip_vectorization=True),
                ],
            )
        self._collection = self._client.collections.get(_CLASS)

    def _chunk_uuid(self, chunk_id: str) -> str:
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        self._ensure_ready()
        with self._collection.batch.dynamic() as batch:
            for ec in embedded_chunks:
                batch.add_object(
                    properties={
                        "chunk_id": ec.chunk_id,
                        "source_doc_id": ec.source_doc_id,
                        "source": ec.source,
                        "content": ec.content,
                        "metadata_json": json.dumps(ec.metadata),
                        "provider": ec.provider,
                        "embedder": ec.embedder,
                    },
                    vector=ec.vector,
                    uuid=self._chunk_uuid(ec.chunk_id),
                )
        agg = self._collection.aggregate.over_all()
        return {"indexed": len(embedded_chunks), "total": agg.total_count}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        self._ensure_ready()
        from weaviate.classes.query import MetadataQuery
        resp = self._collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            SearchResult(
                rank=i + 1,
                score=float(1 - obj.metadata.distance) if obj.metadata.distance else 0.0,
                chunk_id=obj.properties.get("chunk_id", ""),
                source_doc_id=obj.properties.get("source_doc_id", ""),
                source=obj.properties.get("source", ""),
                content=obj.properties.get("content", ""),
                metadata=json.loads(obj.properties.get("metadata_json", "{}")),
                provider="weaviate",
                store="cloud",
            )
            for i, obj in enumerate(resp.objects)
        ]

    def delete(self, ids: list[str]) -> int:
        if self._collection is None:
            return 0
        for cid in ids:
            self._collection.data.delete_by_id(self._chunk_uuid(cid))
        return len(ids)

    def clear(self) -> None:
        if self._client is None:
            return
        if self._client.collections.exists(_CLASS):
            self._client.collections.delete(_CLASS)
        self._collection = None

    def stats(self) -> dict:
        if self._collection is None:
            return {"count": 0, "store": "weaviate/cloud", "url": self._url}
        try:
            agg = self._collection.aggregate.over_all()
            return {"count": agg.total_count, "store": "weaviate/cloud", "url": self._url}
        except Exception:
            return {"count": 0, "store": "weaviate/cloud", "url": self._url}

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "Weaviate Cloud"
