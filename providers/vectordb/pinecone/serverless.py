from __future__ import annotations
import os
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_INDEX_NAME = "ragx-chunks"


@VECTOR_STORES.register(provider="pinecone", name="serverless")
class PineconeServerlessStore(BaseVectorStore):
    """Pinecone serverless — cloud vector DB, free starter tier.

    Requires: pip install pinecone
    Env vars: PINECONE_API_KEY
    Optional: PINECONE_INDEX_HOST (if index already created)
    Config: cloud (default: aws), region (default: us-east-1)
    """

    def __init__(self, cloud: str = "aws", region: str = "us-east-1", **kwargs):
        super().__init__(**kwargs)
        self._cloud = cloud
        self._region = region
        self._pc = None
        self._index = None
        self._dim: int | None = None

    def _ensure_ready(self, dim: int | None = None) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError("pip install pinecone")

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY env var is required")

        if self._pc is None:
            self._pc = Pinecone(api_key=api_key)

        if dim is not None and self._index is None:
            existing = [i.name for i in self._pc.list_indexes()]
            if _INDEX_NAME not in existing:
                self._pc.create_index(
                    name=_INDEX_NAME, dimension=dim, metric="cosine",
                    spec=ServerlessSpec(cloud=self._cloud, region=self._region),
                )
            self._index = self._pc.Index(_INDEX_NAME)
            self._dim = dim

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        dim = embedded_chunks[0].vector_dim
        self._ensure_ready(dim)

        vectors = [
            (ec.chunk_id, ec.vector, {
                "source_doc_id": ec.source_doc_id,
                "source": ec.source,
                "content": ec.content[:1000],  # Pinecone metadata limit
                "provider": ec.provider,
                "embedder": ec.embedder,
                "model": ec.model,
            })
            for ec in embedded_chunks
        ]
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i:i+100])

        stats = self._index.describe_index_stats()
        return {"indexed": len(embedded_chunks),
                "total": stats.get("total_vector_count", 0)}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        self._ensure_ready(len(query_vector))
        kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
        if filters:
            kwargs["filter"] = filters
        resp = self._index.query(**kwargs)

        return [
            SearchResult(
                rank=i + 1,
                score=float(m.score),
                chunk_id=m.id,
                source_doc_id=m.metadata.get("source_doc_id", ""),
                source=m.metadata.get("source", ""),
                content=m.metadata.get("content", ""),
                metadata={k: v for k, v in m.metadata.items()
                          if k not in ("source_doc_id", "source", "content",
                                       "provider", "embedder", "model")},
                provider="pinecone",
                store="serverless",
            )
            for i, m in enumerate(resp.matches)
        ]

    def delete(self, ids: list[str]) -> int:
        if self._index is None:
            return 0
        self._index.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        if self._index is None:
            return
        self._index.delete(delete_all=True)

    def stats(self) -> dict:
        if self._index is None:
            return {"count": 0, "store": "pinecone/serverless"}
        s = self._index.describe_index_stats()
        return {"count": s.get("total_vector_count", 0),
                "store": "pinecone/serverless",
                "vector_dim": self._dim,
                "cloud": self._cloud,
                "region": self._region}

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "Pinecone Serverless"
