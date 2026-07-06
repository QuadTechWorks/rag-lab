from __future__ import annotations
import json
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_TABLE = "ragx_chunks"


@VECTOR_STORES.register(provider="lancedb", name="local")
class LanceDBLocalStore(BaseVectorStore):
    """LanceDB local columnar store — serverless, files in ./ragx_lance.

    Requires: pip install lancedb
    Config: path (default: ./ragx_lance)
    """

    def __init__(self, path: str = "./ragx_lance", **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._db = None
        self._table = None
        self._count = 0

    def _ensure_ready(self) -> None:
        try:
            import lancedb
        except ImportError:
            raise ImportError("pip install lancedb")
        if self._db is None:
            self._db = lancedb.connect(self._path)

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": self._count}
        self._ensure_ready()

        records = [
            {
                "id": ec.chunk_id,
                "vector": ec.vector,
                "content": ec.content,
                "source_doc_id": ec.source_doc_id,
                "source": ec.source,
                "provider": ec.provider,
                "embedder": ec.embedder,
                "model": ec.model,
                "metadata_json": json.dumps(ec.metadata),
            }
            for ec in embedded_chunks
        ]

        existing = self._db.table_names()
        if _TABLE not in existing:
            self._table = self._db.create_table(_TABLE, data=records)
        else:
            self._table = self._db.open_table(_TABLE)
            self._table.add(records)

        self._count = self._table.count_rows()
        return {"indexed": len(embedded_chunks), "total": self._count}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        self._ensure_ready()
        if self._table is None:
            if _TABLE not in self._db.table_names():
                return []
            self._table = self._db.open_table(_TABLE)

        df = self._table.search(query_vector).limit(top_k).to_pandas()
        results = []
        for rank, row in enumerate(df.itertuples()):
            results.append(SearchResult(
                rank=rank + 1,
                score=float(1 - getattr(row, "_distance", 0)),
                chunk_id=row.id,
                source_doc_id=row.source_doc_id,
                source=row.source,
                content=row.content,
                metadata=json.loads(row.metadata_json or "{}"),
                provider="lancedb",
                store="local",
            ))
        return results

    def delete(self, ids: list[str]) -> int:
        if self._table is None:
            return 0
        id_list = ", ".join(f"'{i}'" for i in ids)
        self._table.delete(f"id IN ({id_list})")
        self._count = self._table.count_rows()
        return len(ids)

    def clear(self) -> None:
        self._ensure_ready()
        if _TABLE in self._db.table_names():
            self._db.drop_table(_TABLE)
        self._table = None
        self._count = 0

    def stats(self) -> dict:
        return {
            "count": self._count,
            "store": "lancedb/local",
            "path": self._path,
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
        return "LanceDB (local columnar)"
