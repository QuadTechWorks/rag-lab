from __future__ import annotations
import json
import os
from core.registry import VECTOR_STORES
from core.interfaces.base_vector_store import BaseVectorStore
from core.models.embeddings import EmbeddedChunk
from core.models.search import SearchResult

_TABLE = "ragx_chunks"


@VECTOR_STORES.register(provider="pgvector", name="postgres")
class PgVectorStore(BaseVectorStore):
    """pgvector — vector similarity search inside PostgreSQL.

    Requires: pip install psycopg2-binary pgvector
    Env var: RAGX_PGVECTOR_URL (e.g. postgresql://user:pass@localhost/ragx)
    """

    def __init__(self, dsn: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._dsn = dsn or os.environ.get("RAGX_PGVECTOR_URL",
                                          "postgresql://localhost/ragx")
        self._conn = None
        self._dim: int | None = None

    def _ensure_ready(self, dim: int | None = None) -> None:
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError("pip install psycopg2-binary pgvector")

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            register_vector(self._conn)
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

        if dim is not None and dim != self._dim:
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE} (
                        id TEXT PRIMARY KEY,
                        source_doc_id TEXT,
                        source TEXT,
                        content TEXT,
                        embedding vector({dim}),
                        provider TEXT,
                        embedder TEXT,
                        model TEXT,
                        metadata JSONB
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {_TABLE}_embedding_idx
                    ON {_TABLE} USING hnsw (embedding vector_cosine_ops)
                """)
            self._conn.commit()
            self._dim = dim

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        if not embedded_chunks:
            return {"indexed": 0, "total": 0}
        import numpy as np
        dim = embedded_chunks[0].vector_dim
        self._ensure_ready(dim)

        with self._conn.cursor() as cur:
            for ec in embedded_chunks:
                cur.execute(
                    f"""INSERT INTO {_TABLE}
                        (id, source_doc_id, source, content, embedding, provider, embedder, model, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                    """,
                    (ec.chunk_id, ec.source_doc_id, ec.source, ec.content,
                     np.array(ec.vector), ec.provider, ec.embedder, ec.model,
                     json.dumps(ec.metadata)),
                )
            self._conn.commit()
            cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
            total = cur.fetchone()[0]

        return {"indexed": len(embedded_chunks), "total": total}

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        import numpy as np
        self._ensure_ready(len(query_vector))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, source_doc_id, source, content, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM {_TABLE}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """,
                (np.array(query_vector), np.array(query_vector), top_k),
            )
            rows = cur.fetchall()

        return [
            SearchResult(
                rank=i + 1,
                score=float(row[5]),
                chunk_id=row[0],
                source_doc_id=row[1],
                source=row[2],
                content=row[3],
                metadata=row[4] or {},
                provider="pgvector",
                store="postgres",
            )
            for i, row in enumerate(rows)
        ]

    def delete(self, ids: list[str]) -> int:
        if self._conn is None:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {_TABLE} WHERE id = ANY(%s)", (ids,))
            deleted = cur.rowcount
        self._conn.commit()
        return deleted

    def clear(self) -> None:
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {_TABLE}")
        self._conn.commit()

    def stats(self) -> dict:
        if self._conn is None:
            return {"count": 0, "store": "pgvector/postgres"}
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
                count = cur.fetchone()[0]
            return {"count": count, "store": "pgvector/postgres",
                    "vector_dim": self._dim, "dsn": self._dsn.split("@")[-1]}
        except Exception:
            return {"count": 0, "store": "pgvector/postgres"}

    @property
    def is_ready(self) -> bool:
        try:
            self._ensure_ready()
            return True
        except Exception:
            return False

    @property
    def store_name(self) -> str:
        return "pgvector (PostgreSQL)"
