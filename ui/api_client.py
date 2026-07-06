"""Thin HTTP client for all RAGx API calls from the Streamlit UI."""
from __future__ import annotations
import requests

API_BASE = "http://localhost:8000/api"
_TIMEOUT = 30


def health() -> dict:
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        return r.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


def get_providers() -> dict:
    r = requests.get(f"{API_BASE}/providers", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_ingestion_providers() -> dict:
    r = requests.get(f"{API_BASE}/providers/ingestion", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def ingest_file(file_bytes: bytes, filename: str, provider: str, loader: str) -> dict:
    r = requests.post(
        f"{API_BASE}/ingest/upload",
        files={"file": (filename, file_bytes)},
        data={"provider": provider, "loader": loader},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def ingest_url(url: str, provider: str, loader: str) -> dict:
    r = requests.post(
        f"{API_BASE}/ingest/url",
        json={"url": url, "provider": provider, "loader": loader},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def list_documents() -> dict:
    r = requests.get(f"{API_BASE}/ingest/documents", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def delete_document(doc_id: str) -> dict:
    r = requests.delete(f"{API_BASE}/ingest/documents/{doc_id}", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear_documents() -> dict:
    r = requests.delete(f"{API_BASE}/ingest/documents", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Phase 2: Chunking ─────────────────────────────────────────────────────────

def get_chunking_providers() -> dict:
    r = requests.get(f"{API_BASE}/providers/chunking", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def preview_chunks(doc_id: str, provider: str, chunker: str,
                   chunk_size: int, chunk_overlap: int, max_preview: int = 5) -> dict:
    r = requests.post(f"{API_BASE}/chunk/preview", timeout=_TIMEOUT, json={
        "doc_id": doc_id, "provider": provider, "chunker": chunker,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "max_preview": max_preview,
    })
    r.raise_for_status()
    return r.json()


def run_chunking(doc_ids: list | None, provider: str, chunker: str,
                 chunk_size: int, chunk_overlap: int) -> dict:
    r = requests.post(f"{API_BASE}/chunk/run", timeout=120, json={
        "doc_ids": doc_ids, "provider": provider, "chunker": chunker,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
    })
    r.raise_for_status()
    return r.json()


def list_chunks(doc_id: str | None = None, limit: int = 200) -> dict:
    params = {"limit": limit}
    if doc_id:
        params["doc_id"] = doc_id
    r = requests.get(f"{API_BASE}/chunk/chunks", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear_chunks() -> dict:
    r = requests.delete(f"{API_BASE}/chunk/chunks", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Phase 3: Embedding ────────────────────────────────────────────────────────

def get_embedding_providers() -> dict:
    r = requests.get(f"{API_BASE}/providers/embedding", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def preview_embed(chunk_id: str, provider: str, embedder: str,
                  use_langfuse: bool = False) -> dict:
    r = requests.post(f"{API_BASE}/embed/preview", timeout=60, json={
        "chunk_id": chunk_id, "provider": provider, "embedder": embedder,
        "use_langfuse": use_langfuse,
    })
    r.raise_for_status()
    return r.json()


def run_embedding(doc_ids: list | None, provider: str, embedder: str,
                  use_langfuse: bool = False, batch_size: int = 32) -> dict:
    r = requests.post(f"{API_BASE}/embed/run", timeout=300, json={
        "doc_ids": doc_ids, "provider": provider, "embedder": embedder,
        "use_langfuse": use_langfuse, "batch_size": batch_size,
    })
    r.raise_for_status()
    return r.json()


def list_vectors(doc_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
    params = {"limit": limit, "offset": offset}
    if doc_id:
        params["doc_id"] = doc_id
    r = requests.get(f"{API_BASE}/embed/vectors", params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def embedding_stats() -> dict:
    r = requests.get(f"{API_BASE}/embed/stats", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear_vectors() -> dict:
    r = requests.delete(f"{API_BASE}/embed/vectors", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── Phase 4: Vector DB ────────────────────────────────────────────────────────

def get_vectordb_providers() -> dict:
    r = requests.get(f"{API_BASE}/providers/vectordb", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def connect_vectordb(provider: str, store: str, config: dict | None = None) -> dict:
    r = requests.post(f"{API_BASE}/vectordb/connect", timeout=30, json={
        "provider": provider, "store": store, "config": config or {},
    })
    r.raise_for_status()
    return r.json()


def vectordb_status() -> dict:
    r = requests.get(f"{API_BASE}/vectordb/status", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def index_vectors(doc_ids: list | None = None) -> dict:
    r = requests.post(f"{API_BASE}/vectordb/index", timeout=300, json={
        "doc_ids": doc_ids,
    })
    r.raise_for_status()
    return r.json()


def search_text(query: str, top_k: int = 5, embed_provider: str = "ollama",
                embed_name: str = "nomic", filters: dict | None = None) -> dict:
    r = requests.post(f"{API_BASE}/vectordb/search-text", timeout=60, json={
        "query": query, "top_k": top_k,
        "embed_provider": embed_provider, "embed_name": embed_name,
        "filters": filters,
    })
    r.raise_for_status()
    return r.json()


def vectordb_stats() -> dict:
    r = requests.get(f"{API_BASE}/vectordb/stats", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear_vectordb() -> dict:
    r = requests.delete(f"{API_BASE}/vectordb/vectors", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()
