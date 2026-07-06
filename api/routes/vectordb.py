from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.registry import VECTOR_STORES, EMBEDDERS
from api.embedding_store import embedding_store
from api.vector_store_manager import vector_store_manager

router = APIRouter(prefix="/vectordb", tags=["vectordb"])


class ConnectRequest(BaseModel):
    provider: str
    store: str
    config: dict = {}


class IndexRequest(BaseModel):
    doc_ids: list[str] | None = None  # None = index all embedded chunks


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = 5
    filters: dict | None = None


class SearchTextRequest(BaseModel):
    query: str
    top_k: int = 5
    embed_provider: str = "ollama"
    embed_name: str = "nomic"
    filters: dict | None = None


class DeleteRequest(BaseModel):
    chunk_ids: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/connect")
def connect_store(req: ConnectRequest) -> dict:
    """Set the active vector store. Creates the store instance with optional config."""
    if not VECTOR_STORES.is_registered(req.provider, req.store):
        available = VECTOR_STORES.available()
        raise HTTPException(
            status_code=400,
            detail=f"Vector store {req.provider}/{req.store} not registered. "
                   f"Available: {available}",
        )
    try:
        vector_store_manager.connect(req.provider, req.store, req.config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return vector_store_manager.status()


@router.get("/status")
def store_status() -> dict:
    """Active vector store connection status."""
    return vector_store_manager.status()


@router.post("/index")
def index_vectors(req: IndexRequest) -> dict:
    """Index EmbeddedChunks from the embedding store into the active vector store."""
    store = _active_store()

    if req.doc_ids is not None:
        items = []
        for doc_id in req.doc_ids:
            items.extend(embedding_store.by_doc(doc_id))
    else:
        items = embedding_store.all()

    if not items:
        raise HTTPException(status_code=404,
                            detail="No embedded chunks found. Run POST /api/embed/run first.")

    try:
        result = store.index(items)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {**result, "store": vector_store_manager.status()}


@router.post("/search")
def search_vectors(req: SearchRequest) -> dict:
    """Search with a pre-computed query vector."""
    store = _active_store()
    try:
        results = store.search(req.query_vector, top_k=req.top_k,
                               filters=req.filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"results": [r.model_dump() for r in results], "count": len(results)}


@router.post("/search-text")
def search_text(req: SearchTextRequest) -> dict:
    """Search with a text query — embeds it first, then searches the active store."""
    store = _active_store()

    if not EMBEDDERS.is_registered(req.embed_provider, req.embed_name):
        raise HTTPException(
            status_code=400,
            detail=f"Embedder {req.embed_provider}/{req.embed_name} not found",
        )
    try:
        from core.models.chunks import DocumentChunk
        # Build a dummy chunk to use the embedder interface
        dummy = DocumentChunk(
            content=req.query,
            source_doc_id="query",
            source="query",
            chunk_index=0,
            provider="_query",
            chunker="_query",
            chunk_size=len(req.query),
            chunk_overlap=0,
        )
        embedder = EMBEDDERS.create(provider=req.embed_provider, name=req.embed_name)
        embedded = embedder.embed([dummy])
        query_vector = embedded[0].vector
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embed error: {exc}")

    try:
        results = store.search(query_vector, top_k=req.top_k, filters=req.filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")

    return {
        "query": req.query,
        "embed_model": embedded[0].model,
        "vector_dim": len(query_vector),
        "results": [r.model_dump() for r in results],
        "count": len(results),
    }


@router.get("/stats")
def vectordb_stats() -> dict:
    """Statistics from the active vector store."""
    store = _active_store()
    try:
        return store.stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/vectors")
def clear_store() -> dict:
    """Clear all vectors from the active vector store."""
    store = _active_store()
    try:
        store.clear()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"cleared": True, "store": vector_store_manager.status()}


@router.delete("/vectors/batch")
def delete_by_ids(req: DeleteRequest) -> dict:
    """Delete specific chunk IDs from the active vector store."""
    store = _active_store()
    try:
        deleted = store.delete(req.chunk_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"deleted": deleted}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_store():
    try:
        return vector_store_manager.get_active()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
