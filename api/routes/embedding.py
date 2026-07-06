from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.registry import EMBEDDERS
from api.chunk_store import chunk_store
from api.embedding_store import embedding_store

router = APIRouter(prefix="/embed", tags=["embedding"])


class PreviewRequest(BaseModel):
    chunk_id: str
    provider: str
    embedder: str
    use_langfuse: bool = False


class RunRequest(BaseModel):
    doc_ids: list[str] | None = None
    provider: str
    embedder: str
    use_langfuse: bool = False
    batch_size: int = 32


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/preview")
def preview_embed(req: PreviewRequest) -> dict:
    """Embed a single chunk — no storage, returns vector preview (first 10 dims)."""
    chunk = chunk_store.get(req.chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk {req.chunk_id!r} not found")

    if not EMBEDDERS.is_registered(req.provider, req.embedder):
        raise HTTPException(status_code=400,
                            detail=f"Embedder {req.provider}/{req.embedder} not registered")

    from core.tracing import get_tracer
    tracer = get_tracer(req.use_langfuse)

    try:
        embedder = EMBEDDERS.create(provider=req.provider, name=req.embedder)
        tracer.start_embed(embedder.model_name, [chunk.content])
        result = embedder.embed([chunk])
        tracer.end_embed([r.vector for r in result])
        tracer.flush()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    ec = result[0]
    return {
        "chunk_id": ec.chunk_id,
        "model": ec.model,
        "vector_dim": ec.vector_dim,
        "vector_preview": ec.vector[:10],
        "char_count": ec.char_count,
        "provider": ec.provider,
        "embedder": ec.embedder,
    }


@router.post("/run")
def run_embedding(req: RunRequest) -> dict:
    """Embed chunks (all or by doc_ids list) and store in the embedding store."""
    if not EMBEDDERS.is_registered(req.provider, req.embedder):
        raise HTTPException(status_code=400,
                            detail=f"Embedder {req.provider}/{req.embedder} not registered")

    if req.doc_ids is not None:
        chunks = []
        for doc_id in req.doc_ids:
            chunks.extend(chunk_store.by_doc(doc_id))
    else:
        chunks = chunk_store.all()

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found to embed")

    from core.tracing import get_tracer
    tracer = get_tracer(req.use_langfuse)

    try:
        embedder = EMBEDDERS.create(provider=req.provider, name=req.embedder)
        all_embedded = []
        for i in range(0, len(chunks), req.batch_size):
            batch = chunks[i: i + req.batch_size]
            texts = [c.content for c in batch]
            tracer.start_embed(embedder.model_name, texts)
            embedded = embedder.embed(batch)
            tracer.end_embed([e.vector for e in embedded])
            all_embedded.extend(embedded)
        tracer.flush()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    embedding_store.add(all_embedded)

    return {
        "embedded": len(all_embedded),
        "provider": req.provider,
        "embedder": req.embedder,
        "model": all_embedded[0].model if all_embedded else "",
        "vector_dim": all_embedded[0].vector_dim if all_embedded else 0,
        "store_total": embedding_store.stats()["count"],
    }


@router.get("/vectors")
def list_vectors(doc_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
    """List embedded chunks with optional doc_id filter."""
    items = embedding_store.get_page(doc_id=doc_id, limit=limit, offset=offset)
    return {
        "total": len(embedding_store.all()),
        "returned": len(items),
        "offset": offset,
        "vectors": [
            {
                "id": ec.id,
                "chunk_id": ec.chunk_id,
                "source_doc_id": ec.source_doc_id,
                "source": ec.source,
                "char_count": ec.char_count,
                "vector_dim": ec.vector_dim,
                "provider": ec.provider,
                "embedder": ec.embedder,
                "model": ec.model,
                "vector_preview": ec.vector[:5],
            }
            for ec in items
        ],
    }


@router.get("/stats")
def embedding_stats() -> dict:
    return embedding_store.stats()


@router.delete("/vectors")
def clear_vectors() -> dict:
    count = embedding_store.stats()["count"]
    embedding_store.clear()
    return {"cleared": count}
