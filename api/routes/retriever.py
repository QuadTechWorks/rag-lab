from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.registry import RETRIEVERS, EMBEDDERS
from api.embedding_store import embedding_store
from api.retrieval_store import retrieval_store
from api.vector_store_manager import vector_store_manager

router = APIRouter(prefix="/retrieve", tags=["retriever"])

# Retrievers that query the active vector store with an embedded query.
_NEEDS_VECTOR_STORE = {("vector", "dense"), ("mmr", "cosine")}


class RunRequest(BaseModel):
    provider: str
    retriever: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict | None = None
    config: dict = {}
    embed_provider: str = "ollama"   # only used by vector-store-backed retrievers
    embed_name: str = "nomic"


@router.post("/run")
def run_retriever(req: RunRequest) -> dict:
    """Run one retriever over the embedded chunks / active vector store."""
    if not RETRIEVERS.is_registered(req.provider, req.retriever):
        raise HTTPException(
            status_code=400,
            detail=f"Retriever {req.provider}/{req.retriever} not registered. "
                   f"Available: {RETRIEVERS.available()}",
        )

    kwargs = dict(req.config)
    if (req.provider, req.retriever) in _NEEDS_VECTOR_STORE:
        try:
            kwargs["vector_store"] = vector_store_manager.get_active()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not EMBEDDERS.is_registered(req.embed_provider, req.embed_name):
            raise HTTPException(
                status_code=400,
                detail=f"Embedder {req.embed_provider}/{req.embed_name} not found",
            )
        try:
            kwargs["embedder"] = EMBEDDERS.create(provider=req.embed_provider,
                                                  name=req.embed_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Embedder error: {exc}")
    else:
        if not embedding_store.all():
            raise HTTPException(
                status_code=404,
                detail="No embedded chunks found. Run Phase 3 (POST /api/embed/run) first.",
            )

    try:
        retriever = RETRIEVERS.create(req.provider, req.retriever, **kwargs)
        t0 = time.perf_counter()
        retriever.index(embedding_store.all())
        t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        results = retriever.retrieve(req.query, top_k=req.top_k, filters=req.filters)
        t_query = time.perf_counter() - t0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    run = {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": req.provider,
        "retriever": req.retriever,
        "retriever_name": retriever.retriever_name,
        "query": req.query,
        "top_k": req.top_k,
        "config": req.config,
        "index_ms": round(t_index * 1000, 1),
        "query_ms": round(t_query * 1000, 1),
        "count": len(results),
        "results": [r.model_dump() for r in results],
    }
    retrieval_store.add(run)
    return run


@router.get("/results")
def list_results(limit: int = 20) -> dict:
    """Most recent retrieval runs, newest first."""
    runs = retrieval_store.all()[::-1][:limit]
    return {"runs": runs, "count": len(runs)}


@router.delete("/results")
def clear_results() -> dict:
    retrieval_store.clear()
    return {"cleared": True}
