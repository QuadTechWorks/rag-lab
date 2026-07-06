from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.store import document_store
from api.chunk_store import chunk_store
from core.registry import CHUNKERS

router = APIRouter(prefix="/chunk", tags=["chunking"])


# ── Request models ────────────────────────────────────────────────────────────

class ChunkRequest(BaseModel):
    doc_ids: list[str] | None = None   # None = all docs in store
    provider: str
    chunker: str
    chunk_size: int = 512
    chunk_overlap: int = 50


class PreviewRequest(BaseModel):
    doc_id: str                        # single doc for preview
    provider: str
    chunker: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_preview: int = 5               # how many chunks to return


# ── Preview (no store write) ──────────────────────────────────────────────────

@router.post("/preview")
def preview_chunks(req: PreviewRequest) -> dict:
    """Chunk one document and return a preview — nothing is stored."""
    doc = document_store.get(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {req.doc_id}")

    try:
        chunker = CHUNKERS.create(provider=req.provider, name=req.chunker)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        chunks = chunker.chunk([doc], chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    preview = chunks[: req.max_preview]
    return {
        "doc_id":       req.doc_id,
        "doc_source":   doc.source,
        "total_chunks": len(chunks),
        "avg_chars":    round(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0,
        "preview":      [c.model_dump() for c in preview],
    }


# ── Run (chunk + store) ───────────────────────────────────────────────────────

@router.post("/run")
def run_chunking(req: ChunkRequest) -> dict:
    """Chunk documents and store the results in the chunk store."""
    if req.doc_ids:
        docs = [document_store.get(did) for did in req.doc_ids]
        missing = [did for did, d in zip(req.doc_ids, docs) if d is None]
        if missing:
            raise HTTPException(status_code=404, detail=f"Documents not found: {missing}")
        docs = [d for d in docs if d is not None]
    else:
        docs = document_store.all()

    if not docs:
        raise HTTPException(status_code=422, detail="No documents in store. Run ingestion first.")

    try:
        chunker = CHUNKERS.create(provider=req.provider, name=req.chunker)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    results = []
    total_chunks = 0
    errors = []

    for doc in docs:
        try:
            chunks = chunker.chunk([doc], chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
            chunk_store.add(chunks)
            total_chunks += len(chunks)
            results.append({
                "doc_id": doc.id,
                "source": doc.source,
                "status": "ok",
                "chunks": len(chunks),
            })
        except Exception as exc:
            errors.append({"doc_id": doc.id, "source": doc.source, "error": str(exc)})

    return {
        "provider":     req.provider,
        "chunker":      req.chunker,
        "chunk_size":   req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
        "docs_processed": len(results),
        "docs_failed":   len(errors),
        "total_chunks":  total_chunks,
        "stats":         chunk_store.stats(),
        "results":       results,
        "errors":        errors,
    }


# ── List chunks ───────────────────────────────────────────────────────────────

@router.get("/chunks")
def list_chunks(doc_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
    """List stored chunks — optionally filtered by source document."""
    if doc_id:
        chunks = chunk_store.by_doc(doc_id)
    else:
        chunks = chunk_store.all()

    page = chunks[offset: offset + limit]
    return {
        "total": len(chunks),
        "limit": limit,
        "offset": offset,
        "stats": chunk_store.stats(),
        "chunks": [c.model_dump() for c in page],
    }


# ── Delete chunks ─────────────────────────────────────────────────────────────

@router.delete("/chunks")
def clear_chunks() -> dict:
    chunk_store.clear()
    return {"status": "cleared"}


@router.delete("/chunks/{doc_id}")
def delete_chunks_by_doc(doc_id: str) -> dict:
    removed = chunk_store.delete_by_doc(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc: {doc_id}")
    return {"status": "deleted", "removed": removed, "doc_id": doc_id}
