from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.store import document_store
from core.registry import LOADERS
from core.routing.file_router import file_router

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# ── Request models ────────────────────────────────────────────────────────────

class URLIngestRequest(BaseModel):
    url: str
    provider: str
    loader: str


class LoadFileRequest(BaseModel):
    file_path: str
    provider: str | None = None
    loader: str | None = None


class FolderRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    max_file_size_mb: float = 50.0


# ── Upload file ───────────────────────────────────────────────────────────────

@router.post("/upload")
async def ingest_file(
    file: UploadFile = File(...),
    provider: str = Form(...),
    loader: str = Form(...),
) -> dict:
    suffix = Path(file.filename or "upload").suffix or ".txt"
    raw = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        loader_obj = LOADERS.create(provider=provider, name=loader)
        docs = loader_obj.load(tmp_path)
        for d in docs:
            d.metadata["filename"] = file.filename
        document_store.add(docs)
        return {"count": len(docs), "documents": [d.model_dump() for d in docs]}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)


# ── Ingest URL ────────────────────────────────────────────────────────────────

@router.post("/url")
def ingest_url(req: URLIngestRequest) -> dict:
    try:
        loader_obj = LOADERS.create(provider=req.provider, name=req.loader)
        docs = loader_obj.load(req.url)
        document_store.add(docs)
        return {"count": len(docs), "documents": [d.model_dump() for d in docs]}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Load single file by server-side path (used by UI for per-file progress) ──

@router.post("/file-path")
def load_file_by_path(req: LoadFileRequest) -> dict:
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    if req.provider and req.loader:
        provider, loader_name = req.provider, req.loader
    else:
        route = file_router.route(str(path))
        if route is None:
            raise HTTPException(status_code=422, detail=f"No loader for: {path.suffix}")
        provider, loader_name = route

    try:
        loader_obj = LOADERS.create(provider=provider, name=loader_name)
        docs = loader_obj.load(str(path))
        for d in docs:
            d.metadata.setdefault("filename", path.name)
        document_store.add(docs)
        return {
            "file":     path.name,
            "provider": provider,
            "loader":   loader_name,
            "docs":     len(docs),
            "doc_ids":  [d.id for d in docs],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Folder scan (preview, no loading) ────────────────────────────────────────

@router.post("/folder/scan")
def scan_folder(req: FolderRequest) -> dict:
    try:
        plan = file_router.scan_folder(
            req.folder_path,
            recursive=req.recursive,
            max_file_size_mb=req.max_file_size_mb,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    loadable = [f for f in plan if not f["skipped"]]
    skipped  = [f for f in plan if f["skipped"]]
    return {
        "total_files": len(plan),
        "loadable":    len(loadable),
        "skipped":     len(skipped),
        "plan":        plan,
    }


# ── Folder load (bulk, all at once) ──────────────────────────────────────────

@router.post("/folder/load")
def load_folder(req: FolderRequest) -> dict:
    try:
        plan = file_router.scan_folder(
            req.folder_path,
            recursive=req.recursive,
            max_file_size_mb=req.max_file_size_mb,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    results, total_docs = [], 0
    for entry in plan:
        if entry["skipped"]:
            results.append({"file": entry["file"], "status": "skipped",
                            "reason": entry.get("skip_reason", "unsupported"), "docs": 0})
            continue
        try:
            loader_obj = LOADERS.create(provider=entry["provider"], name=entry["loader"])
            docs = loader_obj.load(entry["path"])
            document_store.add(docs)
            total_docs += len(docs)
            results.append({"file": entry["file"], "status": "ok",
                            "provider": entry["provider"], "loader": entry["loader"],
                            "docs": len(docs)})
        except Exception as exc:
            results.append({"file": entry["file"], "status": "error",
                            "provider": entry["provider"], "loader": entry["loader"],
                            "error": str(exc), "docs": 0})

    ok  = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    skp = sum(1 for r in results if r["status"] == "skipped")
    return {"total_files": len(plan), "loaded_files": ok, "error_files": err,
            "skipped_files": skp, "total_documents": total_docs, "results": results}


# ── List / Delete documents ───────────────────────────────────────────────────

@router.get("/documents")
def list_documents() -> dict:
    docs = document_store.all()
    return {"count": len(docs), "documents": [d.model_dump() for d in docs]}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not document_store.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "id": doc_id}


@router.delete("/documents")
def clear_documents() -> dict:
    document_store.clear()
    return {"status": "cleared"}
