"""Folder ingestion — scan a local directory, auto-route each file to its best loader."""
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.store import document_store
from core.registry import LOADERS
from core.routing.file_router import file_router

router = APIRouter(tags=["folder-ingestion"])


class ScanRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    max_file_size_mb: float = 50.0


class LoadFolderRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    max_file_size_mb: float = 50.0


class LoadFileRequest(BaseModel):
    file_path: str           # absolute path on the server
    provider: str | None = None   # override auto-detected provider
    loader: str | None = None     # override auto-detected loader


# ── Load single file by path (used by UI for per-file progress) ───────────────

@router.post("/ingest/file-path")
def load_file_by_path(req: LoadFileRequest) -> dict:
    """Load one file from its server-side path. UI calls this per-file to show progress."""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    # Auto-detect if not overridden
    if req.provider and req.loader:
        provider, loader_name = req.provider, req.loader
    else:
        route = file_router.route(str(path))
        if route is None:
            raise HTTPException(status_code=422, detail=f"No loader available for: {path.suffix}")
        provider, loader_name = route

    try:
        loader_obj = LOADERS.create(provider=provider, name=loader_name)
        docs = loader_obj.load(str(path))
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


# ── Scan (preview — no loading) ───────────────────────────────────────────────

@router.post("/ingest/folder/scan")
def scan_folder(req: ScanRequest) -> dict:
    """Return the routing plan for all files in the folder without loading them."""
    try:
        plan = file_router.scan_folder(
            req.folder_path,
            recursive=req.recursive,
            max_file_size_mb=req.max_file_size_mb,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    loadable  = [f for f in plan if not f["skipped"]]
    skipped   = [f for f in plan if f["skipped"]]

    return {
        "total_files": len(plan),
        "loadable":    len(loadable),
        "skipped":     len(skipped),
        "plan":        plan,
    }


# ── Load all files ────────────────────────────────────────────────────────────

@router.post("/ingest/folder/load")
def load_folder(req: LoadFolderRequest) -> dict:
    """Scan folder, auto-dispatch each file to its loader, store all documents."""
    try:
        plan = file_router.scan_folder(
            req.folder_path,
            recursive=req.recursive,
            max_file_size_mb=req.max_file_size_mb,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    results = []
    total_docs = 0

    for entry in plan:
        if entry["skipped"]:
            results.append({
                "file":    entry["file"],
                "status":  "skipped",
                "reason":  entry.get("skip_reason", "Unsupported type"),
                "docs":    0,
            })
            continue

        provider = entry["provider"]
        loader   = entry["loader"]
        path     = entry["path"]

        try:
            loader_obj = LOADERS.create(provider=provider, name=loader)
            docs = loader_obj.load(path)
            document_store.add(docs)
            total_docs += len(docs)
            results.append({
                "file":     entry["file"],
                "status":   "ok",
                "provider": provider,
                "loader":   loader,
                "docs":     len(docs),
            })
        except Exception as exc:
            results.append({
                "file":     entry["file"],
                "status":   "error",
                "provider": provider,
                "loader":   loader,
                "error":    str(exc),
                "docs":     0,
            })

    ok      = [r for r in results if r["status"] == "ok"]
    errors  = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    return {
        "total_files":    len(plan),
        "loaded_files":   len(ok),
        "error_files":    len(errors),
        "skipped_files":  len(skipped),
        "total_documents": total_docs,
        "results":        results,
    }
