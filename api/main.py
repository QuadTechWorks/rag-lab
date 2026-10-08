"""RAGx API — FastAPI application.

Single service, all phases share one API (adds routes per phase as they are built).
Runs on :8000. Streamlit UI talks to this exclusively.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("ragx.api")


def _discover_providers() -> None:
    """Import all provider modules so their @register decorators fire."""
    from core.registry import LOADERS, CHUNKERS, EMBEDDERS, VECTOR_STORES, RETRIEVERS

    providers_root = Path(__file__).parent.parent / "providers"

    loaded = LOADERS.discover(providers_root / "ingestion", "providers.ingestion")
    logger.info("Ingestion providers loaded: %s", loaded)

    loaded = CHUNKERS.discover(providers_root / "chunking", "providers.chunking")
    logger.info("Chunking providers loaded: %s", loaded)

    loaded = EMBEDDERS.discover(providers_root / "embedding", "providers.embedding")
    logger.info("Embedding providers loaded: %s", loaded)

    loaded = VECTOR_STORES.discover(providers_root / "vectordb", "providers.vectordb")
    logger.info("VectorDB providers loaded: %s", loaded)

    loaded = RETRIEVERS.discover(providers_root / "retriever", "providers.retriever")
    logger.info("Retriever providers loaded: %s", loaded)


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAGx",
        description="Multi-provider RAG experimentation platform — Phases 1–5",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _discover_providers()

    from api.routes.providers import router as providers_router
    from api.routes.ingestion import router as ingestion_router
    from api.routes.chunking import router as chunking_router
    from api.routes.embedding import router as embedding_router
    from api.routes.vectordb import router as vectordb_router
    from api.routes.retriever import router as retriever_router

    app.include_router(providers_router, prefix="/api")
    app.include_router(ingestion_router, prefix="/api")
    app.include_router(chunking_router, prefix="/api")
    app.include_router(embedding_router, prefix="/api")
    app.include_router(vectordb_router, prefix="/api")
    app.include_router(retriever_router, prefix="/api")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "ragx-api",
            "phases": ["ingestion", "chunking", "embedding", "vectordb", "retriever"],
        }

    return app


app = create_app()
