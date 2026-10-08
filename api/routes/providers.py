from fastapi import APIRouter
from core.registry import LOADERS, CHUNKERS, EMBEDDERS, VECTOR_STORES, RETRIEVERS, LLMS

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("")
def list_all_providers() -> dict:
    """Return all available providers grouped by pipeline phase."""
    return {
        "ingestion":  LOADERS.available(),
        "chunking":   CHUNKERS.available(),
        "embedding":  EMBEDDERS.available(),
        "vectordb":   VECTOR_STORES.available(),
        "retriever":  RETRIEVERS.available(),
        "llm":        LLMS.available(),
    }


@router.get("/ingestion")
def list_ingestion_providers() -> dict:
    return LOADERS.available()


@router.get("/chunking")
def list_chunking_providers() -> dict:
    return CHUNKERS.available()


@router.get("/embedding")
def list_embedding_providers() -> dict:
    return EMBEDDERS.available()


@router.get("/vectordb")
def list_vectordb_providers() -> dict:
    return VECTOR_STORES.available()


@router.get("/retriever")
def list_retriever_providers() -> dict:
    return RETRIEVERS.available()


@router.get("/llm")
def list_llm_providers() -> dict:
    return LLMS.available()
