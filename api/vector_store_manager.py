"""Manages the single active vector store instance (Phase 4)."""
from __future__ import annotations
from core.interfaces.base_vector_store import BaseVectorStore
from core.registry import VECTOR_STORES


class VectorStoreManager:
    def __init__(self) -> None:
        self._active: BaseVectorStore | None = None
        self._active_provider: str | None = None
        self._active_store: str | None = None

    def connect(self, provider: str, store: str, config: dict | None = None) -> BaseVectorStore:
        instance = VECTOR_STORES.create(provider=provider, name=store,
                                        **(config or {}))
        self._active = instance
        self._active_provider = provider
        self._active_store = store
        return instance

    def get_active(self) -> BaseVectorStore:
        if self._active is None:
            raise RuntimeError(
                "No vector store connected. POST /api/vectordb/connect first."
            )
        return self._active

    def status(self) -> dict:
        if self._active is None:
            return {"connected": False}
        try:
            ready = self._active.is_ready
            name = self._active.store_name
        except Exception:
            ready = False
            name = "unknown"
        return {
            "connected": True,
            "provider": self._active_provider,
            "store": self._active_store,
            "store_name": name,
            "is_ready": ready,
        }

    def disconnect(self) -> None:
        self._active = None
        self._active_provider = None
        self._active_store = None


vector_store_manager = VectorStoreManager()
