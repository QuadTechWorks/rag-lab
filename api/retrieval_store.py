"""In-memory store for retrieval runs (Phase 5)."""
from __future__ import annotations
from typing import Any


class RetrievalStore:
    def __init__(self) -> None:
        self._runs: list[dict[str, Any]] = []

    def add(self, run: dict[str, Any]) -> None:
        self._runs.append(run)

    def all(self) -> list[dict[str, Any]]:
        return list(self._runs)

    def clear(self) -> None:
        self._runs.clear()


retrieval_store = RetrievalStore()
