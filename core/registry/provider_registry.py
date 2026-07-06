"""Provider Registry — the plug-and-play backbone of RAGx.

Every phase (ingestion, chunking, embedding, …) has one ProviderRegistry.
Providers decorate their classes with @REGISTRY.register(provider, name),
which stores a (provider, name) → class mapping. The pipeline and UI resolve
names from config/UI selection at runtime.

Adding a new provider = drop a .py file in the right directory,
implement the ABC, add the @register decorator. Zero other changes.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class ProviderRegistry:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self._items: dict[str, dict[str, type]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, provider: str, name: str):
        """Class decorator: @LOADERS.register('langchain', 'pdf')"""
        def decorator(cls: type) -> type:
            if provider not in self._items:
                self._items[provider] = {}
            key = name.lower()
            if key in self._items[provider]:
                existing = self._items[provider][key].__name__
                raise ValueError(
                    f"[ragx] {self.phase}/{provider}/{name} already registered by {existing}."
                )
            self._items[provider][key] = cls
            return cls
        return decorator

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, provider: str, name: str) -> type:
        p = provider.lower()
        n = name.lower()
        if p not in self._items or n not in self._items[p]:
            available = self.available()
            raise KeyError(
                f"[ragx] {self.phase}/{provider}/{name} not found. "
                f"Available: {available}"
            )
        return self._items[p][n]

    def create(self, provider: str, name: str, **kwargs: Any) -> Any:
        return self.get(provider, name)(**kwargs)

    def available(self) -> dict[str, list[str]]:
        return {p: sorted(names.keys()) for p, names in sorted(self._items.items())}

    def is_registered(self, provider: str, name: str) -> bool:
        return provider in self._items and name.lower() in self._items[provider]

    # ── Auto-discovery ────────────────────────────────────────────────────────

    def discover(self, base_path: Path, package_prefix: str) -> list[str]:
        """Scan provider directories and import all modules to fire decorators.

        Returns list of successfully loaded module paths.
        """
        loaded = []
        if not base_path.exists():
            return loaded
        for provider_dir in sorted(base_path.iterdir()):
            if not provider_dir.is_dir() or provider_dir.name.startswith("_"):
                continue
            for py_file in sorted(provider_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                module = f"{package_prefix}.{provider_dir.name}.{py_file.stem}"
                try:
                    importlib.import_module(module)
                    loaded.append(module)
                except Exception:
                    pass
        return loaded
