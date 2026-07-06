"""Custom JSON loader — demonstrates how to add any new provider.

Drop a .py file in providers/ingestion/custom/ with @LOADERS.register,
implement load() — it auto-registers at startup. No other changes needed.
"""
from __future__ import annotations
import json
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="json")
class CustomJSONLoader(BaseLoader):
    """Load JSON files. Each top-level key becomes a document."""

    @property
    def supported_types(self) -> list[str]:
        return [".json"]

    def load(self, source: str) -> list[LoadedDocument]:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            items = [{"value": str(data)}]

        return [
            LoadedDocument(
                content=json.dumps(item, indent=2),
                metadata={"index": i, "source": source},
                source=source,
                provider="custom",
                loader="json",
            )
            for i, item in enumerate(items)
        ]
