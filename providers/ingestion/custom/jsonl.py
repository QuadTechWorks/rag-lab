from __future__ import annotations
import json
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="jsonl")
class CustomJSONLLoader(BaseLoader):
    """Load JSON Lines files (.jsonl) — one document per line."""

    @property
    def supported_types(self) -> list[str]:
        return [".jsonl", ".ndjson"]

    def load(self, source: str) -> list[LoadedDocument]:
        docs = []
        with open(source, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    content = json.dumps(obj, indent=2) if isinstance(obj, dict) else str(obj)
                except json.JSONDecodeError:
                    content = line
                docs.append(LoadedDocument(
                    content=content,
                    metadata={"line": i + 1, "source": source},
                    source=source,
                    provider="custom",
                    loader="jsonl",
                ))
        return docs
