from __future__ import annotations
import json as _json
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="json")
class HaystackJSONLoader(BaseLoader):
    """Load JSON files as Haystack-style documents — pretty-printed content, keys as metadata.

    Haystack's JSONConverter is an output adapter, not a file reader; this implements
    the Haystack document schema directly using stdlib json.
    """

    @property
    def supported_types(self) -> list[str]:
        return [".json", ".jsonl", ".ndjson"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            import haystack  # noqa: F401  — verify haystack installed
        except ImportError:
            raise ImportError("pip install haystack-ai")

        with open(source, encoding="utf-8") as fh:
            raw = _json.load(fh)

        if isinstance(raw, list):
            return [
                LoadedDocument(
                    content=_json.dumps(item, indent=2, ensure_ascii=False),
                    metadata={"index": i, "format": "json"},
                    source=source,
                    provider="haystack",
                    loader="json",
                )
                for i, item in enumerate(raw)
            ]

        return [
            LoadedDocument(
                content=_json.dumps(raw, indent=2, ensure_ascii=False),
                metadata={"format": "json", "keys": list(raw.keys()) if isinstance(raw, dict) else []},
                source=source,
                provider="haystack",
                loader="json",
            )
        ]
