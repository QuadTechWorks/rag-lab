from __future__ import annotations
import csv as _csv
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="haystack", name="csv")
class HaystackCSVLoader(BaseLoader):
    """Load CSV files as Haystack-style documents — one document per row.

    Haystack has no native CSVToDocument; this implements the Haystack document
    schema (content + meta) using stdlib csv for row-level granularity.
    """

    @property
    def supported_types(self) -> list[str]:
        return [".csv"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            import haystack  # noqa: F401  — verify haystack installed
        except ImportError:
            raise ImportError("pip install haystack-ai")

        docs = []
        with open(source, newline="", encoding="utf-8-sig") as fh:
            reader = _csv.DictReader(fh)
            for i, row in enumerate(reader):
                content = ", ".join(f"{k}: {v}" for k, v in row.items())
                docs.append(
                    LoadedDocument(
                        content=content,
                        metadata={"row": i, "columns": list(row.keys())},
                        source=source,
                        provider="haystack",
                        loader="csv",
                    )
                )
        return docs
