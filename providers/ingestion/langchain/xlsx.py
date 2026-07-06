from __future__ import annotations
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="xlsx")
class LangChainExcelLoader(BaseLoader):
    """Load Excel files (.xlsx/.xls) — one document per sheet."""

    @property
    def supported_types(self) -> list[str]:
        return [".xlsx", ".xls"]

    def load(self, source: str) -> list[LoadedDocument]:
        try:
            import openpyxl
        except ImportError:
            raise ImportError("pip install openpyxl")

        wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
        docs = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cleaned = [str(c) if c is not None else "" for c in row]
                if any(cleaned):
                    rows.append("\t".join(cleaned))
            content = "\n".join(rows)
            if content.strip():
                docs.append(LoadedDocument(
                    content=content,
                    metadata={"sheet": sheet_name, "source": source},
                    source=source,
                    provider="langchain",
                    loader="xlsx",
                ))
        wb.close()
        return docs
