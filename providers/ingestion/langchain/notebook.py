from __future__ import annotations
import json
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="notebook")
class LangChainNotebookLoader(BaseLoader):
    """Load Jupyter notebooks (.ipynb) — extracts code cells and markdown cells."""

    @property
    def supported_types(self) -> list[str]:
        return [".ipynb"]

    def load(self, source: str) -> list[LoadedDocument]:
        with open(source, encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        parts: list[str] = []
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            if cell_type == "markdown":
                parts.append(f"[markdown]\n{src}")
            elif cell_type == "code":
                parts.append(f"[code]\n{src}")
                # Include text outputs
                for output in cell.get("outputs", []):
                    text = "".join(output.get("text", []))
                    if text.strip():
                        parts.append(f"[output]\n{text}")

        content = "\n\n---\n\n".join(parts)
        kernel = nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "")

        return [LoadedDocument(
            content=content,
            metadata={"kernel": kernel, "cell_count": len(cells), "source": source},
            source=source,
            provider="langchain",
            loader="notebook",
        )]
