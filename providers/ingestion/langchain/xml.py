from __future__ import annotations
import xml.etree.ElementTree as ET
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="langchain", name="xml")
class LangChainXMLLoader(BaseLoader):
    """Load XML files — extracts all text nodes recursively."""

    @property
    def supported_types(self) -> list[str]:
        return [".xml"]

    def load(self, source: str) -> list[LoadedDocument]:
        tree = ET.parse(source)
        root = tree.getroot()

        def extract_text(elem: ET.Element, depth: int = 0) -> list[str]:
            lines: list[str] = []
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            text = (elem.text or "").strip()
            if text:
                lines.append(f"{'  ' * depth}<{tag}>: {text}")
            for child in elem:
                lines.extend(extract_text(child, depth + 1))
            return lines

        content = "\n".join(extract_text(root))
        return [LoadedDocument(
            content=content,
            metadata={"root_tag": root.tag, "source": source},
            source=source,
            provider="langchain",
            loader="xml",
        )]
