from __future__ import annotations
import re
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="markdown")
class CustomMarkdownLoader(BaseLoader):
    """Load Markdown files. Parses YAML frontmatter into metadata."""

    @property
    def supported_types(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, source: str) -> list[LoadedDocument]:
        with open(source, encoding="utf-8") as f:
            raw = f.read()

        metadata: dict = {"source": source}
        content = raw

        # Strip YAML frontmatter if present
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if fm_match:
            try:
                import yaml
                metadata.update(yaml.safe_load(fm_match.group(1)) or {})
            except Exception:
                pass
            content = raw[fm_match.end():]

        return [
            LoadedDocument(
                content=content.strip(),
                metadata=metadata,
                source=source,
                provider="custom",
                loader="markdown",
            )
        ]
