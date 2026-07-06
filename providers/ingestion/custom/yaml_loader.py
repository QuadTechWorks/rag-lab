from __future__ import annotations
import yaml
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument


@LOADERS.register(provider="custom", name="yaml")
class CustomYAMLLoader(BaseLoader):
    """Load YAML config/data files — renders as formatted text."""

    @property
    def supported_types(self) -> list[str]:
        return [".yaml", ".yml"]

    def load(self, source: str) -> list[LoadedDocument]:
        with open(source, encoding="utf-8") as f:
            raw = f.read()
        try:
            parsed = yaml.safe_load(raw)
            content = yaml.dump(parsed, default_flow_style=False, allow_unicode=True)
        except yaml.YAMLError:
            content = raw

        return [LoadedDocument(
            content=content,
            metadata={"source": source},
            source=source,
            provider="custom",
            loader="yaml",
        )]
