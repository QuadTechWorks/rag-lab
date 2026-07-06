"""Code file loader — reads source files of any programming language.

Supports: .py .js .ts .jsx .tsx .java .go .rs .cpp .c .h .cs .php .rb
          .swift .kt .scala .r .sql .sh .bash .zsh .toml .ini .conf
"""
from __future__ import annotations
from pathlib import Path
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument

LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".cpp": "cpp", ".c": "c", ".h": "c",
    ".cs": "csharp", ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".r": "r", ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".toml": "toml", ".ini": "ini", ".conf": "conf",
}


@LOADERS.register(provider="custom", name="code")
class CustomCodeLoader(BaseLoader):
    """Load source code files with language detection from extension."""

    @property
    def supported_types(self) -> list[str]:
        return list(LANG_MAP.keys())

    def load(self, source: str) -> list[LoadedDocument]:
        path = Path(source)
        ext = path.suffix.lower()
        language = LANG_MAP.get(ext, "text")

        content = path.read_text(encoding="utf-8", errors="replace")

        return [LoadedDocument(
            content=content,
            metadata={
                "language": language,
                "extension": ext,
                "filename": path.name,
                "source": source,
            },
            source=source,
            provider="custom",
            loader="code",
        )]
