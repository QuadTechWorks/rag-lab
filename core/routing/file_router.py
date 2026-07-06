"""FileTypeRouter — maps every file extension to its single best loader.

No fallback chains. Each extension has exactly one (provider, loader) that is
the authoritative best choice for that format. If that loader is not installed,
the file is marked skipped with a clear reason — never silently downgraded.
"""
from __future__ import annotations
from pathlib import Path

# Extension → single best (provider, loader_name) — no fallback lists
FILE_TYPE_MAP: dict[str, tuple[str, str]] = {
    # Documents
    ".pdf":      ("langchain", "pdf"),       # full text + metadata extraction
    ".docx":     ("langchain", "docx"),      # preserves structure
    ".doc":      ("langchain", "unstructured"),
    ".odt":      ("langchain", "unstructured"),
    ".rtf":      ("custom",    "rtf"),       # striprtf — clean RTF text
    ".epub":     ("custom",    "epub"),      # ebooklib — chapter-aware

    # Spreadsheets
    ".xlsx":     ("langchain", "xlsx"),      # openpyxl — row-per-doc
    ".xls":      ("langchain", "xlsx"),
    ".ods":      ("langchain", "unstructured"),

    # Presentations
    ".pptx":     ("langchain", "pptx"),      # slide-per-doc
    ".ppt":      ("langchain", "unstructured"),

    # Plain text / markup
    ".txt":      ("langchain", "txt"),
    ".md":       ("custom",    "markdown"),  # heading-aware splitter
    ".markdown": ("custom",    "markdown"),
    ".rst":      ("langchain", "txt"),

    # Web / HTML
    ".html":     ("langchain", "html"),      # bs4 — strips tags
    ".htm":      ("langchain", "html"),
    ".xml":      ("langchain", "xml"),

    # Data
    ".csv":      ("langchain", "csv"),       # row-per-doc
    ".json":     ("custom",    "json"),      # pretty-printed content
    ".jsonl":    ("custom",    "jsonl"),     # line-per-doc
    ".ndjson":   ("custom",    "jsonl"),
    ".yaml":     ("custom",    "yaml"),
    ".yml":      ("custom",    "yaml"),
    ".toml":     ("custom",    "code"),

    # Email
    ".eml":      ("langchain", "eml"),
    ".msg":      ("langchain", "unstructured"),

    # Notebooks
    ".ipynb":    ("langchain", "notebook"),  # cell-per-doc

    # Code — language-detected, preserves structure
    ".py":    ("custom", "code"),  ".js":    ("custom", "code"),
    ".ts":    ("custom", "code"),  ".jsx":   ("custom", "code"),
    ".tsx":   ("custom", "code"),  ".java":  ("custom", "code"),
    ".go":    ("custom", "code"),  ".rs":    ("custom", "code"),
    ".cpp":   ("custom", "code"),  ".c":     ("custom", "code"),
    ".h":     ("custom", "code"),  ".cs":    ("custom", "code"),
    ".php":   ("custom", "code"),  ".rb":    ("custom", "code"),
    ".swift": ("custom", "code"),  ".kt":    ("custom", "code"),
    ".scala": ("custom", "code"),  ".r":     ("custom", "code"),
    ".sql":   ("custom", "code"),  ".sh":    ("custom", "code"),
    ".bash":  ("custom", "code"),  ".zsh":   ("custom", "code"),
    ".ini":   ("custom", "code"),  ".conf":  ("custom", "code"),
}

# Binary / media / archive types — never parsed
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj",
    ".pyc", ".pyo", ".pyd", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".DS_Store",
}


class FileTypeRouter:
    """Maps each file to its single best (provider, loader) — no silent fallbacks."""

    def __init__(self) -> None:
        self._map  = FILE_TYPE_MAP
        self._skip = SKIP_EXTENSIONS

    def route(self, file_path: str) -> tuple[str, str] | None:
        """Return the best (provider, loader_name) for this file, or None."""
        from core.registry import LOADERS

        path = Path(file_path)
        ext  = path.suffix.lower()

        if ext in self._skip or path.name.startswith("."):
            return None

        best = self._map.get(ext)
        if best is None:
            return None  # extension not in map — skip, no fallback

        provider, name = best
        if LOADERS.is_registered(provider, name):
            return (provider, name)

        return None  # best loader not installed — skip, no downgrade

    def describe(self, file_path: str) -> dict:
        """Return routing metadata for one file (used by scan preview)."""
        path  = Path(file_path)
        ext   = path.suffix.lower()

        if ext in self._skip or path.name.startswith("."):
            return {
                "file": path.name, "ext": ext,
                "provider": None, "loader": None,
                "skipped": True, "skip_reason": "Binary/media — not parsed",
            }

        best = self._map.get(ext)
        if best is None:
            return {
                "file": path.name, "ext": ext,
                "provider": None, "loader": None,
                "skipped": True,
                "skip_reason": f"No loader defined for {ext or 'no-extension'} files",
            }

        from core.registry import LOADERS
        provider, name = best
        if not LOADERS.is_registered(provider, name):
            return {
                "file": path.name, "ext": ext,
                "provider": provider, "loader": name,
                "skipped": True,
                "skip_reason": f"Loader not installed: {provider}/{name}",
            }

        return {
            "file": path.name, "ext": ext,
            "provider": provider, "loader": name,
            "skipped": False,
        }

    def scan_folder(
        self,
        folder: str,
        recursive: bool = True,
        max_file_size_mb: float = 50.0,
    ) -> list[dict]:
        """Walk folder, return routing plan for every file. No silent fallbacks."""
        root = Path(folder)
        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        max_bytes = int(max_file_size_mb * 1024 * 1024)
        pattern   = "**/*" if recursive else "*"
        results   = []

        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            size = path.stat().st_size
            info = self.describe(str(path))
            info["path"]    = str(path)
            info["size_kb"] = round(size / 1024, 1)

            if size > max_bytes and not info["skipped"]:
                info["skipped"]     = True
                info["skip_reason"] = f"File too large (>{max_file_size_mb} MB)"

            results.append(info)

        return results


file_router = FileTypeRouter()
