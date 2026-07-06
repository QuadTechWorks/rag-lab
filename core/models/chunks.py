from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
import uuid


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str        # ID of the LoadedDocument this came from
    source: str               # original file path / URL
    chunk_index: int          # 0-based position within its parent document
    char_count: int = 0
    provider: str             # chunker provider
    chunker: str              # chunker name
    chunk_size: int           # configured chunk_size
    chunk_overlap: int        # configured chunk_overlap

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.content)
