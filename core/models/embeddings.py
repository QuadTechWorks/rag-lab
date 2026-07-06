from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
import uuid


class EmbeddedChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str           # ID of the source DocumentChunk
    source_doc_id: str      # ID of the source LoadedDocument
    source: str             # original file path / URL
    content: str            # the text that was embedded
    vector: list[float]     # the embedding vector
    vector_dim: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider: str           # embedder provider (e.g. "ollama", "litellm")
    embedder: str           # embedder name (e.g. "nomic", "openai")
    model: str              # exact model string used
    char_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.vector_dim:
            self.vector_dim = len(self.vector)
