from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
import uuid


class LoadedDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str
    provider: str
    loader: str
    char_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.content)
