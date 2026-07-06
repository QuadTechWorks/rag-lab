from __future__ import annotations
import os
from core.registry import EMBEDDERS
from core.interfaces.base_embedder import BaseEmbedder
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk

_MODEL = "gemini/text-embedding-004"
_DIM = 768


@EMBEDDERS.register(provider="litellm", name="gemini")
class LiteLLMGeminiEmbedder(BaseEmbedder):
    """Google text-embedding-004 via LiteLLM — 768-dim.

    Requires: GEMINI_API_KEY (or GOOGLE_API_KEY) env var.
    """

    def __init__(self, model: str = _MODEL, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def vector_dim(self) -> int:
        return _DIM

    def embed(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        try:
            import litellm
        except ImportError:
            raise ImportError("pip install litellm")

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        texts = [c.content for c in chunks]
        response = litellm.embedding(model=self._model, input=texts, api_key=api_key)
        vectors = [item["embedding"] for item in response.data]

        return [
            EmbeddedChunk(
                chunk_id=chunk.id,
                source_doc_id=chunk.source_doc_id,
                source=chunk.source,
                content=chunk.content,
                vector=vectors[i],
                metadata=dict(chunk.metadata),
                provider="litellm",
                embedder="gemini",
                model=self._model,
            )
            for i, chunk in enumerate(chunks)
        ]
