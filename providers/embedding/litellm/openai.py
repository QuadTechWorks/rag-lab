from __future__ import annotations
import os
from core.registry import EMBEDDERS
from core.interfaces.base_embedder import BaseEmbedder
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk

_MODEL = "text-embedding-3-small"
_DIM = 1536


@EMBEDDERS.register(provider="litellm", name="openai")
class LiteLLMOpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small via LiteLLM — 1536-dim.

    Requires: OPENAI_API_KEY env var.
    Alternatives: set model to text-embedding-3-large (3072-dim) or ada-002 (1536-dim).
    """

    def __init__(self, model: str = _MODEL, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
                     "text-embedding-ada-002": 1536}.get(model, 1536)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def vector_dim(self) -> int:
        return self._dim

    def embed(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        try:
            import litellm
        except ImportError:
            raise ImportError("pip install litellm")

        api_key = os.environ.get("OPENAI_API_KEY")
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
                embedder="openai",
                model=self._model,
            )
            for i, chunk in enumerate(chunks)
        ]
