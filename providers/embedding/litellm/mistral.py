from __future__ import annotations
import os
from core.registry import EMBEDDERS
from core.interfaces.base_embedder import BaseEmbedder
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk

_MODEL = "mistral/mistral-embed"
_DIM = 1024


@EMBEDDERS.register(provider="litellm", name="mistral")
class LiteLLMMistralEmbedder(BaseEmbedder):
    """Mistral mistral-embed via LiteLLM — 1024-dim.

    Requires: MISTRAL_API_KEY env var.
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

        api_key = os.environ.get("MISTRAL_API_KEY")
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
                embedder="mistral",
                model=self._model,
            )
            for i, chunk in enumerate(chunks)
        ]
