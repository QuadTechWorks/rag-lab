from __future__ import annotations
import os
from core.registry import EMBEDDERS
from core.interfaces.base_embedder import BaseEmbedder
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk

_MODEL = "cohere/embed-english-v3.0"
_DIM = 1024


@EMBEDDERS.register(provider="litellm", name="cohere")
class LiteLLMCohereEmbedder(BaseEmbedder):
    """Cohere embed-english-v3.0 via LiteLLM — 1024-dim.

    Requires: COHERE_API_KEY env var.
    Alternatives: cohere/embed-multilingual-v3.0 (multilingual, same dim).
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

        api_key = os.environ.get("COHERE_API_KEY")
        texts = [c.content for c in chunks]
        response = litellm.embedding(
            model=self._model, input=texts, api_key=api_key,
            input_type="search_document",
        )
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
                embedder="cohere",
                model=self._model,
            )
            for i, chunk in enumerate(chunks)
        ]
