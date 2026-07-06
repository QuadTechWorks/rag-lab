from __future__ import annotations
from core.registry import EMBEDDERS
from core.interfaces.base_embedder import BaseEmbedder
from core.models.chunks import DocumentChunk
from core.models.embeddings import EmbeddedChunk

_MODEL = "nomic-embed-text"
_DIM = 768


@EMBEDDERS.register(provider="ollama", name="nomic")
class OllamaNomicEmbedder(BaseEmbedder):
    """nomic-embed-text via local Ollama — 768-dim, best general-purpose default.

    Requires: ollama running locally with nomic-embed-text pulled.
      ollama pull nomic-embed-text
    """

    @property
    def model_name(self) -> str:
        return _MODEL

    @property
    def vector_dim(self) -> int:
        return _DIM

    def embed(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")

        texts = [c.content for c in chunks]
        vectors = _batch_embed(ollama, _MODEL, texts)

        return [
            EmbeddedChunk(
                chunk_id=chunk.id,
                source_doc_id=chunk.source_doc_id,
                source=chunk.source,
                content=chunk.content,
                vector=vectors[i],
                metadata=dict(chunk.metadata),
                provider="ollama",
                embedder="nomic",
                model=_MODEL,
            )
            for i, chunk in enumerate(chunks)
        ]


def _batch_embed(ollama_mod, model: str, texts: list[str]) -> list[list[float]]:
    try:
        response = ollama_mod.embed(model=model, input=texts)
        return [list(v) for v in response.embeddings]
    except AttributeError:
        # Older ollama client — single-text API
        return [list(ollama_mod.embeddings(model=model, prompt=t)["embedding"]) for t in texts]
