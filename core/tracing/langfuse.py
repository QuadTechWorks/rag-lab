"""Optional Langfuse tracing for embedding calls.

Activated when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY env vars are set.
Falls back to a no-op tracer silently when not configured.
"""
from __future__ import annotations
import os


class NoOpTracer:
    def start_embed(self, model: str, texts: list[str]) -> None:
        pass

    def end_embed(self, vectors: list[list[float]]) -> None:
        pass

    def flush(self) -> None:
        pass


class LangfuseTracer:
    def __init__(self) -> None:
        from langfuse import Langfuse
        self._client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        self._generation = None

    def start_embed(self, model: str, texts: list[str]) -> None:
        trace = self._client.trace(name="ragx.embedding")
        self._generation = trace.generation(
            name="embed",
            model=model,
            model_parameters={"batch_size": len(texts)},
            input={"count": len(texts), "sample": texts[:2]},
        )

    def end_embed(self, vectors: list[list[float]]) -> None:
        if self._generation:
            self._generation.end(output={
                "count": len(vectors),
                "vector_dim": len(vectors[0]) if vectors else 0,
            })
            self._generation = None

    def flush(self) -> None:
        self._client.flush()


def get_tracer(use_langfuse: bool = False) -> NoOpTracer | LangfuseTracer:
    """Return a LangfuseTracer if configured + requested, else NoOpTracer."""
    if use_langfuse and os.environ.get("LANGFUSE_PUBLIC_KEY"):
        try:
            return LangfuseTracer()
        except Exception:
            return NoOpTracer()
    return NoOpTracer()
