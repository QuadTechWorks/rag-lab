from __future__ import annotations
from core.registry import RETRIEVERS
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult


class FusionBase(BaseRetriever):
    """Runs several child retrievers and fuses their rankings.

    Children are given as "provider/name" strings or
    {"name": "provider/name", "config": {...}, "weight": 1.0}.
    `vector_store` / `embedder` / `llm` (injected by the API) are passed to every child.
    Each child is asked for `fetch_k` results (default max(4*top_k, 20)).
    Subclasses implement `_fuse(ranked_lists, weights) -> {chunk_id: score}`.
    """
    provider: str = ""
    variant: str = ""

    def __init__(self, vector_store=None, embedder=None, llm=None,
                 fetch_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._fetch_k = fetch_k
        self._specs = self._child_specs(kwargs)
        self._children: list[tuple[str, float, BaseRetriever]] = []
        for key, cfg, weight in self._specs:
            prov, name = key.split("/")
            if not RETRIEVERS.is_registered(prov, name):
                raise ValueError(f"Child retriever {key!r} not registered. "
                                 f"Available: {RETRIEVERS.available()}")
            child = RETRIEVERS.create(prov, name, vector_store=vector_store,
                                      embedder=embedder, llm=llm, **cfg)
            self._children.append((key, weight, child))

    def _child_specs(self, kwargs: dict) -> list[tuple[str, dict, float]]:
        raise NotImplementedError

    def _fuse(self, ranked: list[list[RetrievalResult]],
              weights: list[float]) -> dict[str, float]:
        raise NotImplementedError

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        for _, _, child in self._children:
            child.index(corpus)

    def retrieve(self, query: str, top_k: int = 5,
                 filters: dict | None = None) -> list[RetrievalResult]:
        fetch_k = self._fetch_k or max(top_k * 4, 20)
        ranked = [child.retrieve(query, top_k=fetch_k, filters=filters)
                  for _, _, child in self._children]
        fused = self._fuse(ranked, [w for _, w, _ in self._children])

        first_seen: dict[str, RetrievalResult] = {}
        child_ranks: dict[str, dict[str, int]] = {}
        for (key, _, _), results in zip(self._children, ranked):
            for r in results:
                first_seen.setdefault(r.chunk_id, r)
                child_ranks.setdefault(r.chunk_id, {})[key] = r.rank

        order = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [
            first_seen[cid].model_copy(update={
                "rank": i + 1, "score": float(fused[cid]),
                "provider": self.provider, "retriever": self.variant,
                "metadata": {**first_seen[cid].metadata,
                             "component_ranks": child_ranks[cid]},
            })
            for i, cid in enumerate(order)
        ]

    @property
    def retriever_name(self) -> str:
        parts = ", ".join(k for k, _, _ in self._children)
        return f"{self.provider}/{self.variant} [{parts}]"


def two_way_specs(kwargs: dict, dense_default="vector/dense", sparse_default="bm25/okapi"):
    """Child specs for the dense + sparse hybrids; kwargs may override either side."""
    dense = kwargs.pop("dense", dense_default)
    sparse = kwargs.pop("sparse", sparse_default)
    return [(dense, kwargs.pop("dense_config", {}), 1.0),
            (sparse, kwargs.pop("sparse_config", {}), 1.0)]
