from __future__ import annotations
from core.registry import RETRIEVERS
from core.fusion import rrf_scores
from core.interfaces.base_retriever import BaseRetriever
from core.models.embeddings import EmbeddedChunk
from core.models.retrieval import RetrievalResult


def build_child(key: str, config: dict, **injected) -> BaseRetriever:
    """Instantiate a registered retriever by "provider/name", passing injected deps."""
    prov, _, name = key.partition("/")
    if not RETRIEVERS.is_registered(prov, name):
        raise ValueError(f"Retriever {key!r} not registered. "
                         f"Available: {RETRIEVERS.available()}")
    return RETRIEVERS.create(prov, name, **injected, **config)


def merge_runs(runs: dict[str, list[RetrievalResult]], top_k: int, provider: str,
               variant: str, rrf_k: int = 60) -> list[RetrievalResult]:
    """RRF-merge several ranked lists (keyed by label) into one result list.

    Each result's metadata.component_ranks records the rank it had per label.
    """
    labels = list(runs)
    fused = rrf_scores([[r.chunk_id for r in runs[l]] for l in labels], k=rrf_k)
    first_seen: dict[str, RetrievalResult] = {}
    ranks: dict[str, dict[str, int]] = {}
    for label in labels:
        for r in runs[label]:
            first_seen.setdefault(r.chunk_id, r)
            ranks.setdefault(r.chunk_id, {})[label] = r.rank
    order = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
    return [
        first_seen[cid].model_copy(update={
            "rank": i + 1, "score": float(fused[cid]), "provider": provider,
            "retriever": variant,
            "metadata": {**first_seen[cid].metadata, "component_ranks": ranks[cid]},
        })
        for i, cid in enumerate(order)
    ]


class AdvancedBase(BaseRetriever):
    """LLM-augmented retriever wrapping a base retriever (default vector/dense).

    Kwargs (injected by the API): llm, vector_store, embedder.
    Config: base ("provider/name", default "vector/dense"), base_config, fetch_k.
    The base can be any retriever, e.g. "hybrid/rrf", so LLM query tricks stack
    on top of sparse, dense or hybrid retrieval.
    """
    variant: str = ""

    def __init__(self, llm=None, vector_store=None, embedder=None,
                 base: str = "vector/dense", base_config: dict | None = None,
                 fetch_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._store = vector_store
        self._embedder = embedder
        self._fetch_k = fetch_k
        self._base_key = base
        self._base = build_child(base, base_config or {}, llm=llm,
                                 vector_store=vector_store, embedder=embedder)

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        self._base.index(corpus)

    def _complete(self, prompt: str, system: str | None = None) -> str:
        if self._llm is None:
            raise RuntimeError(f"advanced/{self.variant} needs an LLM — "
                               "select one (e.g. ollama/chat) in the request")
        return self._llm.generate(prompt, system=system)

    def _fetch(self, top_k: int) -> int:
        return self._fetch_k or max(top_k * 4, 20)

    @property
    def retriever_name(self) -> str:
        model = getattr(self._llm, "model_name", "no-llm")
        return f"advanced/{self.variant} (llm={model}, base={self._base_key})"
