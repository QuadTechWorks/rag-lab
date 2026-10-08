from __future__ import annotations
from core.registry import RETRIEVERS
from core.fusion import rrf_scores
from providers.retriever.hybrid._base import FusionBase


@RETRIEVERS.register(provider="ensemble", name="rrf")
class EnsembleRRFRetriever(FusionBase):
    """Any N retrievers merged with (optionally weighted) Reciprocal Rank Fusion.

    Config: retrievers (required, >= 2): ["bm25/okapi", "tfidf/sklearn", ...] or
            [{"name": "mmr/cosine", "config": {...}, "weight": 2.0}, ...]
            rrf_k (default 60), fetch_k
    """
    provider, variant = "ensemble", "rrf"

    def __init__(self, rrf_k: int = 60, **kwargs):
        self._rrf_k = rrf_k
        super().__init__(**kwargs)

    def _child_specs(self, kwargs):
        raw = kwargs.pop("retrievers", None) or []
        if len(raw) < 2:
            raise ValueError('ensemble/rrf needs "retrievers": a list of at least 2')
        specs = []
        for item in raw:
            if isinstance(item, str):
                specs.append((item, {}, 1.0))
            else:
                specs.append((item["name"], item.get("config", {}),
                              float(item.get("weight", 1.0))))
        return specs

    def _fuse(self, ranked, weights):
        return rrf_scores([[r.chunk_id for r in rs] for rs in ranked],
                          weights=weights, k=self._rrf_k)
