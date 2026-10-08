from __future__ import annotations
from core.registry import RETRIEVERS
from core.fusion import rrf_scores
from providers.retriever.hybrid._base import FusionBase, two_way_specs


@RETRIEVERS.register(provider="hybrid", name="rrf")
class HybridRRFRetriever(FusionBase):
    """Dense + sparse fused with Reciprocal Rank Fusion (rank-based, scale-free).

    Config: dense (default "vector/dense"), sparse (default "bm25/okapi"),
            dense_config, sparse_config, rrf_k (default 60), fetch_k
    """
    provider, variant = "hybrid", "rrf"

    def __init__(self, rrf_k: int = 60, **kwargs):
        self._rrf_k = rrf_k
        super().__init__(**kwargs)

    def _child_specs(self, kwargs):
        return two_way_specs(kwargs)

    def _fuse(self, ranked, weights):
        return rrf_scores([[r.chunk_id for r in rs] for rs in ranked], k=self._rrf_k)
