from __future__ import annotations
from core.registry import RETRIEVERS
from core.fusion import minmax
from providers.retriever.hybrid._base import FusionBase, two_way_specs


@RETRIEVERS.register(provider="hybrid", name="linear")
class HybridLinearRetriever(FusionBase):
    """score = alpha * dense + (1 - alpha) * sparse, each min-max normalised to [0, 1].

    Config: alpha (default 0.5; 1.0 = dense only), dense, sparse, dense_config,
            sparse_config, fetch_k
    A chunk missing from one list contributes 0 for that side.
    """
    provider, variant = "hybrid", "linear"

    def __init__(self, alpha: float = 0.5, **kwargs):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self._alpha = alpha
        super().__init__(**kwargs)

    def _child_specs(self, kwargs):
        return two_way_specs(kwargs)

    def _fuse(self, ranked, weights):
        dense, sparse = (minmax({r.chunk_id: r.score for r in rs}) for rs in ranked)
        return {cid: self._alpha * dense.get(cid, 0.0) + (1 - self._alpha) * sparse.get(cid, 0.0)
                for cid in dense.keys() | sparse.keys()}
