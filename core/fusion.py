"""Score/rank fusion primitives for hybrid and ensemble retrievers."""
from __future__ import annotations


def rrf_scores(rankings: list[list[str]], weights: list[float] | None = None,
               k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank 1-based."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for w, ids in zip(weights, rankings):
        for rank, cid in enumerate(ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
    return fused


def minmax(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1]. A single value or all-equal values map to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
