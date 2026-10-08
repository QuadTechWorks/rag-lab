from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.bm25._base import BM25Base


@RETRIEVERS.register(provider="bm25", name="bm25plus")
class BM25PlusRetriever(BM25Base):
    """BM25 Plus — rank_bm25.BM25Plus over the embedded chunks."""
    variant = "bm25plus"
    _bm25_cls_name = "BM25Plus"
