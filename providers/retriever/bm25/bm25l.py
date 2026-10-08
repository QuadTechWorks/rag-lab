from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.bm25._base import BM25Base


@RETRIEVERS.register(provider="bm25", name="bm25l")
class BM25LRetriever(BM25Base):
    """BM25 L — rank_bm25.BM25L over the embedded chunks."""
    variant = "bm25l"
    _bm25_cls_name = "BM25L"
