from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.bm25._base import BM25Base


@RETRIEVERS.register(provider="bm25", name="okapi")
class BM25OkapiRetriever(BM25Base):
    """BM25 Okapi — rank_bm25.BM25Okapi over the embedded chunks."""
    variant = "okapi"
    _bm25_cls_name = "BM25Okapi"
