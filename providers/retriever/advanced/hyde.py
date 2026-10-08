from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.advanced._base import AdvancedBase
from providers.retriever.vector.dense import embed_query, to_result

_PROMPT = """Write a short passage (3-4 sentences) that would directly answer the \
question below, in the style of a document that contains the answer. It does not \
need to be factually correct.

Question: {query}"""


@RETRIEVERS.register(provider="advanced", name="hyde")
class HyDERetriever(AdvancedBase):
    """HyDE: the LLM writes a hypothetical answer; its embedding (not the question's)
    is used for dense search, matching documents by answer-likeness.

    Always dense: searches the vector store directly, so `base` is unused.
    Config: include_query (default true) — embed "query + passage" together.
    """
    variant = "hyde"

    def __init__(self, include_query: bool = True, **kwargs):
        kwargs["base"] = "vector/dense"
        super().__init__(**kwargs)
        self._include_query = include_query

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        if self._store is None or self._embedder is None:
            raise RuntimeError("advanced/hyde needs a connected vector store and an embedder")
        passage = self._complete(_PROMPT.format(query=query))
        text = f"{query}\n{passage}" if self._include_query else passage
        hits = self._store.search(embed_query(self._embedder, text), top_k=top_k,
                                  filters=filters)
        out = []
        for i, h in enumerate(hits):
            r = to_result(h, i + 1, self.variant, provider="advanced")
            r.metadata["hypothetical_passage"] = passage
            out.append(r)
        return out
