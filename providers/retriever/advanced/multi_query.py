from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.advanced._base import AdvancedBase, merge_runs

_PROMPT = """Generate {n} different rewordings of the search query below. Cover different \
wordings and angles so that relevant documents phrased differently can be found.
Return one query per line, with no numbering and no extra text.

Query: {query}"""


@RETRIEVERS.register(provider="advanced", name="multi_query")
class MultiQueryRetriever(AdvancedBase):
    """Query expansion: the LLM writes n rewordings; each runs through the base
    retriever and the lists (plus the original query) are RRF-merged.

    Config: n_queries (default 3), plus the AdvancedBase options.
    """
    variant = "multi_query"

    def __init__(self, n_queries: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._n = n_queries

    def expand(self, query: str) -> list[str]:
        raw = self._complete(_PROMPT.format(n=self._n, query=query))
        lines = [l.strip(" -•*\t0123456789.)").strip() for l in raw.splitlines()]
        seen = {query.lower()}
        out = []
        for l in lines:
            if l and l.lower() not in seen:
                seen.add(l.lower()); out.append(l)
        return out[: self._n]

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        queries = [query, *self.expand(query)]
        runs = {f"q{i}": self._base.retrieve(q, top_k=self._fetch(top_k), filters=filters)
                for i, q in enumerate(queries)}
        results = merge_runs(runs, top_k, "advanced", self.variant)
        for r in results:
            r.metadata["queries"] = queries
        return results
