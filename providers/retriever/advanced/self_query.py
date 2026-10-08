from __future__ import annotations
import json
import re
from core.registry import RETRIEVERS
from core.models.embeddings import EmbeddedChunk
from providers.retriever.advanced._base import AdvancedBase

_PROMPT = """You convert a search request into a semantic query plus metadata filters.

Filterable metadata fields and their allowed values:
{schema}

Request: {query}

Reply with JSON only: {{"query": "<the request without the filter conditions>", \
"filters": {{"<field>": "<value>"}}}}
Only use listed fields and values; use {{}} for filters if none apply."""


@RETRIEVERS.register(provider="advanced", name="self_query")
class SelfQueryRetriever(AdvancedBase):
    """Self-query: the LLM splits the request into a semantic query and metadata
    filters (e.g. "papers from 2023 about X" -> year=2023), then runs the base
    retriever with those filters.

    The filter schema is learned from the indexed corpus: fields with at most
    `max_values` distinct values. Filters outside that schema are discarded, and
    explicit request filters always win. Exact-match filters only.
    Config: max_values (default 20), plus the AdvancedBase options.
    """
    variant = "self_query"

    def __init__(self, max_values: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._max_values = max_values
        self._schema: dict[str, list[str]] = {}

    def index(self, corpus: list[EmbeddedChunk]) -> None:
        super().index(corpus)
        values: dict[str, set[str]] = {}
        for c in corpus:
            for k, v in c.metadata.items():
                if v not in (None, ""):
                    values.setdefault(k, set()).add(str(v))
        self._schema = {k: sorted(v) for k, v in values.items()
                        if 1 < len(v) <= self._max_values}

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        parsed_query, inferred = query, {}
        if self._schema:
            schema = "\n".join(f"- {k}: {vals}" for k, vals in self._schema.items())
            parsed_query, inferred = self._parse(
                query, self._complete(_PROMPT.format(schema=schema, query=query)))
        merged = {**inferred, **(filters or {})}
        results = self._base.retrieve(parsed_query, top_k=top_k, filters=merged or None)
        for r in results:
            r.metadata.update({"rewritten_query": parsed_query, "inferred_filters": inferred})
            r.provider, r.retriever = "advanced", self.variant
        return results

    def _parse(self, query: str, raw: str) -> tuple[str, dict]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            return query, {}
        filters = {k: str(v) for k, v in (data.get("filters") or {}).items()
                   if k in self._schema and str(v) in self._schema[k]}
        new_query = data.get("query")
        return (new_query.strip() if isinstance(new_query, str) and new_query.strip()
                else query), filters
