from __future__ import annotations
from core.registry import RETRIEVERS
from providers.retriever.advanced._base import AdvancedBase, merge_runs

_PROMPT = """Rewrite the question below as a more general "step-back" question that asks \
about the broader concept, principle or background needed to answer it.
Return only the step-back question.

Question: {query}"""


@RETRIEVERS.register(provider="advanced", name="step_back")
class StepBackRetriever(AdvancedBase):
    """Step-back prompting: retrieve for the original question and for an
    LLM-abstracted, broader version of it, then RRF-merge both lists.

    Config: AdvancedBase options only.
    """
    variant = "step_back"

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None):
        step_back = self._complete(_PROMPT.format(query=query)).splitlines()[0].strip()
        k = self._fetch(top_k)
        runs = {"original": self._base.retrieve(query, top_k=k, filters=filters)}
        if step_back and step_back.lower() != query.lower():
            runs["step_back"] = self._base.retrieve(step_back, top_k=k, filters=filters)
        results = merge_runs(runs, top_k, "advanced", self.variant)
        for r in results:
            r.metadata["step_back_query"] = step_back
        return results
