from .provider_registry import ProviderRegistry

# One registry per pipeline phase — add here as each phase is built.
LOADERS       = ProviderRegistry("ingestion")   # Phase 1
CHUNKERS      = ProviderRegistry("chunking")    # Phase 2
EMBEDDERS     = ProviderRegistry("embedding")   # Phase 3
VECTOR_STORES = ProviderRegistry("vectordb")    # Phase 4
RETRIEVERS    = ProviderRegistry("retriever")   # Phase 5
# RERANKERS   = ProviderRegistry("reranker")    # Phase 6
LLMS          = ProviderRegistry("llm")         # Phase 7 (minimal layer added for Tier 3 retrievers)
# EVALUATORS  = ProviderRegistry("evaluator")   # Phase 8
