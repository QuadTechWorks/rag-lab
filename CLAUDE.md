# RAGx — Multi-Provider RAG Experimentation Platform

**Project root:** `/Users/gokulraj/Documents/Pipeline/ragx`
**GitHub:** `https://github.com/QuadTechWorks/rag-lab` (main branch, 3 commits, author: tharaniesp)
**Purpose:** Experiment with every technique at every stage of a RAG pipeline. Every phase: plug-and-play providers via `@REGISTRY.register`, Streamlit UI, FastAPI routes.

---

## Pipeline Status

```
Ingestion → Chunking → Embedding → Vector DB → Retriever → Reranker → Inference → Evaluation
    ✅          ✅          ✅          ✅          🟡 Tier 1   🔲          🔲          🔲
```

---

## How to Run

```bash
make install     # pip3 install -r requirements.txt (system Python 3.14.6, --break-system-packages, no venv)
make run-api     # FastAPI on :8000  (PYTHONPATH=. uvicorn api.main:app --reload)
make run-ui      # Streamlit on :8501 (PYTHONPATH=. streamlit run ui/app.py)
make run         # both together
```

API docs: http://localhost:8000/docs | Health: http://localhost:8000/health
Kill port: `lsof -ti :8501 | xargs kill -9`

**Ollama required for default embedding:** `ollama pull nomic-embed-text` (768-dim, default)

---

## Directory Structure (current)

```
ragx/
├── api/
│   ├── main.py                    # FastAPI app, provider discovery on startup
│   ├── store.py                   # In-memory document store (Phase 1)
│   ├── chunk_store.py             # In-memory chunk store (Phase 2)
│   ├── embedding_store.py         # In-memory embedding store (Phase 3)
│   ├── vector_store_manager.py    # Active vector store singleton (Phase 4)
│   └── routes/
│       ├── providers.py           # GET /api/providers/{phase}
│       ├── ingestion.py           # Phase 1 routes
│       ├── chunking.py            # Phase 2 routes
│       ├── embedding.py           # Phase 3 routes
│       └── vectordb.py            # Phase 4 routes
│
├── core/
│   ├── registry/
│   │   ├── __init__.py            # LOADERS, CHUNKERS, EMBEDDERS, VECTOR_STORES singletons
│   │   └── provider_registry.py   # ProviderRegistry — @register decorator + discover()
│   ├── interfaces/
│   │   ├── base_loader.py         # BaseLoader ABC
│   │   ├── base_chunker.py        # BaseChunker ABC
│   │   ├── base_embedder.py       # BaseEmbedder ABC
│   │   └── base_vector_store.py   # BaseVectorStore ABC
│   ├── models/
│   │   ├── documents.py           # LoadedDocument
│   │   ├── chunks.py              # DocumentChunk
│   │   ├── embeddings.py          # EmbeddedChunk
│   │   └── search.py              # SearchResult
│   ├── routing/
│   │   └── file_router.py         # FileTypeRouter — single best loader per extension
│   └── tracing/
│       └── langfuse.py            # Optional Langfuse tracer (NoOpTracer when unconfigured)
│
├── providers/
│   ├── ingestion/                  # 44 loaders: langchain/ llamaindex/ haystack/ custom/
│   ├── chunking/                   # 17 chunkers: langchain/ llamaindex/ haystack/ custom/
│   ├── embedding/                  # 8 embedders: ollama/ litellm/
│   └── vectordb/                   # 10 stores: chroma/ qdrant/ faiss/ pinecone/ weaviate/ lancedb/ pgvector/ milvus/
│
├── ui/
│   ├── app.py
│   ├── api_client.py              # HTTP helpers — NOT imported by pages (Streamlit caching bug)
│   └── pages/
│       ├── 1_ingestion.py         # uses direct requests calls
│       ├── 2_chunking.py          # uses direct requests calls
│       ├── 3_embedding.py         # uses direct requests calls
│       └── 4_vectordb.py          # uses direct requests calls
│
├── .env.example
├── .gitignore
├── LICENSE                        # Apache 2.0 — QuadTechWorks
├── Makefile
├── README.md
└── requirements.txt
```

---

## Plug-and-Play Pattern (same for every phase)

```python
# providers/{phase}/{provider}/{name}.py
from core.registry import LOADERS   # or CHUNKERS, EMBEDDERS, VECTOR_STORES
from core.interfaces.base_loader import BaseLoader

@LOADERS.register(provider="myprovider", name="myformat")
class MyLoader(BaseLoader):
    @property
    def supported_types(self): return [".xyz"]
    def load(self, source): ...
```

Drop file → restart API → appears in UI dropdown. No other wiring needed.

| Phase | Registry | Base class | Provider dir |
|---|---|---|---|
| Ingestion | `LOADERS` | `BaseLoader` | `providers/ingestion/` |
| Chunking | `CHUNKERS` | `BaseChunker` | `providers/chunking/` |
| Embedding | `EMBEDDERS` | `BaseEmbedder` | `providers/embedding/` |
| Vector DB | `VECTOR_STORES` | `BaseVectorStore` | `providers/vectordb/` |
| Retriever (next) | `RETRIEVERS` | `BaseRetriever` | `providers/retriever/` |

---

## Phase 1 — Ingestion ✅ (44 loaders)

| Provider | Count | Formats |
|---|---|---|
| langchain | 12 | csv, docx, eml, html, ipynb, pdf, pptx, txt, unstructured, web, xlsx, xml |
| llamaindex | 15 | csv, docx, epub, html, ipynb, json, markdown, pdf, pymupdf, simple, txt, unstructured, web, xlsx, xml |
| haystack | 10 | csv, docx, html, json, markdown, pdf, pptx, tika, txt, unstructured |
| custom | 7 | code, epub, json, jsonl, markdown, rtf, yaml |

`FileTypeRouter` maps each extension to exactly one `(provider, loader)` — no fallback chains. Files with unknown extension are skipped with an explicit reason.

Routes: `POST /api/ingest/upload|url|file-path|folder/scan|folder/load`, `GET|DELETE /api/ingest/documents`

---

## Phase 2 — Chunking ✅ (17 chunkers)

| Provider | Chunkers |
|---|---|
| langchain | recursive, character, token, markdown, html |
| llamaindex | sentence, token, semantic, markdown, code, hierarchical |
| haystack | word, sentence, passage |
| custom | sliding_window, paragraph, sentence_boundary |

`ChunkStore` in-memory singleton (keyed by chunk UUID). Routes: `POST /api/chunk/preview|run`, `GET|DELETE /api/chunk/chunks`

---

## Phase 3 — Embedding ✅ (8 embedders)

| Provider | Name | Model | Dim | Requires |
|---|---|---|---|---|
| ollama | nomic | nomic-embed-text | 768 | Ollama local (default) |
| ollama | mxbai | mxbai-embed-large | 1024 | Ollama local |
| ollama | minilm | all-minilm | 384 | Ollama local |
| litellm | openai | text-embedding-3-small | 1536 | OPENAI_API_KEY |
| litellm | cohere | embed-english-v3.0 | 1024 | COHERE_API_KEY |
| litellm | huggingface | BAAI/bge-small-en-v1.5 | 384 | HUGGINGFACE_API_KEY |
| litellm | gemini | text-embedding-004 | 768 | GEMINI_API_KEY |
| litellm | mistral | mistral-embed | 1024 | MISTRAL_API_KEY |

`EmbeddingStore` in-memory singleton. Optional Langfuse tracing (`LANGFUSE_PUBLIC_KEY` env var).
Routes: `POST /api/embed/preview|run`, `GET|DELETE /api/embed/vectors|stats`

**Streamlit module caching bug:** Pages must use direct `requests` calls — never `from ui import api_client`. The `api_client.py` module is stale-cached for the session lifetime.

---

## Phase 4 — Vector DB ✅ (10 stores)

| Provider | Store | Type | Config |
|---|---|---|---|
| chroma | local | Persistent | `./ragx_chroma` |
| qdrant | local | Persistent | `./ragx_qdrant` |
| qdrant | cloud | Cloud | `QDRANT_URL` + `QDRANT_API_KEY` |
| faiss | flat | Exact cosine, in-memory | none |
| faiss | ivf | Approx ANN, in-memory | none (trains at 100 vectors) |
| pinecone | serverless | Cloud | `PINECONE_API_KEY` |
| weaviate | cloud | Cloud v4 | `WEAVIATE_URL` + `WEAVIATE_API_KEY` |
| lancedb | local | Columnar | `./ragx_lance` |
| pgvector | postgres | PostgreSQL | `RAGX_PGVECTOR_URL` |
| milvus | local | Milvus Lite | `./ragx_milvus.db` |

`VectorStoreManager` singleton holds one active store at a time; `POST /api/vectordb/connect` switches it.
Routes: `POST /api/vectordb/connect|index|search|search-text`, `GET|DELETE /api/vectordb/stats|vectors`

**Important implementation notes:**
- ChromaDB collection named `ragx_{dim}` to avoid dimension mismatch across embedders
- Qdrant point IDs: `abs(hash(chunk_id)) % (2**63)` (requires integer IDs)
- Weaviate uses deterministic UUIDs: `uuid.uuid5(NAMESPACE_DNS, chunk_id)`
- FAISS: L2-normalize vectors before IndexFlatIP for cosine via inner product
- FAISS IVF: buffers vectors until `len >= nlist=100` before training; linear scan until trained

---

## Phase 5 — Retriever 🔲 (NEXT — 13 retrievers planned)

### Files to create

```
core/interfaces/base_retriever.py      # BaseRetriever ABC: retrieve(query, k, **kwargs) -> list[RetrievalResult]
core/models/retrieval.py               # RetrievalResult: rank, score, chunk_id, content, source, metadata, retriever, strategy
api/retrieval_store.py                 # In-memory RetrievalStore singleton
api/routes/retriever.py                # POST /api/retrieve/run, GET /api/retrieve/results, etc.
ui/pages/5_retriever.py               # direct requests, no api_client import
```

Add to `core/registry/__init__.py`: `RETRIEVERS = ProviderRegistry("retriever")`
Add discover + router in `api/main.py`.

### Retriever Inventory (13 total)

**Tier 1 — Core (no LLM needed)**

| Provider | Name | Strategy |
|---|---|---|
| vector | dense | Wraps Phase 4 vectordb search-text endpoint |
| bm25 | okapi | BM25Okapi via `rank_bm25` library |
| bm25 | bm25l | BM25L — better for shorter docs |
| bm25 | bm25plus | BM25+ — handles zero TF |
| tfidf | sklearn | TF-IDF via scikit-learn |
| mmr | cosine | Maximal Marginal Relevance — relevant + diverse |

**Tier 2 — Hybrid Fusion (no LLM needed)**

| Provider | Name | Strategy |
|---|---|---|
| hybrid | rrf | Reciprocal Rank Fusion (dense + BM25 ranked lists) |
| hybrid | linear | α × dense_score + (1-α) × bm25_score |
| ensemble | rrf | Run any N retrievers, merge with RRF |

**Tier 3 — LLM-Augmented**

| Provider | Name | Strategy |
|---|---|---|
| advanced | hyde | HyDE: LLM generates hypothetical answer → embed → search |
| advanced | multi_query | LLM generates 3–5 query variants → union results |
| advanced | self_query | LLM extracts metadata filters → pre-filter → vector search |
| advanced | step_back | LLM abstracts query → search at higher level |

**Build order:** Tier 1 + Tier 2 first (no LLM dependency). Tier 3 after Phase 7 (Inference).

---

## Phases 6–8 — Roadmap

| Phase | Techniques |
|---|---|
| **6 — Reranker** | Cohere Rerank API, CrossEncoder (HuggingFace), Flashrank (local), Identity (no-op) |
| **7 — Inference** | OpenAI GPT-4o, Anthropic Claude, Groq, Google Gemini, Mistral |
| **8 — Evaluation** | RAGAS (faithfulness, relevancy, context precision, recall), experiment comparison table |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No venv | System Python 3.14.6 macOS; `--break-system-packages` |
| Single FastAPI router per phase | Hot-reload double-registration bug with split routers |
| Direct `requests` in UI pages | Streamlit caches imported modules — api_client becomes stale |
| Separate EmbeddingStore / VectorStoreManager | Phase 3 in-memory ≠ Phase 4 persistent — separate by design |
| ChromaDB collection `ragx_{dim}` | Prevents dimension mismatch when switching embedders |
| FileTypeRouter: one entry per extension | Deterministic, no ambiguity; skip > silent downgrade |
| `api/routes/folder.py` | Dead file — routes merged into ingestion.py; safe to delete |

---

## Environment Variables

```bash
# Phase 3 — Embedding (cloud)
OPENAI_API_KEY=
COHERE_API_KEY=
HUGGINGFACE_API_KEY=
GEMINI_API_KEY=
MISTRAL_API_KEY=

# Phase 4 — Vector DB (cloud stores)
PINECONE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
WEAVIATE_URL=
WEAVIATE_API_KEY=
RAGX_PGVECTOR_URL=          # postgresql://user:pass@host:5432/db

# Phase 3 — Tracing (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=

# Phase 7 — Inference
ANTHROPIC_API_KEY=
GROQ_API_KEY=
# (reuses OPENAI_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY from above)
```

---

## Git

```bash
cd /Users/gokulraj/Documents/Pipeline/ragx
git config user.name "tharaniesp"
git config user.email "tharaniesp@users.noreply.github.com"
# Remote uses PAT for auth — rotate token after use; never commit token to repo
```


---

## Phase 5 — Retriever (Tier 1 complete: 6 retrievers)

| Retriever | Needs | Config |
|---|---|---|
| `vector/dense` | vector store + embedder | — |
| `bm25/okapi`, `bm25/bm25l`, `bm25/bm25plus` | embedded chunks | `k1`, `b` |
| `tfidf/sklearn` | embedded chunks | `ngram_max`, `stop_words` |
| `mmr/cosine` | vector store + embedder | `lambda_mult`, `fetch_k` |

- Files: `core/interfaces/base_retriever.py`, `core/models/retrieval.py`, `core/retrieval_utils.py`, `api/retrieval_store.py`, `api/routes/retriever.py`, `providers/retriever/*`, `ui/pages/5_retriever.py`.
- API: `POST /api/retrieve/run`, `GET/DELETE /api/retrieve/results`, `GET /api/providers/retriever`.
- Lifecycle: `RETRIEVERS.create(...)` -> `index(corpus)` -> `retrieve(query, top_k, filters)`. Sparse retrievers use `embedding_store` as corpus (no vector DB needed); dense/MMR get `vector_store` + `embedder` injected by the route.
- Sparse retrievers drop chunks with no query-token overlap. Index is rebuilt per request.
- Next: Tier 2 hybrid (`hybrid/rrf`, `hybrid/linear`, `ensemble/rrf`), then Tier 3 after Phase 7.
