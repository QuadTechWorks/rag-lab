# RAG Lab

> **Multi-provider RAG Experimentation Platform** — swap every component of your Retrieval-Augmented Generation pipeline in seconds, all from a single UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Phases](https://img.shields.io/badge/Phases%20Complete-1%20%E2%80%93%204%20of%208-brightgreen)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

---

## What is RAG Lab?

RAG Lab is a plug-and-play experimentation platform that lets you mix and match every layer of a RAG pipeline — from file parsing to semantic search — without changing a line of application code. Each phase exposes a dropdown in the UI: pick a provider, configure it, and run. Compare results across frameworks side by side.

Built to answer questions like:
- *Does LlamaIndex's SentenceSplitter outperform LangChain's RecursiveCharacterTextSplitter on my corpus?*
- *Is FAISS fast enough, or do I need Qdrant for production?*
- *Does Cohere Rerank improve retrieval over a vanilla cosine search?*

---

## Pipeline Architecture

```
┌─────────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐
│  Ingestion  │──▶│ Chunking │──▶│ Embedding │──▶│ Vector DB │
│     ✅      │   │    ✅    │   │    ✅     │   │    ✅     │
└─────────────┘   └──────────┘   └───────────┘   └───────────┘
       ┌─────────────┐   ┌──────────┐   ┌─────────────┐   ┌────────────┐
  ──▶  │  Retriever  │──▶│ Reranker │──▶│  Inference  │──▶│ Evaluation │
       │     🔲      │   │    🔲    │   │     🔲      │   │     🔲     │
       └─────────────┘   └──────────┘   └─────────────┘   └────────────┘
```

Each phase supports **all major frameworks** via the same provider registry pattern. Drop a `.py` file in the right directory → it auto-registers on the next API restart.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally (for default embedding model)

```bash
ollama pull nomic-embed-text   # default embedding model (768-dim)
```

### Install & Run

```bash
git clone https://github.com/QuadTechWorks/rag-lab.git
cd rag-lab

make install     # pip install -r requirements.txt

make run-api     # FastAPI backend  → http://localhost:8000
make run-ui      # Streamlit UI     → http://localhost:8501

# Or both together:
make run
```

> API docs: http://localhost:8000/docs  
> Health check: http://localhost:8000/health

### Environment Variables

Copy `.env.example` to `.env` and fill in only the keys you need. Phases 1 and 2 require **no API keys** — they run entirely locally.

```bash
cp .env.example .env
```

| Variable | Phase | Required for |
|---|---|---|
| `OPENAI_API_KEY` | 3 | litellm/openai embedder |
| `COHERE_API_KEY` | 3, 6 | litellm/cohere embedder + reranker |
| `HUGGINGFACE_API_KEY` | 3 | litellm/huggingface embedder |
| `GEMINI_API_KEY` | 3, 7 | litellm/gemini embedder + inference |
| `MISTRAL_API_KEY` | 3, 7 | litellm/mistral embedder + inference |
| `PINECONE_API_KEY` | 4 | pinecone/serverless vector store |
| `QDRANT_URL` + `QDRANT_API_KEY` | 4 | qdrant/cloud vector store |
| `WEAVIATE_URL` + `WEAVIATE_API_KEY` | 4 | weaviate/cloud vector store |
| `RAGX_PGVECTOR_URL` | 4 | pgvector/postgres vector store |
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | 3 | Langfuse tracing (optional) |
| `ANTHROPIC_API_KEY` | 7 | Anthropic Claude inference |
| `GROQ_API_KEY` | 7 | Groq inference |

---

## Phase 1 — Ingestion ✅

Load documents from files, URLs, or entire folder trees. **44 loaders** across 4 providers, auto-routed per file extension.

### Loader Inventory

| Provider | Count | Formats |
|---|---|---|
| **LangChain** | 12 | csv, docx, eml, html, notebook (.ipynb), pdf, pptx, txt, unstructured, web, xlsx, xml |
| **LlamaIndex** | 15 | csv, docx, epub, html, ipynb, json, markdown, pdf, pymupdf, simple, txt, unstructured, web, xlsx, xml |
| **Haystack** | 10 | csv, docx, html, json, markdown, pdf, pptx, tika, txt, unstructured |
| **Custom** | 7 | code (.py/.js/.ts/…), epub, json, jsonl, markdown, rtf, yaml |

**Notable loaders:**
- `llamaindex/pymupdf` — best PDF extraction, handles tables and images
- `llamaindex/ipynb` — Jupyter notebooks, one document per cell
- `haystack/tika` — Apache Tika, parses 100+ formats (requires Java)
- `*/unstructured` — wraps [unstructured.io](https://unstructured.io), install separately

### UI Tabs
| Tab | What it does |
|---|---|
| File Upload | Choose provider + loader, drag-and-drop file |
| URL | Fetch any web page with chosen provider + loader |
| Folder Path | Enter a directory path; scans recursively, shows routing plan, loads with per-file progress bar |

### API Routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/ingest/upload` | Upload file (multipart) |
| `POST` | `/api/ingest/url` | Fetch URL |
| `POST` | `/api/ingest/file-path` | Load server-side file path |
| `POST` | `/api/ingest/folder/scan` | Preview folder routing plan |
| `POST` | `/api/ingest/folder/load` | Bulk load a folder |
| `GET` | `/api/ingest/documents` | List loaded documents |
| `DELETE` | `/api/ingest/documents` | Clear all documents |
| `DELETE` | `/api/ingest/documents/{id}` | Delete one document |

---

## Phase 2 — Chunking ✅

Split loaded documents into chunks before embedding. **17 chunkers** across 4 providers with configurable `chunk_size` and `chunk_overlap`.

### Chunker Inventory

| Provider | Chunkers |
|---|---|
| **LangChain** | recursive, character, token (tiktoken), markdown (header-aware), html (header-aware) |
| **LlamaIndex** | sentence, token, semantic (embedding-based), markdown, code (tree-sitter), hierarchical |
| **Haystack** | word, sentence, passage |
| **Custom** | sliding_window, paragraph, sentence_boundary |

### API Routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/chunk/preview` | Preview chunks for one doc (no storage) |
| `POST` | `/api/chunk/run` | Chunk all/selected docs, store results |
| `GET` | `/api/chunk/chunks` | List chunks (filter by doc, paginate) |
| `DELETE` | `/api/chunk/chunks` | Clear all chunks |
| `DELETE` | `/api/chunk/chunks/{doc_id}` | Delete chunks for one document |

---

## Phase 3 — Embedding ✅

Embed chunks into dense vectors. **8 embedders** — Ollama runs locally with no API key; LiteLLM routes to any cloud provider. Optional [Langfuse](https://langfuse.com) tracing on every embed call.

### Embedder Inventory

| Provider | Name | Model | Dimensions | Requires |
|---|---|---|---|---|
| **ollama** | nomic | nomic-embed-text | 768 | Ollama running locally |
| **ollama** | mxbai | mxbai-embed-large | 1024 | Ollama running locally |
| **ollama** | minilm | all-minilm | 384 | Ollama running locally |
| **litellm** | openai | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| **litellm** | cohere | embed-english-v3.0 | 1024 | `COHERE_API_KEY` |
| **litellm** | huggingface | BAAI/bge-small-en-v1.5 | 384 | `HUGGINGFACE_API_KEY` |
| **litellm** | gemini | text-embedding-004 | 768 | `GEMINI_API_KEY` |
| **litellm** | mistral | mistral-embed | 1024 | `MISTRAL_API_KEY` |

### API Routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/embed/preview` | Embed one chunk, return vector preview |
| `POST` | `/api/embed/run` | Embed all/selected chunks, store results |
| `GET` | `/api/embed/vectors` | List embedded chunks |
| `GET` | `/api/embed/stats` | Store statistics |
| `DELETE` | `/api/embed/vectors` | Clear embedding store |

---

## Phase 4 — Vector DB ✅

Index embedded vectors and run semantic search. **10 stores** across 8 providers. Connect to any store with `POST /api/vectordb/connect`, then index and search against it.

### Vector Store Inventory

| Provider | Store | Type | Setup |
|---|---|---|---|
| **chroma** | local | Persistent local | No config — files in `./ragx_chroma` |
| **qdrant** | local | Persistent local | Files in `./ragx_qdrant` |
| **qdrant** | cloud | Qdrant Cloud | `QDRANT_URL` + `QDRANT_API_KEY` |
| **faiss** | flat | Exact cosine, in-memory | No config needed |
| **faiss** | ivf | Approximate ANN, in-memory | No config needed |
| **pinecone** | serverless | Pinecone Cloud | `PINECONE_API_KEY` |
| **weaviate** | cloud | Weaviate Cloud v4 | `WEAVIATE_URL` + `WEAVIATE_API_KEY` |
| **lancedb** | local | Columnar local | Files in `./ragx_lance` |
| **pgvector** | postgres | PostgreSQL + pgvector | `RAGX_PGVECTOR_URL` |
| **milvus** | local | Milvus Lite (no server) | Files in `./ragx_milvus.db` |

### API Routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/vectordb/connect` | Activate a vector store |
| `GET` | `/api/vectordb/status` | Active store connection info |
| `POST` | `/api/vectordb/index` | Index embedded chunks |
| `POST` | `/api/vectordb/search` | Search with pre-computed vector |
| `POST` | `/api/vectordb/search-text` | Search with text query (auto-embed + search) |
| `GET` | `/api/vectordb/stats` | Store statistics |
| `DELETE` | `/api/vectordb/vectors` | Clear all vectors |

---

## Phases 5–8 — Roadmap 🔲

| Phase | Description | Techniques |
|---|---|---|
| **5 — Retriever** | Retrieve relevant chunks | Vector (dense), BM25 (keyword), Hybrid (RRF), HyDE, Multi-query, MMR |
| **6 — Reranker** | Re-score retrieved results | Cohere Rerank, CrossEncoder, Flashrank (local), Identity (no-op) |
| **7 — Inference** | Generate answers from context | OpenAI GPT-4o, Anthropic Claude, Groq, Google Gemini, Mistral |
| **8 — Evaluation** | Measure pipeline quality | RAGAS (faithfulness, relevancy, precision, recall), experiment comparison table |

---

## Directory Structure

```
rag-lab/
├── api/
│   ├── main.py                    # FastAPI app — provider discovery on startup
│   ├── store.py                   # In-memory document store
│   ├── chunk_store.py             # In-memory chunk store
│   ├── embedding_store.py         # In-memory embedding store
│   ├── vector_store_manager.py    # Active vector store singleton
│   └── routes/
│       ├── providers.py           # GET /api/providers/*
│       ├── ingestion.py           # Phase 1 routes
│       ├── chunking.py            # Phase 2 routes
│       ├── embedding.py           # Phase 3 routes
│       └── vectordb.py            # Phase 4 routes
│
├── core/
│   ├── registry/
│   │   └── provider_registry.py  # @register decorator + discover()
│   ├── interfaces/
│   │   ├── base_loader.py        # BaseLoader ABC
│   │   ├── base_chunker.py       # BaseChunker ABC
│   │   ├── base_embedder.py      # BaseEmbedder ABC
│   │   └── base_vector_store.py  # BaseVectorStore ABC
│   ├── models/
│   │   ├── documents.py          # LoadedDocument
│   │   ├── chunks.py             # DocumentChunk
│   │   ├── embeddings.py         # EmbeddedChunk
│   │   └── search.py             # SearchResult
│   ├── routing/
│   │   └── file_router.py        # FileTypeRouter — single best loader per extension
│   └── tracing/
│       └── langfuse.py           # Optional Langfuse tracer (no-op when unconfigured)
│
├── providers/
│   ├── ingestion/                 # 44 loaders: langchain/ llamaindex/ haystack/ custom/
│   ├── chunking/                  # 17 chunkers: langchain/ llamaindex/ haystack/ custom/
│   ├── embedding/                 # 8 embedders: ollama/ litellm/
│   └── vectordb/                  # 10 stores: chroma/ qdrant/ faiss/ pinecone/
│                                  #             weaviate/ lancedb/ pgvector/ milvus/
│
├── ui/
│   ├── app.py                     # Streamlit entry point
│   ├── api_client.py              # HTTP client for all API calls
│   └── pages/
│       ├── 1_ingestion.py
│       ├── 2_chunking.py
│       ├── 3_embedding.py
│       └── 4_vectordb.py
│
├── .env.example
├── Makefile
└── requirements.txt
```

---

## Adding a Custom Provider

Every phase uses the same decorator pattern. Drop a file in the right directory and it auto-registers on the next API restart — no wiring required.

### Example: Add a new loader

```python
# providers/ingestion/myprovider/myformat.py

from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument

@LOADERS.register(provider="myprovider", name="myformat")
class MyFormatLoader(BaseLoader):

    @property
    def supported_types(self) -> list[str]:
        return [".xyz"]

    def load(self, source: str) -> list[LoadedDocument]:
        content = open(source).read()           # parse your format
        return [LoadedDocument(
            content=content,
            metadata={},
            source=source,
            provider="myprovider",
            loader="myformat",
        )]
```

The same pattern applies to every phase:

| Phase | Registry | Base class | Provider dir |
|---|---|---|---|
| Ingestion | `LOADERS` | `BaseLoader` | `providers/ingestion/` |
| Chunking | `CHUNKERS` | `BaseChunker` | `providers/chunking/` |
| Embedding | `EMBEDDERS` | `BaseEmbedder` | `providers/embedding/` |
| Vector DB | `VECTOR_STORES` | `BaseVectorStore` | `providers/vectordb/` |

---

## Optional Dependencies

Install only what you need:

```bash
# Heavy loaders (requires system deps)
pip install "unstructured[pdf,docx,pptx]"
brew install libmagic poppler tesseract     # macOS

# Tika (requires Java)
pip install tika

# LlamaIndex semantic chunker
pip install llama-index-embeddings-huggingface sentence-transformers

# LlamaIndex code chunker (syntax-aware)
pip install tree-sitter tree-sitter-languages

# Optional vector stores
pip install lancedb pyarrow                 # LanceDB
pip install "weaviate-client>=4.0.0"        # Weaviate Cloud
pip install psycopg2-binary pgvector        # pgvector (PostgreSQL)
pip install "pymilvus>=2.4.0"              # Milvus Lite

# Observability
pip install langfuse                        # LLM tracing
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Data models | Pydantic v2 |
| Local embeddings | Ollama |
| Cloud embeddings | LiteLLM (100+ providers) |
| Tracing | Langfuse (optional) |
| Frameworks | LangChain, LlamaIndex, Haystack |

---

## License

Apache License 2.0 © [QuadTechWorks](https://github.com/QuadTechWorks)

See [LICENSE](./LICENSE) for the full license text.
