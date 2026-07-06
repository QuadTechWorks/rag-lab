# RAGx — Multi-Provider RAG Experimentation Platform

**Project root:** `/Users/gokulraj/Documents/Pipeline/ragx`
**Replaces:** `ragos` (deleted and redesigned from scratch)
**Purpose:** Experiment with every technique at every stage of a RAG pipeline — ingestion, chunking, embedding, vector DB, retrieval, reranking, inference, evaluation — controlled from a Streamlit UI, with every phase supporting all major providers (LangChain, LlamaIndex, Haystack, Custom).

---

## Pipeline Architecture

```
Data Ingestion → Chunking → Embedding → Vector DB → Retriever → Reranker → Inference → Evaluation
     ✅              🔲          🔲          🔲          🔲          🔲          🔲          🔲
```

Each phase: plug-and-play providers, Streamlit UI, FastAPI routes. One `ProviderRegistry` per phase.

---

## How to Run

```bash
make install     # pip3 install -r requirements.txt  (system Python, no venv)
make run-api     # FastAPI on :8000   (PYTHONPATH=. uvicorn api.main:app --reload)
make run-ui      # Streamlit on :8501 (PYTHONPATH=. streamlit run ui/app.py)
make run         # both together (api in background)
```

**No virtual environment** — system Python 3.14.6 on macOS, `--break-system-packages`.
If port 8501 is busy: `lsof -ti :8501 | xargs kill -9`

API docs: http://localhost:8000/docs
Health: http://localhost:8000/health

---

## Directory Structure

```
ragx/
├── api/
│   ├── main.py                    # FastAPI app factory, provider discovery on startup
│   ├── store.py                   # In-memory document store (dict, keyed by UUID)
│   └── routes/
│       ├── ingestion.py           # All /api/ingest/* routes (single router, no split)
│       ├── providers.py           # GET /api/providers/ingestion
│       └── folder.py              # DEAD FILE — not imported, routes merged into ingestion.py
│
├── core/
│   ├── registry/
│   │   └── provider_registry.py  # ProviderRegistry — @register decorator + discover()
│   ├── interfaces/
│   │   └── base_loader.py        # BaseLoader ABC (abstract load(), supported_types)
│   ├── models/
│   │   └── documents.py          # LoadedDocument Pydantic model
│   └── routing/
│       └── file_router.py        # FileTypeRouter — single best loader per extension
│
├── providers/
│   └── ingestion/                # Phase 1 providers (44 loaders total)
│       ├── langchain/            # 12 loaders
│       ├── llamaindex/           # 15 loaders
│       ├── haystack/             # 10 loaders
│       └── custom/               #  7 loaders
│
├── ui/
│   ├── app.py                    # Streamlit entry point
│   └── pages/
│       └── 1_ingestion.py        # Phase 1 UI (File Upload / URL / Folder Path tabs)
│
├── Makefile
└── requirements.txt
```

---

## Phase 1 — Ingestion (COMPLETE)

### Provider & Loader Inventory (44 total)

| Provider | Count | Loaders |
|---|---|---|
| **langchain** | 12 | csv, docx, eml, html, notebook, pdf, pptx, txt, unstructured, web, xlsx, xml |
| **llamaindex** | 15 | csv, docx, epub, html, ipynb, json, markdown, pdf, pymupdf, simple, txt, unstructured, web, xlsx, xml |
| **haystack** | 10 | csv, docx, html, json, markdown, pdf, pptx, tika, txt, unstructured |
| **custom** | 7 | code, epub, json, jsonl, markdown, rtf, yaml |

**Notable loaders:**
- `llamaindex/pymupdf` — best for PDFs with tables and images (PyMuPDF)
- `llamaindex/ipynb` — Jupyter notebooks, one doc per cell
- `haystack/tika` — Apache Tika, 100+ formats (requires Java)
- `*/unstructured` — wraps unstructured.io, install separately (see requirements.txt)

### FileTypeRouter — Single Best Per Extension

`core/routing/file_router.py` maps every file extension to **exactly one** `(provider, loader)` — the authoritative best choice. No fallback chains, no silent downgrade.

```python
FILE_TYPE_MAP: dict[str, tuple[str, str]] = {
    ".pdf":  ("langchain", "pdf"),
    ".docx": ("langchain", "docx"),
    ".md":   ("custom",    "markdown"),
    ".py":   ("custom",    "code"),
    # ...one entry per extension, always
}
```

If the best loader is not installed → file is **skipped** with a clear reason. Three skip categories:
1. **Binary/media** — image, video, zip, exe (never parsed)
2. **No loader defined** — extension not in the map
3. **Loader not installed** — best loader known but package missing

Used by the Folder Path tab to auto-route every file in a directory recursively.

### API Routes — `/api/ingest/*`

All routes live in `api/routes/ingestion.py` under one `APIRouter(prefix="/ingest")`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest/upload` | Upload a file (multipart), provider+loader in form data |
| `POST` | `/ingest/url` | Fetch URL with chosen provider+loader |
| `POST` | `/ingest/file-path` | Load a server-side file path (used by UI per-file progress) |
| `POST` | `/ingest/folder/scan` | Preview routing plan for a folder (no loading) |
| `POST` | `/ingest/folder/load` | Bulk load entire folder |
| `GET` | `/ingest/documents` | List all documents in the store |
| `DELETE` | `/ingest/documents` | Clear all documents |
| `DELETE` | `/ingest/documents/{id}` | Delete one document by ID |

### UI — `ui/pages/1_ingestion.py`

Three tabs:
- **File Upload** — pick provider + loader from dropdown, upload file
- **URL** — fetch web page with chosen provider + loader
- **Folder Path** — enter server path, toggle recursive scan, set max file size; Scan shows routing plan table; Load runs per-file with live progress bar

Document Store at the bottom: lists all loaded docs with content preview and per-doc delete.

### Plug-and-Play Pattern

To add a new loader:

1. Create `providers/ingestion/{provider}/{name}.py`
2. Add the `@LOADERS.register(provider="x", name="y")` decorator on a class that extends `BaseLoader`
3. Implement `load(source: str) -> list[LoadedDocument]` and `supported_types`
4. Restart API — auto-discovered on startup, appears in provider dropdown immediately

```python
from core.registry import LOADERS
from core.interfaces.base_loader import BaseLoader
from core.models.documents import LoadedDocument

@LOADERS.register(provider="myprovider", name="myformat")
class MyFormatLoader(BaseLoader):
    @property
    def supported_types(self) -> list[str]:
        return [".xyz"]

    def load(self, source: str) -> list[LoadedDocument]:
        # ... parse the file ...
        return [LoadedDocument(content=text, metadata={}, source=source,
                               provider="myprovider", loader="myformat")]
```

To also route this loader in the Folder Path tab, add the extension to `FILE_TYPE_MAP` in `core/routing/file_router.py`.

---

## Core Internals

### ProviderRegistry (`core/registry/provider_registry.py`)

One singleton `LOADERS = ProviderRegistry("ingestion")` imported by all loader files. Key methods:
- `LOADERS.register(provider, name)` — class decorator
- `LOADERS.create(provider, name)` — instantiate a loader by name
- `LOADERS.is_registered(provider, name)` — used by FileTypeRouter to check availability
- `LOADERS.available()` — returns `{provider: [loader_names]}`, used by the providers API
- `LOADERS.discover(base_path, package_prefix)` — imports all `.py` files in provider dirs to fire decorators

Discovery runs once at API startup in `api/main.py → _discover_providers()`.

### LoadedDocument (`core/models/documents.py`)

```python
class LoadedDocument(BaseModel):
    id: str          # UUID4, auto-generated
    content: str
    metadata: dict
    source: str
    provider: str
    loader: str
    char_count: int  # computed from content length
```

### In-Memory Document Store (`api/store.py`)

Simple `dict[str, LoadedDocument]` wrapped in a class. Cleared on API restart. Phases 4+ will replace this with a real vector store.

---

## Remaining Phases — Roadmap

### Phase 2 — Chunking

**API:** `POST /api/chunk/run` — takes doc IDs + chunker config, returns chunked docs
**UI:** `ui/pages/2_chunking.py` — pick chunker, configure chunk_size / chunk_overlap, live preview

| Provider | Chunkers |
|---|---|
| LangChain | RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, SentenceTransformersTokenTextSplitter |
| LlamaIndex | SentenceSplitter, TokenTextSplitter, SemanticSplitter, MarkdownNodeParser, CodeSplitter, HierarchicalNodeParser |
| Haystack | DocumentSplitter (word/sentence/passage), NLTKDocumentSplitter, SpacySentenceSplitter |
| Custom | SlidingWindow, ParagraphAware, SentenceBoundary |

### Phase 3 — Embedding

Embed chunked docs using cloud inference APIs (no local models).

| Provider | Embedders |
|---|---|
| OpenAI | text-embedding-3-small, text-embedding-3-large, ada-002 |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 |
| HuggingFace Inference API | BAAI/bge-small, sentence-transformers/all-MiniLM-L6-v2 |
| VoyageAI | voyage-3, voyage-3-lite |

### Phase 4 — Vector Database

| Provider | Notes |
|---|---|
| Pinecone | Free serverless tier |
| Qdrant | Free cloud cluster |
| Weaviate | Free sandbox |
| Chroma | Local, no account needed |

### Phase 5 — Retriever

Vector similarity, BM25 (keyword), Hybrid (RRF fusion), HyDE, Multi-query, MMR

### Phase 6 — Reranker

Cohere Rerank API, CrossEncoder (HuggingFace Inference API), Flashrank (local), Identity (no-op)

### Phase 7 — Inference

OpenAI (GPT-4o), Anthropic (Claude), Groq (free tier), Google Gemini, Mistral

### Phase 8 — Evaluation

RAGAS metrics (faithfulness, answer relevancy, context precision, context recall), experiment comparison table, best-config picker

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No virtual environment | System Python on macOS; `--break-system-packages` |
| Single FastAPI router | Hot-reload double-registration bug with multiple routers |
| No fallback in FileTypeRouter | Explicit skip > silent downgrade; user sees exact reason |
| One loader per extension in FILE_TYPE_MAP | Deterministic, no ambiguity during folder scan |
| In-memory store for Phase 1 | Real vector DB added in Phase 4; no premature abstraction |
| Cloud APIs only for embedding/inference | Avoids local GPU requirement; free tiers available |
| `api/routes/folder.py` | Dead file — routes merged into `ingestion.py`; safe to delete |

---

## Dependencies

```
fastapi, uvicorn, pydantic, streamlit, requests, pyyaml
langchain, langchain-community, pypdf, python-docx, docx2txt, beautifulsoup4
llama-index-core, llama-index-readers-file, llama-index-readers-web
pymupdf, lxml, pandas, nbformat, html2text
haystack-ai>=2.5.0
openpyxl, python-pptx, ebooklib, striprtf
```

**Optional (install manually):**
```bash
pip3 install tika                              # Haystack TikaDocumentConverter (needs Java)
pip3 install "unstructured[pdf,docx,pptx]"   # llamaindex/haystack unstructured loaders
brew install libmagic poppler tesseract       # macOS system deps for unstructured
```
