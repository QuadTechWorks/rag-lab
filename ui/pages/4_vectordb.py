"""Phase 4 — Vector DB."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import requests

API = "http://localhost:8000/api"
_T  = 60


def _raise(r: requests.Response):
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()


st.set_page_config(page_title="RAGx — Vector DB", layout="wide")
st.title("Phase 4 — Vector DB")
st.caption("Index vectors and run semantic search across 8 vector store providers.")

# ── Sidebar: API health ───────────────────────────────────────────────────────
with st.sidebar:
    try:
        h = requests.get("http://localhost:8000/health", timeout=5).json()
        if h.get("status") == "ok":
            st.success("API online")
        else:
            st.error("API offline"); st.stop()
    except Exception:
        st.error("API offline — run `make run-api`"); st.stop()

# ── Load providers ─────────────────────────────────────────────────────────────
try:
    vdb_providers: dict  = _raise(requests.get(f"{API}/providers/vectordb",  timeout=_T))
    embed_providers: dict = _raise(requests.get(f"{API}/providers/embedding", timeout=_T))
except Exception as exc:
    st.error(f"Failed to load providers: {exc}"); st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_connect, tab_index, tab_search = st.tabs(["Connect", "Index", "Search"])

# ── Connect Tab ────────────────────────────────────────────────────────────────
with tab_connect:
    st.subheader("Connect to a Vector Store")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        vdb_prov_names = list(vdb_providers.keys())
        vdb_prov  = st.selectbox("Provider", vdb_prov_names, key="vdb_prov")
        store_names = vdb_providers.get(vdb_prov, [])
        vdb_store = st.selectbox("Store", store_names, key="vdb_store") if store_names else None

        _HINTS = {
            ("chroma",   "local"):      "persist_dir (default: ./ragx_chroma)",
            ("qdrant",   "local"):      "path (default: ./ragx_qdrant)",
            ("qdrant",   "cloud"):      "Requires QDRANT_URL + QDRANT_API_KEY env vars",
            ("faiss",    "flat"):       "No config needed — in-memory, exact cosine",
            ("faiss",    "ivf"):        "nlist (default: 100), nprobe (default: 10)",
            ("pinecone", "serverless"): "Requires PINECONE_API_KEY env var",
            ("weaviate", "cloud"):      "Requires WEAVIATE_URL + WEAVIATE_API_KEY env vars",
            ("lancedb",  "local"):      "path (default: ./ragx_lance)",
            ("pgvector", "postgres"):   "Requires RAGX_PGVECTOR_URL env var",
            ("milvus",   "local"):      "path (default: ./ragx_milvus.db)",
        }
        if vdb_prov and vdb_store:
            hint = _HINTS.get((vdb_prov, vdb_store), "")
            if hint:
                st.caption(f"ℹ️  {hint}")

        config_json = st.text_area("Extra config (JSON)", value="{}", height=80,
                                   help='Override store defaults: {"path": "./custom_dir"}')

        if st.button("Connect", type="primary", use_container_width=True,
                     disabled=vdb_store is None):
            try:
                cfg = json.loads(config_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON in config")
            else:
                with st.spinner("Connecting…"):
                    try:
                        result = _raise(requests.post(f"{API}/vectordb/connect", timeout=30, json={
                            "provider": vdb_prov, "store": vdb_store, "config": cfg,
                        }))
                        st.success(f"Connected: {result.get('store_name', '')}")
                    except Exception as exc:
                        st.error(str(exc))

    with col_right:
        st.subheader("Active Store")
        try:
            status = _raise(requests.get(f"{API}/vectordb/status", timeout=_T))
        except Exception:
            status = {}
        if status.get("connected"):
            st.success(status.get("store_name", "connected"))
            st.json({k: v for k, v in status.items() if k != "connected"})
            if st.button("Clear Store Vectors"):
                try:
                    _raise(requests.delete(f"{API}/vectordb/vectors", timeout=_T))
                    st.success("Store cleared")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("No store connected yet. Choose a provider above and click Connect.")

# ── Index Tab ──────────────────────────────────────────────────────────────────
with tab_index:
    st.subheader("Index Embeddings into Vector Store")

    try:
        estats = _raise(requests.get(f"{API}/embed/stats", timeout=_T))
        total_vecs = estats.get("count", 0)
    except Exception:
        total_vecs = 0

    st.info(f"{total_vecs} vectors in the embedding store (Phase 3 output)")

    if total_vecs == 0:
        st.warning("No embeddings yet — complete Phase 3 (Embedding) first.")
    else:
        if st.button("Index All Vectors", type="primary", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    result = _raise(requests.post(f"{API}/vectordb/index", timeout=300,
                                                  json={"doc_ids": None}))
                    st.success(f"Indexed {result.get('indexed', 0)} vectors "
                               f"(total in store: {result.get('total', 0)})")
                    st.json(result)
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    st.subheader("Store Stats")
    try:
        db_stats = _raise(requests.get(f"{API}/vectordb/stats", timeout=_T))
        s1, s2 = st.columns(2)
        s1.metric("Vectors in Store", db_stats.get("count", 0))
        s2.metric("Vector Dim", db_stats.get("vector_dim", "—"))
        st.json(db_stats)
    except Exception as exc:
        st.caption(f"Stats unavailable: {exc}")

# ── Search Tab ─────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Semantic Search")

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        query_text = st.text_area("Query", placeholder="What is the main topic discussed?",
                                  height=80, key="search_query")
    with scol2:
        top_k = st.slider("Top K", 1, 20, 5)
        ep_names = list(embed_providers.keys())
        ep = st.selectbox("Embed provider", ep_names, key="s_ep") if ep_names else "ollama"
        en_list = embed_providers.get(ep, [])
        en = st.selectbox("Embedder", en_list, key="s_en") if en_list else "nomic"

    if st.button("Search", type="primary", use_container_width=True,
                 disabled=not (query_text or "").strip()):
        with st.spinner("Embedding query + searching…"):
            try:
                resp = _raise(requests.post(f"{API}/vectordb/search-text", timeout=60, json={
                    "query": query_text.strip(), "top_k": top_k,
                    "embed_provider": ep, "embed_name": en,
                }))
                st.session_state["search_results"] = resp
            except Exception as exc:
                st.error(str(exc))

    if "search_results" in st.session_state:
        resp = st.session_state["search_results"]
        results = resp.get("results", [])
        if results:
            st.caption(
                f"Query embedded with **{resp.get('embed_model', '')}** "
                f"({resp.get('vector_dim', 0)}-dim) · {len(results)} results"
            )
            for r in results:
                score_pct = f"{r['score'] * 100:.1f}%"
                with st.expander(
                    f"#{r['rank']} · {score_pct} · {r['source'].split('/')[-1]}"
                ):
                    st.write(r["content"])
                    meta_clean = {k: v for k, v in r["metadata"].items() if v}
                    if meta_clean:
                        st.json(meta_clean)
        else:
            st.info("No results found.")
