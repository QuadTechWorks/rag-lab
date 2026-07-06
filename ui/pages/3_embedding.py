"""Phase 3 — Embedding."""
import sys
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


st.set_page_config(page_title="RAGx — Embedding", layout="wide")
st.title("Phase 3 — Embedding")
st.caption("Embed chunked documents using Ollama (local) or cloud models via LiteLLM.")

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
    providers: dict = _raise(requests.get(f"{API}/providers/embedding", timeout=_T))
except Exception as exc:
    st.error(f"Failed to load embedding providers: {exc}"); st.stop()

provider_names = list(providers.keys())
if not provider_names:
    st.warning("No embedding providers registered."); st.stop()

# ── Load chunks ────────────────────────────────────────────────────────────────
try:
    chunk_resp = _raise(requests.get(f"{API}/chunk/chunks", params={"limit": 1000}, timeout=_T))
    all_chunks = chunk_resp.get("chunks", [])
except Exception as exc:
    st.error(f"Failed to load chunks: {exc}"); all_chunks = []

doc_ids = sorted({c["source_doc_id"] for c in all_chunks}) if all_chunks else []

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Configuration")

    embed_all = st.checkbox("Embed all chunks", value=True)
    selected_docs = None if embed_all else st.multiselect("Select source documents", doc_ids)

    provider = st.selectbox("Provider", provider_names)
    embedder_names = providers.get(provider, [])
    embedder = st.selectbox("Embedder", embedder_names) if embedder_names else None

    if provider == "litellm":
        st.info("API key read from env var (OPENAI_API_KEY, COHERE_API_KEY, etc.)")

    batch_size = st.slider("Batch size", 1, 128, 32,
                           help="Chunks per call to the embedding service")
    use_langfuse = st.checkbox("Enable Langfuse tracing",
                               help="Requires LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY env vars")

    st.divider()

    preview_chunk_id = all_chunks[0]["id"] if all_chunks else None
    if preview_chunk_id and st.button("Preview single chunk", use_container_width=True,
                                      disabled=embedder is None):
        with st.spinner("Embedding one chunk…"):
            try:
                pv = _raise(requests.post(f"{API}/embed/preview", timeout=60, json={
                    "chunk_id": preview_chunk_id, "provider": provider, "embedder": embedder,
                    "use_langfuse": use_langfuse,
                }))
                st.session_state["embed_preview"] = pv
            except Exception as exc:
                st.error(str(exc))

    chunk_count = len(all_chunks) if embed_all else sum(
        1 for c in all_chunks if c["source_doc_id"] in (selected_docs or [])
    )
    st.info(f"{chunk_count} chunks selected")

    if st.button("Run Embedding", type="primary", use_container_width=True,
                 disabled=embedder is None or chunk_count == 0):
        with st.spinner(f"Embedding {chunk_count} chunks…"):
            try:
                result = _raise(requests.post(f"{API}/embed/run", timeout=300, json={
                    "doc_ids": None if embed_all else selected_docs,
                    "provider": provider, "embedder": embedder,
                    "use_langfuse": use_langfuse, "batch_size": batch_size,
                }))
                st.session_state["embed_run"] = result
            except Exception as exc:
                st.error(str(exc))

    if st.button("Clear Embedding Store", use_container_width=True):
        try:
            r = _raise(requests.delete(f"{API}/embed/vectors", timeout=_T))
            st.success(f"Cleared {r['cleared']} vectors")
            st.session_state.pop("embed_run", None)
        except Exception as exc:
            st.error(str(exc))

with right:
    if "embed_preview" in st.session_state:
        pv = st.session_state["embed_preview"]
        st.subheader("Preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", pv.get("model", "—"))
        c2.metric("Vector Dim", pv.get("vector_dim", 0))
        c3.metric("Char Count", pv.get("char_count", 0))
        st.caption("First 10 dimensions:")
        st.code(str([round(v, 6) for v in pv.get("vector_preview", [])]))

    if "embed_run" in st.session_state:
        r = st.session_state["embed_run"]
        st.subheader("Run Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Embedded", r.get("embedded", 0))
        c2.metric("Store Total", r.get("store_total", 0))
        c3.metric("Vector Dim", r.get("vector_dim", 0))
        c4.metric("Model", r.get("model", "—"))

    st.divider()
    st.subheader("Embedding Store")
    try:
        stats = _raise(requests.get(f"{API}/embed/stats", timeout=_T))
        if stats.get("count", 0) > 0:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Vectors", stats["count"])
            sc2.metric("Vector Dim", stats.get("vector_dim", 0))
            sc3.metric("Avg Chars", stats.get("avg_chars", 0))
            if stats.get("models"):
                st.caption(f"Models: {', '.join(stats['models'])}")
            vecs = _raise(requests.get(f"{API}/embed/vectors", params={"limit": 50}, timeout=_T))
            rows = vecs.get("vectors", [])
            if rows:
                import pandas as pd
                df = pd.DataFrame([{
                    "chunk_id": v["chunk_id"][:12] + "…",
                    "source": v["source"].split("/")[-1],
                    "chars": v["char_count"],
                    "dim": v["vector_dim"],
                    "model": v["model"],
                    "v[0:3]": str([round(x, 4) for x in v["vector_preview"][:3]]),
                } for v in rows])
                st.dataframe(df, use_container_width=True, height=300)
        else:
            st.info("No embeddings yet. Run embedding above.")
    except Exception as exc:
        st.error(str(exc))
