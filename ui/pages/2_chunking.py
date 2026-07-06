"""Phase 2 — Chunking.

Takes loaded documents from Phase 1, splits them into chunks using the selected
provider and technique. Supports preview before committing to the chunk store.
"""
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
        raise Exception(detail)


def _get(path: str, **kw):
    r = requests.get(f"{API}{path}", timeout=_T, **kw)
    _raise(r)
    return r.json()


def _post(path: str, **kw):
    r = requests.post(f"{API}{path}", timeout=120, **kw)
    _raise(r)
    return r.json()


def _delete(path: str):
    r = requests.delete(f"{API}{path}", timeout=_T)
    _raise(r)
    return r.json()


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chunking — RAGx", layout="wide")
st.title("Phase 2 — Chunking")
st.caption(
    "Split loaded documents into chunks · Compare chunking strategies across "
    "LangChain, LlamaIndex, Haystack, and Custom providers"
)

try:
    h = requests.get("http://localhost:8000/health", timeout=4).json()
    assert h.get("status") == "ok"
except Exception:
    st.error("API unreachable — run `make run-api` first.")
    st.stop()

# ── Fetch providers and documents ─────────────────────────────────────────────
try:
    chunkers: dict = _get("/providers/chunking")
except Exception as exc:
    st.error(f"Cannot fetch chunking providers: {exc}")
    st.stop()

try:
    doc_store = _get("/ingest/documents")
    all_docs  = doc_store.get("documents", [])
except Exception as exc:
    st.error(f"Cannot fetch documents: {exc}")
    st.stop()

if not all_docs:
    st.warning("No documents loaded yet. Go to **Phase 1 — Ingestion** first.")
    st.stop()

provider_names = list(chunkers.keys())

# ── Layout ────────────────────────────────────────────────────────────────────
col_cfg, col_out = st.columns([1, 2], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Configuration
# ════════════════════════════════════════════════════════════════════════════
with col_cfg:
    st.subheader("Configure")

    # Document selection
    st.markdown("**Source documents**")
    use_all = st.toggle("Use all documents", value=True)
    selected_ids: list[str] | None = None
    if not use_all:
        doc_options = {
            (
                d.get("metadata", {}).get("filename")
                or Path(d.get("source", "")).name
                or d["id"][:8]
            ): d["id"]
            for d in all_docs
        }
        selected_names = st.multiselect(
            "Select documents",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),
        )
        selected_ids = [doc_options[n] for n in selected_names] or None

    n_docs = len(all_docs) if use_all else (len(selected_ids) if selected_ids else 0)
    st.caption(f"{n_docs} document(s) selected · "
               f"{sum(d['char_count'] for d in all_docs):,} total chars")

    st.divider()

    # Provider + chunker
    st.markdown("**Chunker**")
    sel_prov    = st.selectbox("Provider", provider_names, key="ch_prov")
    sel_chunker = st.selectbox("Chunker",  chunkers.get(sel_prov, []), key="ch_name")

    # Chunker-aware parameter hints
    UNIT_HINT = {
        "token":     "tokens  (~4 chars each)",
        "sentence":  "sentences",
        "passage":   "passages (\\n\\n blocks)",
        "word":      "words",
        "markdown":  "ignored — splits at headings",
        "html":      "ignored — splits at <h1-h4> tags",
        "code":      "lines of code",
        "hierarchical": "chars (leaf level only)",
    }
    hint = UNIT_HINT.get(sel_chunker, "characters")
    st.caption(f"chunk_size unit: **{hint}**")

    chunk_size    = st.slider("Chunk size",    32, 2048, 512, step=32)
    chunk_overlap = st.slider("Chunk overlap", 0,  512,  50,  step=10)

    if chunk_overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
        st.stop()

    st.divider()

    # Preview doc selector
    st.markdown("**Preview target**")
    preview_doc_options = {
        (
            d.get("metadata", {}).get("filename")
            or Path(d.get("source", "")).name
            or d["id"][:8]
        ): d["id"]
        for d in (all_docs if use_all else
                  [d for d in all_docs if d["id"] in (selected_ids or [])])
    }
    preview_doc_name = st.selectbox("Preview document", list(preview_doc_options.keys()))
    preview_doc_id   = preview_doc_options[preview_doc_name]

    btn_preview = st.button("🔍 Preview", use_container_width=True)
    btn_chunk   = st.button("⚡ Chunk & Store", use_container_width=True,
                             type="primary", disabled=(n_docs == 0))

# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Preview & Results
# ════════════════════════════════════════════════════════════════════════════
with col_out:

    # ── PREVIEW ──────────────────────────────────────────────────────────────
    if btn_preview:
        with st.spinner(f"Previewing with **{sel_prov} / {sel_chunker}**…"):
            try:
                pv = _post("/chunk/preview", json={
                    "doc_id":       preview_doc_id,
                    "provider":     sel_prov,
                    "chunker":      sel_chunker,
                    "chunk_size":   chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "max_preview":  5,
                })
                st.session_state["chunk_preview"] = pv
            except Exception as exc:
                st.error(f"Preview failed: {exc}")

    if "chunk_preview" in st.session_state:
        pv = st.session_state["chunk_preview"]
        st.subheader("Preview")

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated chunks", pv["total_chunks"])
        m2.metric("Avg chars / chunk", pv["avg_chars"])
        m3.metric("Source doc",
                  Path(pv.get("doc_source", "")).name or pv.get("doc_source", ""))

        for i, chunk in enumerate(pv["preview"]):
            label = (f"Chunk {chunk['chunk_index']} "
                     f"· {chunk['char_count']} chars")
            with st.expander(label, expanded=(i == 0)):
                st.code(chunk["content"][:800] +
                        ("…" if len(chunk["content"]) > 800 else ""),
                        language=None)
                if chunk.get("metadata"):
                    st.json(chunk["metadata"], expanded=False)

    # ── CHUNK & STORE ─────────────────────────────────────────────────────────
    if btn_chunk:
        with st.spinner(f"Chunking {n_docs} doc(s) with **{sel_prov} / {sel_chunker}**…"):
            try:
                result = _post("/chunk/run", json={
                    "doc_ids":      selected_ids,
                    "provider":     sel_prov,
                    "chunker":      sel_chunker,
                    "chunk_size":   chunk_size,
                    "chunk_overlap": chunk_overlap,
                })
                st.session_state["chunk_result"] = result
            except Exception as exc:
                st.error(f"Chunking failed: {exc}")

    if "chunk_result" in st.session_state:
        res = st.session_state["chunk_result"]
        st.subheader("Result")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Docs processed", res["docs_processed"])
        m2.metric("Total chunks",   res["total_chunks"])
        m3.metric("Avg chars",      res.get("stats", {}).get("avg_chars", "—"))
        m4.metric("Errors",         res["docs_failed"])

        if res["docs_failed"] == 0:
            st.success(
                f"✅  **{res['total_chunks']}** chunks stored  "
                f"·  **{sel_prov} / {sel_chunker}**  "
                f"·  size={chunk_size}  overlap={chunk_overlap}"
            )
        else:
            st.warning(
                f"{res['docs_processed']} OK · {res['docs_failed']} errors"
            )
            with st.expander("Errors"):
                for err in res.get("errors", []):
                    st.error(f"`{Path(err['source']).name}` — {err['error']}")

        if res.get("results"):
            table = [{"File": Path(r["source"]).name,
                      "Status": r["status"], "Chunks": r["chunks"]}
                     for r in res["results"]]
            st.dataframe(table, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# Chunk Store (bottom, shared)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Chunk Store")

col_r, col_c, _ = st.columns([1, 1, 5])
if col_r.button("Refresh", use_container_width=True):
    st.session_state.pop("chunk_result", None)
    st.session_state.pop("chunk_preview", None)
    st.rerun()
if col_c.button("Clear All Chunks", use_container_width=True):
    try:
        _delete("/chunk/chunks")
        st.session_state.pop("chunk_result", None)
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

try:
    store = _get("/chunk/chunks", params={"limit": 500})
except Exception as exc:
    st.error(str(exc))
    st.stop()

stats  = store.get("stats", {})
chunks = store.get("chunks", [])

if stats.get("count", 0) > 0:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total chunks", stats["count"])
    s2.metric("Avg chars",    stats["avg_chars"])
    s3.metric("Min chars",    stats["min_chars"])
    s4.metric("Max chars",    stats["max_chars"])

st.caption(f"**{len(chunks)}** chunk(s) in store")

# Group by source doc for display
by_doc: dict[str, list] = {}
for c in chunks:
    by_doc.setdefault(c["source_doc_id"], []).append(c)

for doc_id, doc_chunks in by_doc.items():
    sample = doc_chunks[0]
    fname  = (Path(sample["source"]).name or doc_id[:8])
    header = (f"📄 {fname}  ·  {len(doc_chunks)} chunks  "
              f"·  `{sample['provider']} / {sample['chunker']}`  "
              f"·  size={sample['chunk_size']}  overlap={sample['chunk_overlap']}")
    with st.expander(header):
        for chunk in doc_chunks[:10]:
            st.markdown(
                f"**Chunk {chunk['chunk_index']}** "
                f"({chunk['char_count']} chars)"
            )
            st.code(chunk["content"][:400] +
                    ("…" if len(chunk["content"]) > 400 else ""),
                    language=None)
        if len(doc_chunks) > 10:
            st.caption(f"… {len(doc_chunks) - 10} more chunks not shown")
        if st.button("Delete chunks for this doc", key=f"del_{doc_id}"):
            try:
                _delete(f"/chunk/chunks/{doc_id}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
