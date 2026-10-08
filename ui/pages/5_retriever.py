"""Phase 5 — Retriever."""
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


st.set_page_config(page_title="RAGx — Retriever", layout="wide")
st.title("Phase 5 — Retriever")
st.caption("Run one or several retrievers on the same query and compare results side by side.")

with st.sidebar:
    try:
        h = requests.get("http://localhost:8000/health", timeout=5).json()
        if h.get("status") == "ok":
            st.success("API online")
        else:
            st.error("API offline"); st.stop()
    except Exception:
        st.error("API offline — run `make run-api`"); st.stop()

try:
    providers: dict = _raise(requests.get(f"{API}/providers/retriever", timeout=_T))
    embed_providers: dict = _raise(requests.get(f"{API}/providers/embedding", timeout=_T))
    estats = _raise(requests.get(f"{API}/embed/stats", timeout=_T))
    vstatus = _raise(requests.get(f"{API}/vectordb/status", timeout=_T))
except Exception as exc:
    st.error(f"Failed to load: {exc}"); st.stop()

n_chunks = estats.get("count", 0)
c1, c2 = st.columns(2)
c1.metric("Embedded chunks (corpus)", n_chunks)
c2.metric("Vector store", vstatus.get("store_name", "not connected")
          if vstatus.get("connected") else "not connected")

_HINTS = {
    ("vector", "dense"):    "Needs a connected vector store (Phase 4) + embedder",
    ("mmr", "cosine"):      'Needs vector store. Config: {"lambda_mult": 0.5, "fetch_k": 20}',
    ("bm25", "okapi"):      'Config: {"k1": 1.5, "b": 0.75}',
    ("bm25", "bm25l"):      'Config: {"k1": 1.5, "b": 0.75}',
    ("bm25", "bm25plus"):   'Config: {"k1": 1.5, "b": 0.75}',
    ("tfidf", "sklearn"):   'Config: {"ngram_max": 2, "stop_words": "english"}',
}
all_keys = [f"{p}/{n}" for p, names in providers.items() for n in names]

tab_run, tab_hist = st.tabs(["Run & Compare", "History"])

with tab_run:
    query = st.text_area("Query", placeholder="What is the main topic discussed?",
                         height=80, key="r_query")
    left, right = st.columns([2, 1])
    with left:
        selected = st.multiselect("Retrievers", all_keys,
                                  default=[k for k in ("bm25/okapi", "vector/dense")
                                           if k in all_keys])
        config_json = st.text_area(
            "Per-retriever config (JSON, keyed by provider/name)", value="{}", height=80,
            help='e.g. {"bm25/okapi": {"k1": 1.2}, "mmr/cosine": {"lambda_mult": 0.7}}')
        for k in selected:
            hint = _HINTS.get(tuple(k.split("/")), "")
            if hint:
                st.caption(f"ℹ️  **{k}** — {hint}")
    with right:
        top_k = st.slider("Top K", 1, 20, 5)
        ep = st.selectbox("Embed provider", list(embed_providers), key="r_ep")
        en = st.selectbox("Embedder", embed_providers.get(ep, []), key="r_en")
        st.caption("Embedder is used only by vector/dense and mmr/cosine. "
                   "It must match the one used for indexing.")

    if st.button("Run", type="primary", use_container_width=True,
                 disabled=not ((query or "").strip() and selected)):
        try:
            cfgs = json.loads(config_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON in config"); st.stop()
        runs = []
        with st.spinner("Retrieving…"):
            for key in selected:
                prov, name = key.split("/")
                try:
                    runs.append(_raise(requests.post(f"{API}/retrieve/run", timeout=120, json={
                        "provider": prov, "retriever": name, "query": query.strip(),
                        "top_k": top_k, "config": cfgs.get(key, {}),
                        "embed_provider": ep, "embed_name": en,
                    })))
                except Exception as exc:
                    runs.append({"retriever": key, "error": str(exc)})
        st.session_state["retrieval_runs"] = runs

    runs = st.session_state.get("retrieval_runs", [])
    if runs:
        cols = st.columns(len(runs))
        for col, run in zip(cols, runs):
            with col:
                if "error" in run:
                    st.markdown(f"**{run['retriever']}**")
                    st.error(run["error"]); continue
                st.markdown(f"**{run['provider']}/{run['retriever']}**")
                st.caption(f"{run['count']} results · query {run['query_ms']} ms"
                           f" · index {run['index_ms']} ms")
                if not run["results"]:
                    st.info("No results (no lexical match / empty store).")
                for r in run["results"]:
                    with st.expander(f"#{r['rank']} · {r['score']:.3f} · "
                                     f"{r['source'].split('/')[-1]}"):
                        st.write(r["content"])
                        meta = {k: v for k, v in r["metadata"].items() if v}
                        if meta:
                            st.json(meta)

with tab_hist:
    try:
        hist = _raise(requests.get(f"{API}/retrieve/results?limit=50", timeout=_T))["runs"]
    except Exception as exc:
        hist = []; st.caption(f"History unavailable: {exc}")
    if hist:
        st.dataframe([{
            "time": h["created_at"][:19], "retriever": f"{h['provider']}/{h['retriever']}",
            "query": h["query"][:60], "top_k": h["top_k"], "results": h["count"],
            "query_ms": h["query_ms"],
        } for h in hist], use_container_width=True)
        if st.button("Clear history"):
            _raise(requests.delete(f"{API}/retrieve/results", timeout=_T))
            st.rerun()
    else:
        st.info("No retrieval runs yet.")
