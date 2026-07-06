"""RAGx Streamlit UI — home / dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from ui.api_client import health, get_providers

st.set_page_config(
    page_title="RAGx",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAGx — Multi-Provider RAG Experimentation Platform")
st.caption("Plug-and-play · Every technique, every provider · Controlled from UI")

# ── API health ────────────────────────────────────────────────────────────────
h = health()
if h.get("status") == "ok":
    st.success("API is running on :8000")
else:
    st.error("API unreachable — run `make run-api` in a terminal first.")
    st.stop()

st.markdown("---")

# ── Available providers ───────────────────────────────────────────────────────
st.subheader("Available Providers by Phase")

try:
    providers = get_providers()
except Exception as exc:
    st.error(f"Could not fetch providers: {exc}")
    providers = {}

PHASE_LABELS = {
    "ingestion":  "Phase 1 — Ingestion",
    "chunking":   "Phase 2 — Chunking",
    "embedding":  "Phase 3 — Embedding",
    "vectordb":   "Phase 4 — Vector DB",
    "retriever":  "Phase 5 — Retriever",
    "reranker":   "Phase 6 — Reranker",
    "llm":        "Phase 7 — Inference",
    "evaluator":  "Phase 8 — Evaluation",
}

cols = st.columns(4)
for i, (phase, label) in enumerate(PHASE_LABELS.items()):
    with cols[i % 4]:
        phase_providers = providers.get(phase, {})
        total = sum(len(v) for v in phase_providers.values())
        status = "✅" if total > 0 else "🔜"
        with st.expander(f"{status} {label}", expanded=(total > 0)):
            if phase_providers:
                for prov, names in phase_providers.items():
                    st.markdown(f"**{prov}**")
                    for n in names:
                        st.markdown(f"  - `{n}`")
            else:
                st.caption("Coming in a future phase.")

st.markdown("---")
st.markdown("""
**Navigate using the sidebar:**

| Page | Phase | What it does |
|------|-------|-------------|
| 1 Ingestion | Phase 1 | Load files/URLs via LangChain, LlamaIndex, Haystack or Custom |
| 2 Chunking | Phase 2 | *(coming next)* |
| 3 Embedding | Phase 3 | *(coming next)* |
""")
