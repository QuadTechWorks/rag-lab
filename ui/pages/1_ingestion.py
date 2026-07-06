"""Phase 1 — Ingestion.

Three input modes:
  • File Upload  — pick provider + loader, upload one file
  • URL          — fetch a web page with chosen provider + loader
  • Folder Path  — scan a local folder; auto-routes every file; loads with per-file progress
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import requests

API = "http://localhost:8000/api"
_T  = 60   # default request timeout


def _raise(r: requests.Response):
    """Raise with the API's detail message, not just the HTTP status."""
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise Exception(f"{detail}")


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
st.set_page_config(page_title="Ingestion — RAGx", layout="wide")
st.title("Phase 1 — Ingestion")
st.caption("Load documents from files, URLs, or entire folders · Auto-routes every file to its best loader")

try:
    h = requests.get("http://localhost:8000/health", timeout=4).json()
    assert h.get("status") == "ok"
except Exception:
    st.error("API unreachable — run `make run-api` first.")
    st.stop()

try:
    providers: dict = _get("/providers/ingestion")
except Exception as exc:
    st.error(f"Cannot fetch providers: {exc}")
    st.stop()

provider_names = list(providers.keys())

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_file, tab_url, tab_folder = st.tabs(["📄 File Upload", "🌐 URL", "📁 Folder Path"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — File upload
# ════════════════════════════════════════════════════════════════════════════
with tab_file:
    col_cfg, col_doc = st.columns([1, 2], gap="large")
    with col_cfg:
        st.subheader("Configure")
        sel_prov = st.selectbox("Provider", provider_names, key="fp_prov")
        sel_ldr  = st.selectbox("Loader",   providers.get(sel_prov, []), key="fp_ldr")
        uploaded = st.file_uploader(
            "Choose file",
            type=["pdf","txt","md","csv","docx","xlsx","pptx","html","htm",
                  "json","jsonl","eml","ipynb","xml","yaml","yml",
                  "epub","rtf","py","js","ts","java","go","rs","cpp","c","sql"],
        )
        if st.button("Load File", disabled=not uploaded, use_container_width=True):
            with st.spinner(f"Using **{sel_prov} / {sel_ldr}** to load `{uploaded.name}`…"):
                try:
                    res = _post("/ingest/upload",
                        files={"file": (uploaded.name, uploaded.read())},
                        data={"provider": sel_prov, "loader": sel_ldr},
                    )
                    st.success(f"✅ Loaded **{res['count']}** document(s) from `{uploaded.name}`")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — URL
# ════════════════════════════════════════════════════════════════════════════
with tab_url:
    col_cfg2, _ = st.columns([1, 2], gap="large")
    with col_cfg2:
        st.subheader("Configure")
        sel_prov2 = st.selectbox("Provider", provider_names, key="url_prov")
        web_ldrs  = [n for n in providers.get(sel_prov2, []) if n in ("web", "simple")] \
                    or providers.get(sel_prov2, [])
        sel_ldr2  = st.selectbox("Loader", web_ldrs, key="url_ldr")
        url = st.text_input("URL", placeholder="https://example.com/docs/page")
        if st.button("Load URL", disabled=not url.strip(), use_container_width=True):
            with st.spinner(f"Fetching via **{sel_prov2} / {sel_ldr2}**…"):
                try:
                    res = _post("/ingest/url",
                        json={"url": url.strip(), "provider": sel_prov2, "loader": sel_ldr2},
                    )
                    st.success(f"✅ Loaded **{res['count']}** document(s) from URL")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Folder Path  (with per-file live progress)
# ════════════════════════════════════════════════════════════════════════════
with tab_folder:
    st.subheader("Folder Scanner")
    st.caption(
        "RAGx scans the folder, picks the best loader per file type, "
        "then loads each file individually — you see real-time progress."
    )

    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        folder_path   = st.text_input("Folder path",
                            placeholder="/Users/you/Documents/project")
        recursive     = st.toggle("Scan sub-folders recursively", value=True)
        max_size_mb   = st.slider("Skip files larger than (MB)", 1, 500, 50)

        # Validate path exists before enabling buttons
        path_ok = False
        if folder_path.strip():
            p = Path(folder_path.strip())
            if p.exists() and p.is_dir():
                st.success(f"✅ Folder found")
                path_ok = True
            else:
                st.error(f"Folder not found: `{folder_path.strip()}`\n\n"
                         f"Tip: try `/Users/gokulraj/Documents/Pipeline/ragx`")

        btn_scan = st.button("🔍 Scan Folder",
                             disabled=not path_ok,
                             use_container_width=True)
        btn_load = st.button("⚡ Load All Files",
                             disabled=not path_ok,
                             use_container_width=True)

    with col_r:
        # ── SCAN ─────────────────────────────────────────────────────────
        if btn_scan:
            with st.spinner("Scanning folder…"):
                try:
                    scan = _post("/ingest/folder/scan", json={
                        "folder_path": folder_path.strip(),
                        "recursive":   recursive,
                        "max_file_size_mb": max_size_mb,
                    })
                    st.session_state["folder_scan"] = scan
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")

        if "folder_scan" in st.session_state and st.session_state["folder_scan"]:
            scan = st.session_state["folder_scan"]
            plan = scan.get("plan", [])
            loadable = [f for f in plan if not f["skipped"]]
            skipped  = [f for f in plan if f["skipped"]]

            m1, m2, m3 = st.columns(3)
            m1.metric("Total files",  scan["total_files"])
            m2.metric("Will load",    scan["loadable"])
            m3.metric("Skipped",      scan["skipped"])

            if loadable:
                st.markdown("**Files that will be loaded:**")
                table = [{
                    "File":      f["file"],
                    "Ext":       f["ext"],
                    "Provider":  f["provider"],
                    "Loader":    f["loader"],
                    "Size (KB)": f["size_kb"],
                } for f in loadable]
                st.dataframe(table, use_container_width=True, hide_index=True)

            if skipped:
                with st.expander(f"Skipped files ({len(skipped)})"):
                    for f in skipped:
                        reason = f.get("skip_reason", "unsupported type")
                        icon   = "🔇" if "Binary" in reason else ("📦" if "not installed" in reason else "❓")
                        st.caption(f"  {icon} `{f['file']}` — {reason}")

        # ── LOAD with per-file progress ───────────────────────────────────
        if btn_load:
            # Always do a fresh scan first
            try:
                scan = _post("/ingest/folder/scan", json={
                    "folder_path": folder_path.strip(),
                    "recursive":   recursive,
                    "max_file_size_mb": max_size_mb,
                })
                st.session_state["folder_scan"] = scan
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                st.stop()

            plan     = scan.get("plan", [])
            loadable = [f for f in plan if not f["skipped"]]

            if not loadable:
                st.warning("No loadable files found in this folder.")
            else:
                st.markdown(f"**Loading {len(loadable)} files…**")

                progress_bar  = st.progress(0)
                status_text   = st.empty()
                log_container = st.container()

                ok_count  = 0
                err_count = 0
                doc_total = 0

                for i, entry in enumerate(loadable):
                    fname    = entry["file"]
                    provider = entry["provider"]
                    loader   = entry["loader"]
                    fpath    = entry["path"]

                    # Live status line
                    status_text.info(
                        f"[{i+1}/{len(loadable)}]  📄 `{fname}`  →  "
                        f"using **{provider} / {loader}**…"
                    )

                    try:
                        res = _post("/ingest/file-path", json={
                            "file_path": fpath,
                            "provider":  provider,
                            "loader":    loader,
                        })
                        doc_count = res.get("docs", 0)
                        doc_total += doc_count
                        ok_count  += 1
                        log_container.success(
                            f"✅  `{fname}`  →  **{provider} / {loader}**  "
                            f"→  {doc_count} doc(s)"
                        )
                    except Exception as exc:
                        err_count += 1
                        log_container.error(f"❌  `{fname}`  →  {exc}")

                    progress_bar.progress((i + 1) / len(loadable))

                status_text.empty()
                progress_bar.empty()

                if err_count == 0:
                    st.success(
                        f"Done — **{ok_count}** files loaded  →  "
                        f"**{doc_total}** total documents"
                    )
                else:
                    st.warning(
                        f"Done — {ok_count} OK  ·  {err_count} errors  ·  "
                        f"{doc_total} total documents"
                    )

# ════════════════════════════════════════════════════════════════════════════
# Document Store (shared across all tabs)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Document Store")

col_r2, col_c2, _ = st.columns([1, 1, 5])
if col_r2.button("Refresh", use_container_width=True):
    st.rerun()
if col_c2.button("Clear All", use_container_width=True):
    try:
        _delete("/ingest/documents")
        st.session_state.pop("folder_scan", None)
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

try:
    store = _get("/ingest/documents")
except Exception as exc:
    st.error(str(exc))
    st.stop()

docs = store.get("documents", [])
st.caption(f"**{len(docs)}** document(s) in store")

for doc in docs:
    fname = (
        doc.get("metadata", {}).get("filename")
        or Path(doc.get("source", "")).name
        or doc["id"][:8]
    )
    header = (
        f"📄 {fname}  ·  {doc['char_count']:,} chars  "
        f"·  `{doc['provider']} / {doc['loader']}`"
    )
    with st.expander(header):
        meta = doc.get("metadata", {})
        if meta:
            st.json(meta, expanded=False)
        content = doc.get("content", "")
        st.code(content[:700] + ("…" if len(content) > 700 else ""), language=None)
        if st.button("Delete", key=f"del_{doc['id']}"):
            _delete(f"/ingest/documents/{doc['id']}")
            st.rerun()
