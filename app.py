"""
app.py
------
Lattice — Smart Document Knowledge Assistant
Main Streamlit application entry point.

Layout: 3-column dark UI
  LEFT   → Document Vault (upload, index, manage docs)
  CENTER → Knowledge Chat (Q&A with history)
  RIGHT  → Evidence Panel (retrieved sources)
"""

import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing llm (which also loads it, but being explicit is safer)
load_dotenv()

from document_processor import process_multiple_documents
from rag import index_chunks, load_index, clear_index, retrieve, TOP_K
from llm import generate_answer, get_api_key

# ──────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lattice — Smart Document Knowledge Assistant",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Dark premium theme
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root theme ── */
:root {
    --bg-primary:   #0a0d14;
    --bg-secondary: #111520;
    --bg-card:      #161b27;
    --bg-card-hover:#1c2235;
    --border:       #252d42;
    --border-accent:#3d4f7c;
    --text-primary: #e8eaf0;
    --text-secondary:#8892a4;
    --text-muted:   #4a5568;
    --accent-blue:  #4f8ef7;
    --accent-purple:#7c5cbf;
    --accent-teal:  #2dd4bf;
    --accent-green: #34d399;
    --accent-amber: #f59e0b;
    --accent-red:   #f87171;
    --gradient-1:   linear-gradient(135deg, #4f8ef7 0%, #7c5cbf 100%);
    --gradient-2:   linear-gradient(135deg, #2dd4bf 0%, #4f8ef7 100%);
}

/* ── Base overrides ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background-color: var(--bg-primary) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* ── Column containers ── */
[data-testid="column"] {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1rem !important;
    min-height: 90vh;
}

/* ── Headers ── */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Lattice branding ── */
.lattice-logo {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.lattice-tagline {
    color: var(--text-secondary);
    font-size: 0.72rem;
    margin-top: -2px;
    letter-spacing: 0.3px;
}

/* ── Section labels ── */
.section-label {
    color: var(--text-secondary);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* ── Document cards ── */
.doc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin: 0.4rem 0;
    transition: border-color 0.2s;
}
.doc-card:hover { border-color: var(--border-accent); }
.doc-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-meta {
    font-size: 0.70rem;
    color: var(--text-secondary);
    margin-top: 2px;
}
.doc-badge {
    display: inline-block;
    background: rgba(79, 142, 247, 0.15);
    color: var(--accent-blue);
    border: 1px solid rgba(79, 142, 247, 0.3);
    border-radius: 20px;
    font-size: 0.62rem;
    padding: 1px 8px;
    font-weight: 600;
    margin-right: 4px;
}
.doc-badge.green {
    background: rgba(52, 211, 153, 0.15);
    color: var(--accent-green);
    border-color: rgba(52, 211, 153, 0.3);
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-box {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.5rem;
    text-align: center;
}
.stat-number {
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.62rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Chat messages ── */
.chat-user {
    background: rgba(79, 142, 247, 0.12);
    border: 1px solid rgba(79, 142, 247, 0.25);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: var(--text-primary);
    max-width: 88%;
    margin-left: auto;
}
.chat-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: var(--text-primary);
    max-width: 92%;
    line-height: 1.65;
}
.chat-label-user {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
    text-align: right;
}
.chat-label-ai {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent-teal);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.retrieval-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid rgba(45, 212, 191, 0.25);
    color: var(--accent-teal);
    border-radius: 20px;
    font-size: 0.65rem;
    padding: 2px 10px;
    font-weight: 600;
    margin-bottom: 6px;
}

/* ── Welcome card ── */
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 2rem 0;
}
.welcome-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
}
.welcome-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.welcome-sub {
    color: var(--text-secondary);
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ── Source / evidence cards ── */
.source-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-purple);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}
.source-card:hover { border-left-color: var(--accent-blue); }
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.source-filename {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-blue);
}
.source-score {
    font-size: 0.65rem;
    font-weight: 600;
    background: rgba(124, 92, 191, 0.15);
    color: var(--accent-purple);
    border: 1px solid rgba(124, 92, 191, 0.3);
    border-radius: 20px;
    padding: 1px 8px;
}
.source-meta-row {
    display: flex;
    gap: 6px;
    margin-bottom: 0.5rem;
}
.source-meta-tag {
    font-size: 0.62rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
    color: var(--text-secondary);
}
.source-snippet {
    font-size: 0.76rem;
    color: var(--text-secondary);
    line-height: 1.55;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
    padding: 0.5rem 0.6rem;
    max-height: 120px;
    overflow-y: auto;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gradient-1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Danger / clear buttons */
.danger-btn > button {
    background: rgba(248, 113, 113, 0.1) !important;
    color: var(--accent-red) !important;
    border: 1px solid rgba(248, 113, 113, 0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-accent) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

/* ── Text input ── */
.stTextInput > div > div > input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(79, 142, 247, 0.2) !important;
}

/* ── Status messages ── */
.stSuccess { background: rgba(52, 211, 153, 0.1) !important; border-color: rgba(52, 211, 153, 0.3) !important; }
.stWarning { background: rgba(245, 158, 11, 0.1) !important; border-color: rgba(245, 158, 11, 0.3) !important; }
.stError   { background: rgba(248, 113, 113, 0.1) !important; border-color: rgba(248, 113, 113, 0.3) !important; }
.stInfo    { background: rgba(79, 142, 247, 0.1) !important;  border-color: rgba(79, 142, 247, 0.3) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────
def init_session():
    """Initialise all session state keys used across the app."""
    defaults = {
        "chat_history": [],          # [{"role": "user"|"assistant", "content": str}]
        "last_sources": [],          # last retrieved chunks for the evidence panel
        "faiss_index": None,         # in-memory FAISS index
        "metadata": None,            # list of chunk dicts
        "indexed_files": [],         # filenames that have been processed + indexed
        "total_chunks": 0,           # running count of indexed chunks
        "processing_errors": {},     # filename → error message
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to the documents/ directory.
    Returns the absolute path of the saved file.
    """
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    dest = docs_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def format_score(score: float) -> str:
    """Format cosine similarity score as a percentage string."""
    return f"{score * 100:.1f}%"


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text for preview display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ──────────────────────────────────────────────
# PROCESS + INDEX DOCUMENTS
# ──────────────────────────────────────────────
def process_and_index_files(uploaded_files) -> None:
    """
    For each uploaded file:
      1. Save to disk
      2. Extract + chunk text (document_processor)
      3. Build / update FAISS index (rag)
      4. Update session state
    """
    if not uploaded_files:
        return

    # Filter out already-indexed files
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state["indexed_files"]
    ]
    if not new_files:
        st.info("All selected files are already indexed.", icon="ℹ️")
        return

    with st.spinner(f"Processing {len(new_files)} file(s)…"):
        # Save files to disk
        saved_paths = []
        for uf in new_files:
            try:
                path = save_uploaded_file(uf)
                saved_paths.append(path)
            except Exception as e:
                st.session_state["processing_errors"][uf.name] = str(e)

        if not saved_paths:
            st.error("No files could be saved. Check write permissions.")
            return

        # Process (extract + chunk)
        try:
            chunks, errors = process_multiple_documents(saved_paths)
            st.session_state["processing_errors"].update(errors)
        except Exception as e:
            st.error(f"Document processing failed: {e}")
            return

        if not chunks:
            st.error("No text could be extracted from the uploaded files.")
            return

        # Merge with any existing chunks
        existing_meta = st.session_state.get("metadata") or []
        all_chunks = existing_meta + chunks
        # Re-assign global chunk IDs
        for i, c in enumerate(all_chunks):
            c["chunk_id"] = i

        # Build / rebuild FAISS index
        try:
            index, metadata = index_chunks(all_chunks)
            st.session_state["faiss_index"] = index
            st.session_state["metadata"] = metadata
            st.session_state["total_chunks"] = len(metadata)
            # Mark files as indexed
            for uf in new_files:
                if uf.name not in errors:
                    st.session_state["indexed_files"].append(uf.name)
        except Exception as e:
            st.error(f"FAISS indexing failed: {e}")
            return

    # Report results
    success_count = len(new_files) - len(errors)
    if success_count:
        st.success(
            f"✅ {success_count} file(s) indexed — "
            f"{len(all_chunks)} total chunks in knowledge base.",
            icon="🔷",
        )
    for fname, err in errors.items():
        st.warning(f"⚠️ {fname}: {err}")


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():
    init_session()

    # ── Top bar ──
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
            <div>
                <div class="lattice-logo">🔷 LATTICE</div>
                <div class="lattice-tagline">Your documents. Your knowledge. One intelligent workspace.</div>
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    # ── 3-Column Layout ──
    col_left, col_center, col_right = st.columns([1.1, 2.2, 1.5], gap="small")

    # ══════════════════════════════════════════
    # LEFT COLUMN — DOCUMENT VAULT
    # ══════════════════════════════════════════
    with col_left:
        st.markdown('<div class="section-label">📂 Document Vault</div>', unsafe_allow_html=True)

        # Stats strip
        n_docs = len(st.session_state["indexed_files"])
        n_chunks = st.session_state["total_chunks"]
        st.markdown(
            f"""
            <div class="stats-strip">
                <div class="stat-box">
                    <div class="stat-number">{n_docs}</div>
                    <div class="stat-label">Docs</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{n_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )

        if st.button("⚡ Index Documents", use_container_width=True):
            if uploaded_files:
                process_and_index_files(uploaded_files)
                st.rerun()
            else:
                st.warning("Please upload at least one file first.", icon="⚠️")

        # Indexed documents list
        if st.session_state["indexed_files"]:
            st.markdown('<div class="section-label">✅ Indexed Files</div>', unsafe_allow_html=True)
            for fname in st.session_state["indexed_files"]:
                # Count chunks belonging to this file
                meta = st.session_state.get("metadata") or []
                file_chunks = sum(1 for c in meta if c["source"] == fname)
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-name">📄 {fname}</div>
                        <div class="doc-meta">
                            <span class="doc-badge green">{file_chunks} chunks</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Processing errors
        if st.session_state["processing_errors"]:
            st.markdown('<div class="section-label">⚠️ Errors</div>', unsafe_allow_html=True)
            for fname, err in st.session_state["processing_errors"].items():
                st.markdown(
                    f'<div class="doc-card" style="border-left:3px solid #f87171;">'
                    f'<div class="doc-name" style="color:#f87171;">❌ {fname}</div>'
                    f'<div class="doc-meta" style="font-size:0.68rem;">{err[:120]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Clear knowledge base ──
        st.markdown('<div class="section-label">⚙️ Reset</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                try:
                    clear_index()
                except Exception:
                    pass
                # Reset all document-related state
                st.session_state["faiss_index"] = None
                st.session_state["metadata"] = None
                st.session_state["indexed_files"] = []
                st.session_state["total_chunks"] = 0
                st.session_state["processing_errors"] = {}
                st.session_state["last_sources"] = []
                st.session_state["chat_history"] = []
                st.success("Knowledge base cleared.", icon="✅")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # API key status indicator (no key value shown)
        api_ok = bool(get_api_key())
        key_status = "🟢 API Key Loaded" if api_ok else "🔴 API Key Missing"
        key_color = "#34d399" if api_ok else "#f87171"
        st.markdown(
            f'<div style="font-size:0.68rem;color:{key_color};margin-top:1rem;'
            f'text-align:center;">{key_status}</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════
    # CENTER COLUMN — KNOWLEDGE CHAT
    # ══════════════════════════════════════════
    with col_center:
        st.markdown('<div class="section-label">💬 Knowledge Chat</div>', unsafe_allow_html=True)

        has_index = st.session_state["faiss_index"] is not None

        # ── Welcome card (shown when no chat history) ──
        if not st.session_state["chat_history"]:
            if has_index:
                welcome_msg = (
                    f"Knowledge base ready with <strong>{n_chunks} chunks</strong> "
                    f"from <strong>{n_docs}</strong> document(s). Ask me anything!"
                )
            else:
                welcome_msg = (
                    "Upload one or more PDF / TXT files in the <strong>Document Vault</strong> "
                    "on the left, then click <strong>⚡ Index Documents</strong> to get started."
                )
            st.markdown(
                f"""
                <div class="welcome-card">
                    <div class="welcome-icon">🔷</div>
                    <div class="welcome-title">Welcome to Lattice</div>
                    <div class="welcome-sub">{welcome_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Chat history display ──
        chat_container = st.container()
        with chat_container:
            for turn in st.session_state["chat_history"]:
                if turn["role"] == "user":
                    st.markdown(
                        f'<div class="chat-label-user">You</div>'
                        f'<div class="chat-user">{turn["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    retrieval_info = turn.get("retrieval_info", "")
                    retrieval_badge = (
                        f'<div class="retrieval-badge">🔍 {retrieval_info}</div>'
                        if retrieval_info
                        else ""
                    )
                    st.markdown(
                        f'<div class="chat-label-ai">🔷 Lattice</div>'
                        f"{retrieval_badge}"
                        f'<div class="chat-assistant">{turn["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Question input ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. What is a Hilbert space?",
                label_visibility="collapsed",
                key="question_input",
            )
            submitted = st.form_submit_button("🚀 Ask Lattice", use_container_width=True)

        if submitted and question.strip():
            q = question.strip()

            # Guard: must have an index
            if not has_index:
                st.warning(
                    "⚠️ No documents indexed yet. "
                    "Please upload files and click ⚡ Index Documents first.",
                    icon="⚠️",
                )
            else:
                # ── RETRIEVE ──
                with st.spinner("Searching knowledge base…"):
                    try:
                        sources = retrieve(
                            q,
                            st.session_state["faiss_index"],
                            st.session_state["metadata"],
                            top_k=TOP_K,
                        )
                        st.session_state["last_sources"] = sources
                    except Exception as e:
                        st.error(f"Retrieval error: {e}")
                        sources = []

                # ── GENERATE ──
                retrieval_info = (
                    f"Found {len(sources)} relevant passage(s)"
                    if sources
                    else "No relevant passages found"
                )

                # Build chat history list for the LLM (exclude retrieval_info key)
                history_for_llm = [
                    {"role": t["role"], "content": t["content"]}
                    for t in st.session_state["chat_history"]
                ]

                with st.spinner("Generating answer…"):
                    answer = generate_answer(q, sources, history_for_llm)

                # ── Store in history ──
                st.session_state["chat_history"].append(
                    {"role": "user", "content": q}
                )
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "retrieval_info": retrieval_info,
                    }
                )
                st.rerun()

        # ── Clear chat ──
        if st.session_state["chat_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Conversation", use_container_width=False):
                st.session_state["chat_history"] = []
                st.session_state["last_sources"] = []
                st.rerun()

    # ══════════════════════════════════════════
    # RIGHT COLUMN — EVIDENCE / RETRIEVED SOURCES
    # ══════════════════════════════════════════
    with col_right:
        st.markdown('<div class="section-label">🔍 Retrieved Sources</div>', unsafe_allow_html=True)

        sources = st.session_state.get("last_sources", [])

        if not sources:
            st.markdown(
                """
                <div style="text-align:center;padding:2rem 0.5rem;color:#4a5568;">
                    <div style="font-size:2rem;margin-bottom:0.6rem;">📑</div>
                    <div style="font-size:0.78rem;">
                        Source passages will appear here after you ask a question.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#8892a4;margin-bottom:0.6rem;">'
                f"Showing top {len(sources)} retrieved chunk(s)</div>",
                unsafe_allow_html=True,
            )
            for i, chunk in enumerate(sources, start=1):
                source = chunk.get("source", "Unknown")
                page = chunk.get("page", "?")
                chunk_id = chunk.get("chunk_id", "?")
                score = chunk.get("score", 0.0)
                text = chunk.get("text", "")
                snippet = truncate(text, 300)

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-filename">📄 {source}</span>
                            <span class="source-score">{format_score(score)}</span>
                        </div>
                        <div class="source-meta-row">
                            <span class="source-meta-tag">Page {page}</span>
                            <span class="source-meta-tag">Chunk #{chunk_id}</span>
                            <span class="source-meta-tag">#{i}</span>
                        </div>
                        <div class="source-snippet">{snippet}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
