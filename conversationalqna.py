from __future__ import annotations

import os
import tempfile
import time

import chromadb  # noqa: F401
import streamlit as st
from chromadb.config import Settings  # noqa: F401
from groq import RateLimitError
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.bm25_rag import build_bm25_rag
from rag.evaluator import RAGMetrics, evaluate
from rag.tree_rag import build_tree_rag
from rag.vector_rag import build_vector_rag, rerank_documents

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatPDF-RAG v2",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.82rem;
    color: #cdd6f4;
}
.metric-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 4px; }
.metric-chip {
    background: #313244;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #89dceb;
}
.metric-label { color: #a6adc8; font-size: 0.75rem; }
.strategy-badge {
    display: inline-block;
    background: #45475a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.73rem;
    color: #cba6f7;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Secrets / env resolution ──────────────────────────────────────────────────
def _get_secret(key: str) -> str | None:
    try:
        value = st.secrets[key]
        return str(value).strip() if value else None
    except (KeyError, FileNotFoundError):
        return os.getenv(key) or None

def _get_secret_list(key: str) -> list[str]:
    """Support comma-separated keys: GROQ_API_KEY=key1,key2,key3"""
    raw = _get_secret(key)
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


GROQ_KEYS: list[str] = _get_secret_list("GROQ_API_KEY")
HF_TOKEN: str | None = _get_secret("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# ── Groq LLM with automatic key fallback ─────────────────────────────────────
def get_llm(keys: list[str], key_index: int = 0) -> ChatGroq:
    return ChatGroq(
        groq_api_key=keys[key_index],
        model_name="llama-3.1-8b-instant",
        temperature=0,
        streaming=True,
        max_tokens=512,
    )

def llm_with_fallback(keys: list[str]) -> ChatGroq:
    """Return an LLM; if rate-limited, rotate to next key."""
    if not keys:
        return None
    idx = st.session_state.get("groq_key_index", 0)
    return get_llm(keys, idx % len(keys))

def rotate_groq_key(keys: list[str]):
    """Called when a RateLimitError is caught."""
    idx = st.session_state.get("groq_key_index", 0)
    next_idx = (idx + 1) % len(keys)
    st.session_state.groq_key_index = next_idx
    st.warning(f"⚠️ Groq rate limit hit. Rotating to key {next_idx + 1}/{len(keys)}…")
    return get_llm(keys, next_idx)

# ── Cached: embedding model ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading embedding model…")
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Session-state initialisation ─────────────────────────────────────────────
STRATEGIES = ["⚡ Vector RAG", "🔍 VectorLess RAG", "🌲 Tree RAG"]

def _init_state():
    defaults = {
        "store": {},
        "chat_messages": {s: [] for s in STRATEGIES},
        "rag_chains": {},
        "processed_files": set(),
        "tree_stats": None,
        "vectorstores": {},
        "groq_key_index": 0,
        "splits": [],
        # Per-strategy chroma clients
        "chroma_client_vector": None,
        "chroma_client_tree": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Helper: session history ───────────────────────────────────────────────────
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# ── Helper: load PDFs ─────────────────────────────────────────────────────────
def load_pdfs(uploaded_files) -> list:
    documents = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="chatpdf_") as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        finally:
            os.remove(tmp_path)
    return documents

# ── Helper: render metrics panel ─────────────────────────────────────────────
def render_metrics(metrics: RAGMetrics, strategy: str):
    strategy_label = strategy.split(" ", 1)[1] if " " in strategy else strategy
    bar_color = (   # noqa
        "#89b4fa" if "Vector" in strategy
        else "#a6e3a1" if "VectorLess" in strategy
        else "#cba6f7"
    )
    faithfulness_pct = int(metrics.faithfulness * 100)
    score_pct = int(metrics.mean_score * 100)

    scores_display = " ".join(
        f'<span class="metric-chip">{s:.2f}</span>'
        for s in metrics.retrieval_scores
    )
    pages_display = (
        " ".join(f'<span class="metric-chip">p{p}</span>' for p in metrics.source_pages)
        if metrics.source_pages else '<span class="metric-chip">—</span>'
    )

    st.markdown(f"""
<div class="metric-box">
  <span class="strategy-badge">{strategy_label}</span>
  <div class="metric-row" style="margin-top:6px;">
    <div>
      <div class="metric-label">⏱ Latency</div>
      <strong>{metrics.latency_ms:.0f} ms</strong>
    </div>
    <div>
      <div class="metric-label">📦 Chunks</div>
      <strong>{metrics.chunks_used}</strong>
    </div>
    <div>
      <div class="metric-label">🎯 Mean Retrieval Score</div>
      <strong>{score_pct}%</strong>
    </div>
    <div>
      <div class="metric-label">✅ Faithfulness</div>
      <strong>{faithfulness_pct}%</strong>
    </div>
  </div>
  <div style="margin-top:8px;">
    <div class="metric-label">📊 Per-chunk scores</div>
    <div class="metric-row">{scores_display}</div>
  </div>
  <div style="margin-top:8px;">
    <div class="metric-label">📄 Source pages</div>
    <div class="metric-row">{pages_display}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helper: stream response with fallback + rerank + metrics ─────────────────
def run_query(
    strategy: str,
    user_input: str,
    session_id: str,
    active_keys: list[str],
):
    chain = st.session_state.rag_chains.get(strategy)
    if chain is None:
        st.error("Chain not built. Process your PDFs first.")
        return

    start_time = time.time()
    streamed_answer = ""
    source_docs = []

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            for chunk in chain.stream(
                {"input": user_input},
                config={"configurable": {"session_id": f"{strategy}_{session_id}"}},
            ):
                if "answer" in chunk:
                    streamed_answer += chunk["answer"]
                    placeholder.markdown(streamed_answer + "▌")
                if "context" in chunk:
                    source_docs = chunk["context"]

        except RateLimitError:
            # Rotate key and retry once
            llm_new = rotate_groq_key(active_keys)   # noqa
            st.info("Retrying with next API key…")
            try:
                result = chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": f"{strategy}_{session_id}"}},
                )
                streamed_answer = result.get("answer", "")
                source_docs = result.get("context", [])
            except Exception as e:
                streamed_answer = f"❌ All API keys exhausted or error: {e}"
        except Exception as e:
            streamed_answer = (
                f"❌ Error: {str(e)}\n\nTry re-uploading PDFs."
            )

        placeholder.markdown(streamed_answer)

        # ── Apply cross-encoder reranking post-hoc for display only ──────
        if strategy == STRATEGIES[0] and source_docs:
            source_docs = rerank_documents(user_input, source_docs, top_k=4)

        # ── Compute and render metrics ─────────────────────────────────────
        metrics = evaluate(
            query=user_input,
            answer=streamed_answer,
            source_docs=source_docs,
            embeddings=load_embeddings(),
            start_time=start_time,
        )
        render_metrics(metrics, strategy)

    # Append to chat history for this strategy
    st.session_state.chat_messages[strategy].append(
        {"role": "assistant", "content": streamed_answer, "metrics": metrics.as_dict()}
    )


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 ChatPDF-RAG v2")
    st.caption("Industry-grade RAG with 3 retrieval strategies")
    st.divider()

    # ── API key management ─────────────────────────────────────────────────
    st.markdown("### 🔑 API Keys")

    if GROQ_KEYS:
        st.success(f"✅ {len(GROQ_KEYS)} Groq key(s) loaded from environment.")
        st.caption("Keys rotate automatically on rate-limit.")
        manual_keys_input = None
    else:
        manual_keys_input = st.text_area(
            "Groq API Key(s)",
            placeholder="gsk_key1\ngsk_key2  (one per line for fallback)",
            height=100,
            help="Paste one key per line. Multiple keys rotate on rate-limit.",
        )

    st.divider()

    # ── Session ID ──────────────────────────────────────────────────────────
    session_id = st.text_input(
        "🆔 Session ID",
        value="default_session",
        help="Change to start a fresh conversation.",
    )

    st.divider()

    # ── File upload & processing ────────────────────────────────────────────
    st.markdown("### 📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡ Process PDFs", use_container_width=True, type="primary")

    if st.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.chat_messages = {s: [] for s in STRATEGIES}
        st.session_state.store = {}
        st.rerun()

    st.divider()

    # ── Tree index stats ────────────────────────────────────────────────────
    if st.session_state.tree_stats:
        ts = st.session_state.tree_stats
        st.markdown("### 🌲 Tree Index")
        st.markdown(f"""
- Leaf nodes: **{ts['leaf_count']}**
- Branch summaries: **{ts['branch_count']}**
- Root summary: **1**
- Total nodes: **{ts['total_nodes']}**
""")

    # ── Current key status ──────────────────────────────────────────────────
    active_idx = st.session_state.get("groq_key_index", 0)
    if GROQ_KEYS:
        st.caption(f"Active key: #{active_idx + 1}/{len(GROQ_KEYS)}")

# ── Resolve active keys ────────────────────────────────────────────────────────
if GROQ_KEYS:
    active_keys = GROQ_KEYS
elif manual_keys_input:
    active_keys = [k.strip() for k in manual_keys_input.strip().splitlines() if k.strip()]
else:
    active_keys = []

if not active_keys:
    st.warning("⚠️ Please provide at least one Groq API key in the sidebar.")
    st.stop()

llm = llm_with_fallback(active_keys)

# ── PDF processing ─────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        file_ids = {f.name for f in uploaded_files}
        if file_ids != st.session_state.processed_files:
            with st.spinner("📄 Loading and splitting PDFs…"):
                docs = load_pdfs(uploaded_files)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600, chunk_overlap=80
                )
                splits = splitter.split_documents(docs)
                st.session_state.splits = splits

            col1, col2, col3 = st.columns(3)

            # ── Strategy 1: Vector RAG ──────────────────────────────────────
            with col1:
                with st.spinner("⚡ Building Vector RAG (Hybrid + Rerank)…"):
                    chain_v, vs = build_vector_rag(
                        splits, llm, load_embeddings(), get_session_history
                    )
                    st.session_state.rag_chains[STRATEGIES[0]] = chain_v
                    st.session_state.vectorstores["vector"] = vs
                st.success("✅ Vector RAG ready")

            # ── Strategy 2: VectorLess RAG ─────────────────────────────────
            with col2:
                with st.spinner("🔍 Building VectorLess RAG (BM25)…"):
                    chain_b = build_bm25_rag(splits, llm, get_session_history)
                    st.session_state.rag_chains[STRATEGIES[1]] = chain_b
                st.success("✅ VectorLess RAG ready")

            # ── Strategy 3: Tree RAG ───────────────────────────────────────
            with col3:
                chain_t, tree_stats = build_tree_rag(
                    splits, llm, load_embeddings(), get_session_history
                )
                st.session_state.rag_chains[STRATEGIES[2]] = chain_t
                st.session_state.tree_stats = tree_stats
                st.success("✅ Tree RAG ready")

            st.session_state.processed_files = file_ids
            st.session_state.chat_messages = {s: [] for s in STRATEGIES}
            st.session_state.store = {}
            st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) processed across all 3 strategies!")
        else:
            st.sidebar.info("These PDFs are already loaded.")

# ── Main: check if chains are built ────────────────────────────────────────────
if not st.session_state.rag_chains:
    st.markdown("""
## 👋 Welcome to ChatPDF-RAG v2

Upload your PDFs in the sidebar and click **Process PDFs** to activate all three RAG strategies.

| Tab | Strategy | Best for |
|---|---|---|
| ⚡ Vector RAG | Hybrid dense+BM25, cross-encoder reranked | General Q&A, semantic questions |
| 🔍 VectorLess RAG | Pure BM25 + LLM filter, zero embeddings | Exact terms, legal/medical docs |
| 🌲 Tree RAG | Hierarchical tree index (RAPTOR-style) | Long docs, summarization, big-picture |
""")
    st.stop()

# ── Tabs ────────────────────────────────────────────────────────────────────────
st.title("📄 ChatPDF-RAG v2")
tabs = st.tabs(STRATEGIES)

for tab, strategy in zip(tabs, STRATEGIES):
    with tab:
        is_ready = strategy in st.session_state.rag_chains
        if not is_ready:
            st.info(f"Process your PDFs to activate {strategy}.")
            continue

        # Show chat history for this strategy
        for msg in st.session_state.chat_messages[strategy]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "metrics" in msg:
                    m = msg["metrics"]
                    render_metrics(
                        RAGMetrics(
                            retrieval_scores=m["retrieval_scores"],
                            mean_score=m["mean_score"],
                            faithfulness=m["faithfulness"],
                            latency_ms=m["latency_ms"],
                            chunks_used=m["chunks_used"],
                            source_pages=m["source_pages"],
                        ),
                        strategy,
                    )

        # User input — unique key per tab to avoid Streamlit widget collision
        user_input = st.chat_input(
            f"Ask a question using {strategy}…",
            key=f"input_{strategy}",
        )

        if user_input:
            st.session_state.chat_messages[strategy].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            run_query(strategy, user_input, session_id, active_keys)
            st.rerun()