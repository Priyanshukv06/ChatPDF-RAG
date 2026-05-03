"""
ChatPDF-RAG v2 — Industry-Grade Conversational RAG
====================================================
Tabs: ⚡ Vector | 🔍 VectorLess | 🌲 Tree | 💡 HyDE | 🤔 Self-RAG | ⚖️ Compare
Features: Document Analysis | Source Citations | Session Analytics | Export
"""
from __future__ import annotations
import os, tempfile, time
from datetime import datetime
import chromadb
import streamlit as st
from chromadb.config import Settings
from groq import RateLimitError
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.analyzer import analyze_document
from rag.bm25_rag import build_bm25_rag
from rag.evaluator import RAGMetrics, evaluate
from rag.hyde_rag import build_hyde_rag
from rag.self_rag import build_self_rag, SelfRAGChain
from rag.tree_rag import build_tree_rag
from rag.vector_rag import build_vector_rag, rerank_documents

st.set_page_config(page_title="ChatPDF-RAG v2", page_icon="📄", layout="wide")

st.markdown("""<style>
.metric-box{background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:10px 14px;margin-top:8px;font-size:0.82rem;color:#cdd6f4}
.metric-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:4px}
.metric-chip{background:#313244;border-radius:6px;padding:3px 10px;font-size:0.78rem;color:#89dceb}
.metric-label{color:#a6adc8;font-size:0.75rem}
.strategy-badge{display:inline-block;background:#45475a;border-radius:20px;padding:2px 10px;font-size:0.73rem;color:#cba6f7;margin-bottom:4px}
.citation-box{background:#181825;border-left:3px solid #89b4fa;border-radius:0 8px 8px 0;padding:8px 12px;margin:4px 0;font-size:0.80rem;color:#cdd6f4}
.doc-analysis-box{background:#1e1e2e;border:1px solid #45475a;border-radius:10px;padding:14px 18px;margin-bottom:16px}
.topic-tag{display:inline-block;background:#313244;border-radius:12px;padding:2px 10px;font-size:0.76rem;color:#a6e3a1;margin:2px}
</style>""", unsafe_allow_html=True)

# ── Secrets ──────────────────────────────────────────────────────────────────
def _get_secret(key):
    try:
        v = st.secrets[key]
        return str(v).strip() if v else None
    except (KeyError, FileNotFoundError):
        return os.getenv(key) or None

def _get_secret_list(key):
    raw = _get_secret(key)
    return [k.strip() for k in raw.split(",") if k.strip()] if raw else []

GROQ_KEYS = _get_secret_list("GROQ_API_KEY")
HF_TOKEN = _get_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# ── LLM ──────────────────────────────────────────────────────────────────────
def get_llm(keys, idx=0):
    return ChatGroq(groq_api_key=keys[idx], model_name="llama-3.1-8b-instant",
                   temperature=0, streaming=True, max_tokens=512)

def llm_with_fallback(keys):
    if not keys: return None
    return get_llm(keys, st.session_state.get("groq_key_index", 0) % len(keys))

def rotate_groq_key(keys):
    idx = (st.session_state.get("groq_key_index", 0) + 1) % len(keys)
    st.session_state.groq_key_index = idx
    st.warning("⚠️ Rate limit — rotating to key " + str(idx + 1) + "/" + str(len(keys)))
    return get_llm(keys, idx)

@st.cache_resource(show_spinner="⏳ Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Constants ────────────────────────────────────────────────────────────────
STRATEGIES = ["⚡ Vector RAG", "🔍 VectorLess RAG", "🌲 Tree RAG", "💡 HyDE RAG", "🤔 Self-RAG"]
COMPARE_TAB = "⚖️ Compare"
ALL_TABS = STRATEGIES + [COMPARE_TAB]

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "store": {}, "chat_messages": {s: [] for s in ALL_TABS},
        "rag_chains": {}, "processed_files": set(), "tree_stats": None,
        "groq_key_index": 0, "splits": [], "doc_analysis": None,
        "session_analytics": {s: {"q": 0, "lat": 0.0, "faith": 0.0} for s in STRATEGIES},
        "chroma_client_vector": None, "chroma_client_tree": None,
        "chroma_client_hyde": None, "chroma_client_self_rag": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

def get_session_history(sid):
    if sid not in st.session_state.store:
        st.session_state.store[sid] = InMemoryChatMessageHistory()
    return st.session_state.store[sid]

def load_pdfs(files):
    docs = []
    for uf in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue())
            p = tmp.name
        try:
            docs.extend(PyPDFLoader(p).load())
        finally:
            os.remove(p)
    return docs

# ── Render: metrics ──────────────────────────────────────────────────────────
def render_metrics(metrics, strategy):
    label = strategy.split(" ", 1)[1] if " " in strategy else strategy
    fp = int(metrics.faithfulness * 100)
    sp = int(metrics.mean_score * 100)
    scores = " ".join('<span class="metric-chip">' + "{:.2f}".format(s) + '</span>' for s in metrics.retrieval_scores)
    pages = (" ".join('<span class="metric-chip">p' + str(p) + '</span>' for p in metrics.source_pages)
             if metrics.source_pages else '<span class="metric-chip">—</span>')
    html = (
        '<div class="metric-box"><span class="strategy-badge">' + label + '</span>'
        '<div class="metric-row" style="margin-top:6px;">'
        '<div><div class="metric-label">⏱ Latency</div><strong>' + "{:.0f}".format(metrics.latency_ms) + ' ms</strong></div>'
        '<div><div class="metric-label">📦 Chunks</div><strong>' + str(metrics.chunks_used) + '</strong></div>'
        '<div><div class="metric-label">🎯 Retrieval Score</div><strong>' + str(sp) + '%</strong></div>'
        '<div><div class="metric-label">✅ Faithfulness</div><strong>' + str(fp) + '%</strong></div>'
        '</div><div style="margin-top:8px;"><div class="metric-label">📊 Per-chunk scores</div>'
        '<div class="metric-row">' + scores + '</div></div>'
        '<div style="margin-top:8px;"><div class="metric-label">📄 Source pages</div>'
        '<div class="metric-row">' + pages + '</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ── Render: citations ────────────────────────────────────────────────────────
def render_citations(source_docs):
    if not source_docs: return
    with st.expander("📎 Source Citations (" + str(len(source_docs)) + " chunks)", expanded=False):
        for i, doc in enumerate(source_docs):
            page = doc.metadata.get("page", "?")
            if isinstance(page, int): page = page + 1
            snippet = doc.page_content[:300].replace(chr(10), " ").strip()
            if len(doc.page_content) > 300: snippet += "…"
            html = (
                '<div class="citation-box"><strong>Chunk ' + str(i+1) + '</strong>'
                ' <span style="color:#a6adc8;font-size:0.76rem;">Page ' + str(page) + '</span>'
                '<div style="margin-top:4px;color:#bac2de;">' + snippet + '</div></div>'
            )
            st.markdown(html, unsafe_allow_html=True)

# ── Render: doc analysis ─────────────────────────────────────────────────────
def render_doc_analysis(a):
    if not a: return
    stats = a.get("stats", {})
    topics_html = " ".join('<span class="topic-tag">' + t + '</span>' for t in a.get("topics", []))
    facts = a.get("key_facts", [])
    html = (
        '<div class="doc-analysis-box">'
        '<div style="font-size:0.85rem;color:#a6adc8;margin-bottom:8px;">📋 Document Intelligence</div>'
        '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px;">'
        '<div><div style="font-size:0.75rem;color:#a6adc8;">📄 Pages</div><strong>' + str(stats.get("pages","—")) + '</strong></div>'
        '<div><div style="font-size:0.75rem;color:#a6adc8;">📝 Words</div><strong>' + "{:,}".format(stats.get("words",0)) + '</strong></div>'
        '<div><div style="font-size:0.75rem;color:#a6adc8;">🧩 Chunks</div><strong>' + str(stats.get("chunks","—")) + '</strong></div>'
        '<div><div style="font-size:0.75rem;color:#a6adc8;">🏷️ Type</div><strong>' + a.get("doc_type","") + '</strong></div>'
        '</div><div style="margin-bottom:8px;font-size:0.87rem;color:#cdd6f4;">' + a.get("summary","") + '</div>'
        '<div style="margin-bottom:6px;">' + topics_html + '</div>'
    )
    if facts:
        html += '<div style="font-size:0.78rem;color:#a6adc8;margin-top:8px;">Key facts: ' + " | ".join(facts) + '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Analytics ────────────────────────────────────────────────────────────────
def update_analytics(strategy, latency_ms, faithfulness):
    a = st.session_state.session_analytics.get(strategy)
    if not a: return
    a["q"] += 1
    a["lat"] += latency_ms
    a["faith"] += faithfulness

def render_analytics():
    active = {s: a for s, a in st.session_state.session_analytics.items() if a["q"] > 0}
    if not active:
        st.caption("No queries yet.")
        return
    rows = [(s.split(" ",1)[1] if " " in s else s, a["q"],
             a["lat"]/a["q"], a["faith"]/a["q"]*100) for s, a in active.items()]
    rows.sort(key=lambda x: x[3], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, (label, q, avg_lat, avg_faith) in enumerate(rows):
        medal = medals[i] if i < 3 else "  "
        st.markdown(
            medal + " **" + label + "**  " + chr(10)
            + "Queries: `" + str(q) + "` | Avg lat: `"
            + "{:.0f}".format(avg_lat) + "ms` | Faith: `"
            + "{:.0f}".format(avg_faith) + "%`"
        )

def export_chat_markdown():
    out = ["# ChatPDF-RAG v2 — Export", "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"), ""]
    for s in STRATEGIES:
        msgs = st.session_state.chat_messages.get(s, [])
        if not msgs: continue
        out.append("## " + s)
        out.append("")
        for msg in msgs:
            role = "**You**" if msg["role"] == "user" else "**Assistant**"
            out.append(role + ": " + msg["content"])
            if msg["role"] == "assistant" and msg.get("metrics"):
                m = msg["metrics"]
                out.append("> Lat: " + str(round(m.get("latency_ms",0))) + "ms | "
                           + "Faith: " + str(round(m.get("faithfulness",0)*100)) + "% | "
                           + "Chunks: " + str(m.get("chunks_used",0)))
            out.append("")
    return chr(10).join(out)

# ── Core query runner ────────────────────────────────────────────────────────
def run_query(strategy, user_input, session_id, active_keys, return_result=False):
    chain = st.session_state.rag_chains.get(strategy)
    if chain is None:
        if not return_result: st.error("Chain not built.")
        return None, [], {}
    start_time = time.time()
    streamed_answer = ""
    source_docs = []
    is_self_rag = isinstance(chain, SelfRAGChain)
    placeholder = st.empty()
    try:
        for chunk in chain.stream({"input": user_input},
                                   config={"configurable": {"session_id": strategy + "_" + session_id}}):
            if "answer" in chunk:
                streamed_answer += chunk["answer"]
                placeholder.markdown(streamed_answer + ("" if is_self_rag else "▌"))
            if "context" in chunk:
                source_docs = chunk["context"]
    except RateLimitError:
        rotate_groq_key(active_keys)
        try:
            result = chain.invoke({"input": user_input},
                                   config={"configurable": {"session_id": strategy + "_" + session_id}})
            streamed_answer = result.get("answer", "")
            source_docs = result.get("context", [])
        except Exception as e:
            streamed_answer = "❌ Keys exhausted: " + str(e)
    except Exception as e:
        streamed_answer = "❌ Error: " + str(e)
    placeholder.markdown(streamed_answer)
    if strategy == STRATEGIES[0] and source_docs:
        source_docs = rerank_documents(user_input, source_docs, top_k=4)
    metrics_obj = None
    metrics_dict = {}
    if not is_self_rag or source_docs:
        metrics_obj = evaluate(query=user_input, answer=streamed_answer,
                               source_docs=source_docs, embeddings=load_embeddings(),
                               start_time=start_time)
        metrics_dict = metrics_obj.as_dict()
        render_metrics(metrics_obj, strategy)
        update_analytics(strategy, metrics_obj.latency_ms, metrics_obj.faithfulness)
    render_citations(source_docs)
    if not return_result:
        st.session_state.chat_messages[strategy].append({
            "role": "assistant", "content": streamed_answer,
            "metrics": metrics_dict, "sources": [d.page_content[:200] for d in source_docs],
        })
    return streamed_answer, source_docs, metrics_dict

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 ChatPDF-RAG v2")
    st.caption("5-strategy RAG + Document Intelligence")
    st.divider()
    st.markdown("### 🔑 API Keys")
    if GROQ_KEYS:
        st.success("✅ " + str(len(GROQ_KEYS)) + " Groq key(s) loaded.")
        st.caption("Auto-rotate on rate-limit enabled.")
        manual_keys_input = None
    else:
        manual_keys_input = st.text_area("Groq API Key(s)", placeholder="gsk_key1", height=90)
    st.divider()
    session_id = st.text_input("🆔 Session ID", value="default_session")
    st.divider()
    st.markdown("### 📂 Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf",
                                       accept_multiple_files=True, label_visibility="collapsed")
    process_btn = st.button("⚡ Process PDFs", use_container_width=True, type="primary")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chats", use_container_width=True):
            st.session_state.chat_messages = {s: [] for s in ALL_TABS}
            st.session_state.store = {}
            st.session_state.session_analytics = {s: {"q":0,"lat":0.0,"faith":0.0} for s in STRATEGIES}
            st.rerun()
    with col_b:
        st.download_button("💾 Export", data=export_chat_markdown(),
                           file_name="chatpdf_export.md", mime="text/markdown",
                           use_container_width=True)
    st.divider()
    st.markdown("### 📊 Session Analytics")
    render_analytics()
    st.divider()
    if st.session_state.tree_stats:
        ts = st.session_state.tree_stats
        st.markdown("### 🌲 Tree Index")
        st.markdown("Leaves: **" + str(ts["leaf_count"]) + "** | Branches: **"
                    + str(ts["branch_count"]) + "** | Nodes: **" + str(ts["total_nodes"]) + "**")
    active_idx = st.session_state.get("groq_key_index", 0)
    if GROQ_KEYS:
        st.caption("Active key: #" + str(active_idx+1) + "/" + str(len(GROQ_KEYS)))

# ── Resolve keys ─────────────────────────────────────────────────────────────
if GROQ_KEYS:
    active_keys = GROQ_KEYS
elif manual_keys_input:
    active_keys = [k.strip() for k in manual_keys_input.strip().splitlines() if k.strip()]
else:
    active_keys = []
if not active_keys:
    st.warning("⚠️ Please provide at least one Groq API key.")
    st.stop()
llm = llm_with_fallback(active_keys)

# ── PDF Processing ────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        file_ids = {f.name for f in uploaded_files}
        if file_ids != st.session_state.processed_files:
            with st.spinner("📄 Loading and splitting PDFs…"):
                docs = load_pdfs(uploaded_files)
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
                splits = splitter.split_documents(docs)
                st.session_state.splits = splits
            with st.spinner("🔍 Analysing document…"):
                st.session_state.doc_analysis = analyze_document(splits, llm)
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)
            with col1:
                with st.spinner("⚡ Vector RAG…"):
                    chain_v, _ = build_vector_rag(splits, llm, load_embeddings(), get_session_history)
                    st.session_state.rag_chains[STRATEGIES[0]] = chain_v
                st.success("✅ Vector RAG")
            with col2:
                with st.spinner("🔍 VectorLess RAG…"):
                    chain_b = build_bm25_rag(splits, llm, get_session_history)
                    st.session_state.rag_chains[STRATEGIES[1]] = chain_b
                st.success("✅ VectorLess RAG")
            with col3:
                chain_t, tree_stats = build_tree_rag(splits, llm, load_embeddings(), get_session_history)
                st.session_state.rag_chains[STRATEGIES[2]] = chain_t
                st.session_state.tree_stats = tree_stats
                st.success("✅ Tree RAG")
            with col4:
                with st.spinner("💡 HyDE RAG…"):
                    chain_h, _ = build_hyde_rag(splits, llm, load_embeddings(), get_session_history)
                    st.session_state.rag_chains[STRATEGIES[3]] = chain_h
                st.success("✅ HyDE RAG")
            with col5:
                with st.spinner("🤔 Self-RAG…"):
                    chain_s = build_self_rag(splits, llm, load_embeddings(), st.session_state.store)
                    st.session_state.rag_chains[STRATEGIES[4]] = chain_s
                st.success("✅ Self-RAG")
            st.session_state.processed_files = file_ids
            st.session_state.chat_messages = {s: [] for s in ALL_TABS}
            st.session_state.store = {}
            st.sidebar.success("✅ " + str(len(uploaded_files)) + " PDF(s) ready!")
        else:
            st.sidebar.info("PDFs already loaded.")

# ── Welcome screen ───────────────────────────────────────────────────────────
if not st.session_state.rag_chains:
    st.markdown("""
## 👋 Welcome to ChatPDF-RAG v2

Upload your PDFs in the sidebar and click **⚡ Process PDFs**.

| Tab | Strategy | Best For |
|---|---|---|
| ⚡ Vector RAG | Hybrid dense+BM25 + cross-encoder | General Q&A |
| 🔍 VectorLess RAG | Pure BM25 + LLM filter | Exact terms, legal/medical |
| 🌲 Tree RAG | RAPTOR hierarchical index | Long docs, summarization |
| 💡 HyDE RAG | Hypothetical Document Embeddings | Vague/conceptual questions |
| 🤔 Self-RAG | Adaptive retrieval + self-critique | Trust & explainability |
| ⚖️ Compare | All 5 strategies side-by-side | Research & benchmarking |
""")
    st.stop()

# ── Doc Analysis Banner ───────────────────────────────────────────────────────
if st.session_state.doc_analysis:
    render_doc_analysis(st.session_state.doc_analysis)

st.title("📄 ChatPDF-RAG v2")
tabs = st.tabs(ALL_TABS)

# ── Tabs 1-5 ─────────────────────────────────────────────────────────────────
for tab, strategy in zip(tabs[:5], STRATEGIES):
    with tab:
        if strategy not in st.session_state.rag_chains:
            st.info("Process your PDFs to activate " + strategy + ".")
            continue
        for msg in st.session_state.chat_messages[strategy]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("metrics"):
                    m = msg["metrics"]
                    render_metrics(RAGMetrics(
                        retrieval_scores=m.get("retrieval_scores", []),
                        mean_score=m.get("mean_score", 0.0),
                        faithfulness=m.get("faithfulness", 0.0),
                        latency_ms=m.get("latency_ms", 0.0),
                        chunks_used=m.get("chunks_used", 0),
                        source_pages=m.get("source_pages", []),
                    ), strategy)
                    if msg.get("sources"):
                        with st.expander("📎 Cached Sources", expanded=False):
                            for i, snip in enumerate(msg["sources"]):
                                st.markdown('<div class="citation-box">Chunk ' + str(i+1) + ': ' + snip + '…</div>',
                                            unsafe_allow_html=True)
        user_input = st.chat_input("Ask using " + strategy + "…", key="input_" + strategy)
        if user_input:
            st.session_state.chat_messages[strategy].append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)
            with st.chat_message("assistant"): run_query(strategy, user_input, session_id, active_keys)
            st.rerun()

# ── Tab 6: Compare ───────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### ⚖️ Strategy Comparison")
    st.caption("One question → all 5 strategies answer simultaneously. Compare latency, faithfulness, retrieval.")
    ready = [s for s in STRATEGIES if s in st.session_state.rag_chains]
    if len(ready) < 2:
        st.info("Process your PDFs first.")
    else:
        for msg in st.session_state.chat_messages[COMPARE_TAB]:
            if msg.get("role") != "compare_result": continue
            st.markdown("---")
            st.markdown("**❓ Question:** " + msg["question"])
            cols = st.columns(len(msg["results"]))
            for col, (strat, data) in zip(cols, msg["results"].items()):
                with col:
                    st.markdown("**" + strat + "**")
                    ans = data["answer"]
                    st.markdown(ans[:400] + ("…" if len(ans) > 400 else ""))
                    m = data.get("metrics", {})
                    if m:
                        st.caption("⏱ " + str(round(m.get("latency_ms",0))) + "ms | "
                                   + "✅ " + str(round(m.get("faithfulness",0)*100)) + "% | "
                                   + "📦 " + str(m.get("chunks_used",0)) + " chunks")
            if msg.get("leaderboard"):
                st.markdown("#### 📊 Head-to-Head")
                st.table(msg["leaderboard"])
        compare_input = st.chat_input("Ask to compare all strategies…", key="input_compare")
        if compare_input:
            st.markdown("---")
            st.markdown("**❓ Question:** " + compare_input)
            cols = st.columns(len(ready))
            compare_results = {}
            for col, strategy in zip(cols, ready):
                with col:
                    st.markdown("**" + strategy + "**")
                    answer, _, metrics_dict = run_query(
                        strategy, compare_input, session_id + "_cmp", active_keys, return_result=True)
                    compare_results[strategy] = {"answer": answer or "", "metrics": metrics_dict}
            leaderboard = []
            for strat, data in compare_results.items():
                m = data.get("metrics", {})
                leaderboard.append({
                    "Strategy": strat.split(" ",1)[1] if " " in strat else strat,
                    "Latency (ms)": round(m.get("latency_ms", 0)),
                    "Faithfulness": str(round(m.get("faithfulness",0)*100)) + "%",
                    "Retrieval Score": str(round(m.get("mean_score",0)*100)) + "%",
                    "Chunks": m.get("chunks_used", 0),
                })
            leaderboard.sort(key=lambda x: int(x["Faithfulness"].replace("%","")), reverse=True)
            st.markdown("#### 📊 Head-to-Head")
            st.table(leaderboard)
            st.session_state.chat_messages[COMPARE_TAB].append({
                "role": "compare_result", "question": compare_input,
                "results": compare_results, "leaderboard": leaderboard,
            })
            st.rerun()
