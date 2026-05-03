"""
ChatPDF-RAG — Production-ready Conversational RAG with Streamlit Community Cloud
Python 3.10+ | LangChain v0.3+ | Groq (Gemma2-9b-It) | In-memory ChromaDB
"""

import os
import tempfile

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatPDF-RAG",
    page_icon="📄",
    layout="centered",
)

# ── Secrets / env resolution ─────────────────────────────────────────────────
def _get_secret(key: str) -> str | None:
    """Read from st.secrets first (Cloud), fall back to env var (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)


GROQ_API_KEY: str | None = _get_secret("GROQ_API_KEY")
HF_TOKEN: str | None = _get_secret("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# ── Cached resources (loaded once per session) ────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading embedding model (once per session)…")
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Session-state initialisation ─────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store: dict[str, ChatMessageHistory] = {}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_messages" not in st.session_state:
    # List of {"role": "user"|"assistant", "content": str}
    st.session_state.chat_messages: list[dict] = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files: set[str] = set()


# ── Helper: session history ───────────────────────────────────────────────────
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# ── Helper: build RAG chain from documents ────────────────────────────────────
def build_rag_chain(documents, llm: ChatGroq) -> RunnableWithMessageHistory:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)

    # In-memory ChromaDB — no persist_directory (ephemeral, Cloud-safe)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=load_embeddings(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ── Prompt: contextualise question against history ────────────────────────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer it — "
            "just reformulate if needed; otherwise return it as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ── Prompt: answer using retrieved context ────────────────────────────────
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant for question-answering tasks. "
            "Use only the retrieved context below to answer. "
            "If the answer is not in the context, say you don't know.\n\n"
            "{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Helper: load PDFs from uploaded files ─────────────────────────────────────
def load_pdfs(uploaded_files) -> list:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix="chatpdf_"
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        finally:
            os.remove(tmp_path)  # always clean up
    return documents


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📄 ChatPDF-RAG")
st.caption("Upload PDFs and chat with their content using Groq + LangChain.")

# ── Sidebar: config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Groq API key: env/secret → pre-filled; otherwise user types it
    groq_key_input = st.text_input(
        "🔑 Groq API Key",
        value=GROQ_API_KEY or "",
        type="password",
        help="Set GROQ_API_KEY in .streamlit/secrets.toml or as an env var.",
    )
    session_id = st.text_input(
        "🆔 Session ID",
        value="default_session",
        help="Use different IDs to keep separate conversation histories.",
    )
    st.divider()
    uploaded_files = st.file_uploader(
        "📂 Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.store = {}
        st.rerun()

# ── Guard: API key ─────────────────────────────────────────────────────────────
active_groq_key = groq_key_input.strip() or GROQ_API_KEY
if not active_groq_key:
    st.warning("⚠️ Please enter your **Groq API key** in the sidebar to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=active_groq_key, model_name="Gemma2-9b-It")

# ── PDF processing ─────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
    else:
        file_ids = {f.name for f in uploaded_files}
        if file_ids != st.session_state.processed_files:
            with st.spinner("🔍 Processing PDFs — splitting & embedding…"):
                docs = load_pdfs(uploaded_files)
                st.session_state.rag_chain = build_rag_chain(docs, llm)
                st.session_state.processed_files = file_ids
                st.session_state.chat_messages = []
                st.session_state.store = {}
            st.sidebar.success(
                f"✅ {len(uploaded_files)} PDF(s) processed! Start chatting."
            )
        else:
            st.sidebar.info("These PDFs are already loaded.")

# ── Chat UI ────────────────────────────────────────────────────────────────────
if st.session_state.rag_chain is None:
    st.info("👈 Upload PDF(s) in the sidebar and click **Process PDFs** to begin.")
    st.stop()

# Render existing conversation
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
user_input = st.chat_input("Ask a question about your PDFs…")
if user_input:
    # Display user message immediately
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        streamed_answer = ""

        # Stream token-by-token via .stream()
        for chunk in st.session_state.rag_chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        ):
            # create_retrieval_chain yields dicts; answer is in "answer" key
            if "answer" in chunk:
                streamed_answer += chunk["answer"]
                response_placeholder.markdown(streamed_answer + "▌")

        response_placeholder.markdown(streamed_answer)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": streamed_answer}
    )
