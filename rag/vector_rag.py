"""
Strategy 1 — Enhanced Vector RAG
  • Hybrid retrieval: dense (Chroma) + sparse (BM25) via EnsembleRetriever
  • Cross-encoder re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2
  • MultiQueryRetriever for improved recall
  • History-aware conversational chain
"""
from __future__ import annotations

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import CrossEncoder

import chromadb
from chromadb.config import Settings

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner="⏳ Loading cross-encoder re-ranker…")
def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 4,
) -> list[Document]:
    """Re-rank docs using cross-encoder, return top_k."""
    if not docs:
        return docs
    cross_enc = load_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_enc.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def build_vector_rag(
    splits: list[Document],
    llm,
    embeddings,
    get_session_history,
    chroma_client_key: str = "chroma_client_vector",
) -> tuple[RunnableWithMessageHistory, Chroma]:
    # Fresh EphemeralClient stored in session_state to prevent SQLite GC
    import streamlit as st 
    st.session_state[chroma_client_key] = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client=st.session_state[chroma_client_key],
        collection_name="vector_rag",
    )

    # Dense retriever — fetch more candidates for re-ranker
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # BM25 sparse retriever
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 12

    # Ensemble: 60% dense, 40% BM25
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    # MultiQueryRetriever wraps ensemble for better recall
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    # History-aware wrapper
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, "
         "formulate a standalone question. Do NOT answer it — "
         "just reformulate if needed; otherwise return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, multi_query_retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise AI assistant. Use ONLY the context below to answer. "
         "Cite the page number when possible. If not in context, say you don't know.\n\n"
         "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain, vectorstore