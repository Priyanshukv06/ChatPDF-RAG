"""
Strategy 4 — HyDE RAG (Hypothetical Document Embeddings)
=========================================================
Paper: https://arxiv.org/abs/2212.10496

Instead of embedding the raw user question (short, vague),
the LLM first generates a *hypothetical ideal answer* to the question.
That fake answer is embedded and used for retrieval.

Why it works:
  A question like "What is LSTM?" sits far from document sentences like
  "LSTM was introduced by Hochreiter in 1997..." in embedding space.
  A hypothetical answer "LSTM is a type of RNN that uses gating mechanisms..."
  sits much closer. Closes the query-document semantic gap.

Stack:
  • LLM generates hypothetical answer via a prompt
  • HypotheticalDocumentEmbedder wraps base embeddings + LLM chain
  • Chroma EphemeralClient for in-memory storage
  • History-aware conversational chain
"""
from __future__ import annotations

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory

import chromadb
from chromadb.config import Settings


def build_hyde_rag(
    splits: list[Document],
    llm,
    embeddings,
    get_session_history,
    chroma_client_key: str = "chroma_client_hyde",
) -> tuple[RunnableWithMessageHistory, None]:
    """
    Build HyDE RAG chain.
    Uses HypotheticalDocumentEmbedder to embed hypothetical answers
    instead of raw queries for improved semantic retrieval.
    """

    # ── HyDE: hypothetical doc generation prompt ──────────────────────────
    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert assistant. Given a question, write a short paragraph "
         "(3-5 sentences) that would appear in a document answering this question. "
         "Write as if it is an excerpt from a factual document, not a direct answer."),
        ("human", "{question}"),
    ])

    # Build HyDE LLM chain: prompt | llm | string output
    hyde_llm_chain = hyde_prompt | llm | StrOutputParser()

    # Wrap base embeddings with HyDE
    hyde_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=hyde_llm_chain,
        base_embeddings=embeddings,
    )

    # ── Index documents with base embeddings (not HyDE — HyDE is query-side only) ──
    st.session_state[chroma_client_key] = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,          # index with normal embeddings
        client=st.session_state[chroma_client_key],
        collection_name="hyde_rag",
    )

    # ── Retriever uses HyDE embeddings at query time ──────────────────────
    # We build a custom retriever that uses hyde_embeddings.embed_query()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Override the retriever's embed_query by using a custom wrapper
    class HyDERetriever:
        """Thin wrapper: uses hyde_embeddings to embed query, base retriever to fetch."""
        def __init__(self, vectorstore, hyde_emb, k=4):
            self._vs = vectorstore
            self._hyde = hyde_emb
            self._k = k

        def invoke(self, query: str) -> list[Document]:
            hyde_vector = self._hyde.embed_query(query)
            return self._vs.similarity_search_by_vector(hyde_vector, k=self._k)

        def get_relevant_documents(self, query: str) -> list[Document]:
            return self.invoke(query)

        # Make it compatible with LangChain retriever interface
        def as_retriever(self):
            return self

    # Use vectorstore directly with HyDE-generated embeddings
    # The cleanest approach: use a standard retriever but with hyde_embeddings
    hyde_vectorstore = Chroma(
        client=st.session_state[chroma_client_key],
        collection_name="hyde_rag",
        embedding_function=hyde_embeddings,   # HyDE used at query time
    )
    retriever = hyde_vectorstore.as_retriever(search_kwargs={"k": 4})

    # ── History-aware chain ────────────────────────────────────────────────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, "
         "formulate a standalone question. Do NOT answer it — "
         "just reformulate if needed; otherwise return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise AI assistant using HyDE (Hypothetical Document Embedding) "
         "retrieval. Use ONLY the context below. Cite page numbers when possible. "
         "If the answer is not in context, say so.\'\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ), None
