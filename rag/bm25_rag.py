"""
Strategy 2 — VectorLess RAG (Pure BM25 + LLM Contextual Compression)
  • Zero embeddings at query time — pure keyword retrieval
  • ContextualCompressionRetriever with LLMChainFilter to keep only
    relevant chunks (the LLM itself judges relevance)
  • History-aware conversational chain
  • Best for: exact-term queries, legal/medical/technical documents
"""
from __future__ import annotations

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


def build_bm25_rag(
    splits: list[Document],
    llm,
    get_session_history,
) -> RunnableWithMessageHistory:
    # BM25 base retriever — fetch generous candidates
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10

    # LLM filter: keeps only chunks actually relevant to the query
    llm_filter = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=llm_filter,
        base_retriever=bm25_retriever,
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
        llm, compression_retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise AI assistant specializing in keyword-based document retrieval. "
         "Use ONLY the context below. If not in context, say you don't know.\n\n"
         "{context}"),
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
    )