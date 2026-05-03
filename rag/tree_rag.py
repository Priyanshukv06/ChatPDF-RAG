"""
Strategy 3 — Tree Index RAG (Hierarchical Summarization)
  Inspired by RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

  Structure:
    Level 0 (leaves)  : original text chunks
    Level 1 (branches): LLM summaries of every 5 leaf chunks
    Level 2 (root)    : LLM summary of all branch summaries

  At query time, retrieves from ALL levels. Higher-level nodes answer
  "what is this document about" questions; lower-level nodes answer
  specific factual questions. Cross-encoder re-ranks across levels.

  Best for: long documents, summarization queries, big-picture questions.
"""
from __future__ import annotations

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import chromadb
from chromadb.config import Settings

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Summarize the following document excerpts into a single coherent paragraph. "
     "Preserve key facts, names, and numbers. Be concise.\n\n{context}"),
    ("human", "Provide the summary now."),
])

LEAF_BATCH = 5  # Number of leaf chunks grouped into one branch summary


def _summarize_batch(docs: list[Document], llm) -> str:
    """Summarize a batch of documents into one string via LLM."""
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = SUMMARIZE_PROMPT.format_messages(context=context)
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


def build_tree_rag(
    splits: list[Document],
    llm,
    embeddings,
    get_session_history,
    chroma_client_key: str = "chroma_client_tree",
) -> tuple[RunnableWithMessageHistory, dict]:
    progress = st.progress(0, text="🌲 Building tree index — Level 0 (leaves)…")

    # ── Level 0: original chunks ───────────────────────────────────────────
    leaf_docs = [
        Document(
            page_content=d.page_content,
            metadata={**d.metadata, "tree_level": 0},
        )
        for d in splits
    ]
    progress.progress(20, text="🌲 Building Level 1 summaries (branches)…")

    # ── Level 1: summarize every LEAF_BATCH leaves ─────────────────────────
    branch_docs = []
    for i in range(0, len(leaf_docs), LEAF_BATCH):
        batch = leaf_docs[i : i + LEAF_BATCH]
        summary_text = _summarize_batch(batch, llm)
        pages = [d.metadata.get("page", 0) for d in batch]
        branch_docs.append(Document(
            page_content=summary_text,
            metadata={"tree_level": 1, "source_pages": str(pages), "page": pages[0]},
        ))

    progress.progress(60, text="🌲 Building Level 2 root summary…")

    # ── Level 2: summarize all branches into root ──────────────────────────
    root_text = _summarize_batch(branch_docs, llm)
    root_doc = Document(
        page_content=root_text,
        metadata={"tree_level": 2, "page": 0},
    )

    progress.progress(80, text="🌲 Indexing all levels into vector store…")

    # ── Index all levels together ──────────────────────────────────────────
    all_docs = leaf_docs + branch_docs + [root_doc]

    import streamlit as _st
    _st.session_state[chroma_client_key] = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        client=_st.session_state[chroma_client_key],
        collection_name="tree_rag",
    )

    # Retriever fetches across all levels — cross-encoder will re-rank
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    progress.progress(100, text="✅ Tree index ready!")
    progress.empty()

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
         "You are an AI assistant with access to a hierarchically indexed document. "
         "The context contains both detailed chunks (level 0) and summaries (levels 1-2). "
         "Use the most relevant level to answer accurately. "
         "If not in context, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    tree_stats = {
        "leaf_count": len(leaf_docs),
        "branch_count": len(branch_docs),
        "root_count": 1,
        "total_nodes": len(all_docs),
    }

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain, tree_stats