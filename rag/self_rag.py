"""
Strategy 5 — Self-RAG (Self-Reflective Retrieval-Augmented Generation)
========================================================================
Paper: https://arxiv.org/abs/2310.11511

Unlike all other RAG strategies that always retrieve, Self-RAG is adaptive:

  Step 1 — RETRIEVE decision:
    LLM decides: "Do I need to retrieve context to answer this?"
    If NO → answers from parametric knowledge directly (fast path)
    If YES → retrieves chunks

  Step 2 — ISREL grading:
    For each retrieved chunk, LLM grades: "Is this chunk relevant?"
    Irrelevant chunks are discarded before generation

  Step 3 — Generate answer

  Step 4 — ISSUP critique:
    LLM critiques its own answer: "Is my answer supported by the context?"
    Adds a transparency flag to the response: SUPPORTED / PARTIALLY / NOT SUPPORTED

This makes the system self-aware about its own retrieval quality.
Best for: showing LLM reasoning, trust & explainability demos.
"""
from __future__ import annotations

import re
import streamlit as st

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import chromadb
from chromadb.config import Settings

# ── Grading prompts ────────────────────────────────────────────────────────────

RETRIEVE_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are deciding whether a question requires retrieving external documents to answer. "
     "Answer with ONLY one word: YES or NO.\n"
     "Answer YES if the question asks about specific facts, documents, data, or content "
     "that would be in an uploaded PDF.\n"
     "Answer NO if the question is a general greeting, asks about you, or is answerable "
     "from general world knowledge without any document."),
    ("human", "Question: {question}\nRetrieve documents? Answer YES or NO:"),
])

RELEVANCE_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading whether a retrieved document chunk is relevant to a question. "
     "Answer with ONLY one word: RELEVANT or IRRELEVANT."),
    ("human",
     "Question: {question}\n\nDocument chunk:\n{chunk}\n\n"
     "Is this chunk relevant? Answer RELEVANT or IRRELEVANT:"),
])

SUPPORT_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are critiquing whether an answer is supported by the provided context. "
     "Answer with ONLY one of: SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED."),
    ("human",
     "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
     "Is the answer supported by the context? "
     "Answer SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED:"),
])

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Answer the question directly from your knowledge. "
     "Be concise and accurate."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise AI assistant. Use ONLY the context below to answer. "
     "If the answer is not in context, say you don\'t know.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, "
     "formulate a standalone question. Do NOT answer it — "
     "just reformulate if needed; otherwise return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def _parse_yes_no(text: str) -> bool:
    return "yes" in text.strip().lower()


def _parse_relevance(text: str) -> bool:
    return "irrelevant" not in text.strip().lower()


def _parse_support(text: str) -> str:
    t = text.strip().upper()
    if "NOT_SUPPORTED" in t or "NOT SUPPORTED" in t:
        return "🔴 NOT SUPPORTED"
    elif "PARTIALLY" in t:
        return "🟡 PARTIALLY SUPPORTED"
    else:
        return "🟢 SUPPORTED"


class SelfRAGChain:
    """
    Self-RAG pipeline with full reflection loop.
    Implements the 4-step adaptive retrieval + self-critique pattern.
    """

    def __init__(self, retriever, llm, store: dict):
        self.retriever = retriever
        self.llm = llm
        self.store = store

    def _get_history(self, session_id: str) -> list:
        hist = self.store.get(session_id)
        if hist is None:
            return []
        return hist.messages

    def stream(self, inputs: dict, config: dict):
        """
        Streams self-RAG reasoning steps as structured chunks.
        Yields dicts with 'answer' key for compatibility with main app streamer.
        """
        question = inputs["input"]
        session_id = config.get("configurable", {}).get("session_id", "default")
        chat_history = self._get_history(session_id)

        # ── Step 1: Retrieve decision ──────────────────────────────────────
        yield {"answer": "🤔 **[Self-RAG]** Deciding whether retrieval is needed…\n\n"}

        decision_chain = RETRIEVE_DECISION_PROMPT | self.llm
        decision_result = decision_chain.invoke({"question": question})
        needs_retrieval = _parse_yes_no(decision_result.content)

        if not needs_retrieval:
            yield {"answer": "💡 **[Self-RAG]** No retrieval needed — answering from general knowledge.\n\n"}
            direct_chain = DIRECT_ANSWER_PROMPT | self.llm
            direct_result = direct_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            answer = direct_result.content
            yield {"answer": answer}
            yield {"answer": "\n\n---\n🏷️ **Retrieval:** Skipped | **Support:** N/A (general knowledge)"}
            self._save_history(session_id, question, answer)
            return

        # ── Step 2: Retrieve + grade relevance ────────────────────────────
        yield {"answer": "📥 **[Self-RAG]** Retrieving documents…\n\n"}

        # Contextualize question with history
        if chat_history:
            ctx_chain = CONTEXTUALIZE_PROMPT | self.llm
            ctx_result = ctx_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            standalone_q = ctx_result.content.strip()
        else:
            standalone_q = question

        raw_docs = self.retriever.invoke(standalone_q)

        yield {"answer": f"🔍 **[Self-RAG]** Grading {len(raw_docs)} retrieved chunks for relevance…\n\n"}

        grade_chain = RELEVANCE_GRADE_PROMPT | self.llm
        relevant_docs = []
        for doc in raw_docs:
            grade = grade_chain.invoke({
                "question": standalone_q,
                "chunk": doc.page_content[:500],
            })
            if _parse_relevance(grade.content):
                relevant_docs.append(doc)

        yield {"answer": f"✅ **[Self-RAG]** {len(relevant_docs)}/{len(raw_docs)} chunks passed relevance grading.\n\n"}

        if not relevant_docs:
            yield {"answer": "⚠️ **[Self-RAG]** No relevant chunks found. Answering from general knowledge.\n\n"}
            direct_chain = DIRECT_ANSWER_PROMPT | self.llm
            direct_result = direct_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            answer = direct_result.content
            yield {"answer": answer}
            yield {"answer": "\n\n---\n🏷️ **Retrieval:** No relevant chunks | **Support:** N/A"}
            self._save_history(session_id, question, answer)
            return

        # ── Step 3: Generate answer from relevant chunks ──────────────────
        yield {"answer": "✍️ **[Self-RAG]** Generating answer from relevant context…\n\n"}

        context_text = "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in relevant_docs
        )

        qa_chain = QA_PROMPT | self.llm
        qa_result = qa_chain.invoke({
            "input": question,
            "context": context_text,
            "chat_history": chat_history,
        })
        answer = qa_result.content

        # ── Step 4: Self-critique — is answer supported? ───────────────────
        yield {"answer": "🔎 **[Self-RAG]** Critiquing answer quality…\n\n"}

        critique_chain = SUPPORT_CRITIQUE_PROMPT | self.llm
        critique_result = critique_chain.invoke({
            "context": context_text[:1500],
            "answer": answer,
        })
        support_label = _parse_support(critique_result.content)

        # Yield final answer + critique badge
        yield {"answer": answer}
        yield {"answer": f"\n\n---\n🏷️ **Retrieval:** {len(relevant_docs)} chunks used | **Self-Critique:** {support_label}"}
        yield {"context": relevant_docs}

        self._save_history(session_id, question, answer)

    def _save_history(self, session_id: str, question: str, answer: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        self.store[session_id].add_user_message(question)
        self.store[session_id].add_ai_message(answer)


def build_self_rag(
    splits: list[Document],
    llm,
    embeddings,
    store: dict,
    chroma_client_key: str = "chroma_client_self_rag",
) -> SelfRAGChain:
    """
    Build Self-RAG chain.
    Returns SelfRAGChain which implements .stream() compatible with main app.
    """
    st.session_state[chroma_client_key] = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client=st.session_state[chroma_client_key],
        collection_name="self_rag",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return SelfRAGChain(retriever=retriever, llm=llm, store=store)
