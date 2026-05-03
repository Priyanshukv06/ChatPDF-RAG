"""
Evaluation metrics computed locally — no extra API calls.
Metrics:
  - retrieval_scores   : cosine similarity of each chunk to the query
  - mean_score         : average retrieval score
  - faithfulness       : token-level overlap of answer with context
  - latency_ms         : end-to-end response time in milliseconds
  - chunks_used        : number of source chunks retrieved
  - source_pages       : page numbers from chunk metadata
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from langchain_core.documents import Document


@dataclass
class RAGMetrics:
    retrieval_scores: List[float] = field(default_factory=list)
    mean_score: float = 0.0
    faithfulness: float = 0.0
    latency_ms: float = 0.0
    chunks_used: int = 0
    source_pages: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "retrieval_scores": [round(s, 3) for s in self.retrieval_scores],
            "mean_score": round(self.mean_score, 3),
            "faithfulness": round(self.faithfulness, 3),
            "latency_ms": round(self.latency_ms, 1),
            "chunks_used": self.chunks_used,
            "source_pages": self.source_pages,
        }


def compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate(
    query: str,
    answer: str,
    source_docs: List[Document],
    embeddings,
    start_time: float,
) -> RAGMetrics:
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # Retrieval scores — cosine similarity between query embedding and each chunk
    try:
        query_emb = np.array(embeddings.embed_query(query))
        doc_embs = np.array(embeddings.embed_documents([d.page_content for d in source_docs]))
        scores = [compute_cosine(query_emb, de) for de in doc_embs]
    except Exception:
        scores = [0.0] * len(source_docs)

    mean_score = float(np.mean(scores)) if scores else 0.0

    # Faithfulness — token overlap between answer and combined context
    context_tokens = set(
        " ".join(d.page_content for d in source_docs).lower().split()
    )
    answer_tokens = set(answer.lower().split())
    faithfulness = (
        len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    )

    # Source pages from metadata
    pages = []
    for doc in source_docs:
        p = doc.metadata.get("page")
        if p is not None:
            pages.append(int(p) + 1)  # 0-indexed → 1-indexed
    pages = sorted(set(pages))

    return RAGMetrics(
        retrieval_scores=scores,
        mean_score=mean_score,
        faithfulness=faithfulness,
        latency_ms=latency_ms,
        chunks_used=len(source_docs),
        source_pages=pages,
    )