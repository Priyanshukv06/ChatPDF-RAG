# 📄 ChatPDF-RAG v2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2.17-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F54E27?style=for-the-badge)

**An industry-grade Conversational RAG platform implementing 5 distinct retrieval architectures with real-time evaluation metrics, document intelligence, and strategy benchmarking.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-chatpdf--rag--06.streamlit.app-FF4B4B?style=for-the-badge)](https://chatpdf-rag-06.streamlit.app/)

</div>

---

## 🏗️ Architecture Overview

ChatPDF-RAG v2 is not a single RAG implementation — it is a **RAG research platform** that lets you compare 5 fundamentally different retrieval strategies on the same document, side by side, with quantitative evaluation metrics per response.

```
PDF Upload → Chunking (600 tokens, 80 overlap)
                │
    ┌───────────┼───────────┬───────────┬───────────┐
    ▼           ▼           ▼           ▼           ▼
⚡ Vector    🔍 BM25     🌲 Tree     💡 HyDE     🤔 Self
  RAG          RAG         RAG         RAG         RAG
    │           │           │           │           │
    └───────────┴───────────┴───────────┴───────────┘
                            │
                    ⚖️ Compare Tab
              (Side-by-side benchmarking)
                            │
                  📊 Evaluation Metrics
          (Latency | Faithfulness | Retrieval Score)
```

---

## 🔬 RAG Strategies

### ⚡ Tab 1 — Vector RAG (Hybrid + Reranking)
The gold-standard dense retrieval pipeline.

- **Hybrid retrieval** — Combines dense (ChromaDB cosine similarity) and sparse (BM25) retrieval using `EnsembleRetriever` with weighted fusion
- **MultiQuery expansion** — LLM generates 3 query variants to improve recall across different phrasings
- **Cross-encoder reranking** — Retrieved candidates are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision
- **Contextual compression** — `LLMChainFilter` removes chunks irrelevant to the query before generation
- **History-aware** — Reformulates follow-up questions into standalone queries using chat history

### 🔍 Tab 2 — VectorLess RAG (Pure BM25)
Zero embeddings — keyword-only retrieval.

- **BM25 retrieval** — Okapi BM25 probabilistic ranking over raw token frequencies
- **LLM Contextual Compression** — `LLMChainFilter` grades each chunk for relevance post-retrieval
- **No vector operations** — Entire pipeline runs without a single embedding model call
- **Best for** — Legal documents, medical reports, technical specs with precise terminology

### 🌲 Tab 3 — Tree RAG (RAPTOR-style)
Hierarchical summarization index for long-document understanding.

- **Level 0 (Leaves)** — Raw text chunks from the PDF
- **Level 1 (Branches)** — LLM-generated summaries of every 5 leaf chunks
- **Level 2 (Root)** — Single summary of all branch summaries
- **Multi-level retrieval** — Retrieves from all levels simultaneously, enabling both specific and high-level answers
- **Best for** — Summarization, cross-section questions, long documents

### 💡 Tab 4 — HyDE RAG (Hypothetical Document Embeddings)
Closes the semantic gap between short questions and long document sentences.

> **Paper**: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (Gao et al., 2022)

- **Hypothetical generation** — LLM generates a fake ideal answer to the query
- **Embed the hypothesis** — The fake answer (not the raw question) is embedded and used for retrieval
- **Why it works** — `"What is LSTM?"` is semantically far from `"LSTM was introduced in 1997 by Hochreiter..."`. A hypothetical answer bridges this gap
- **Best for** — Vague questions, conceptual queries, subject-matter deep dives

### 🤔 Tab 5 — Self-RAG (Self-Reflective RAG)
The only strategy that reasons about its own retrieval quality.

> **Paper**: [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) (Asai et al., 2023)

The pipeline executes **4 visible reasoning steps** in real time:

| Step | Token | Decision |
|------|-------|----------|
| 1 | `RETRIEVE` | Does this question need document retrieval at all? |
| 2 | `ISREL` | Is each retrieved chunk relevant to the question? |
| 3 | Generate | Answer from relevant chunks only |
| 4 | `ISSUP` | Is the answer actually supported by the context? |

Each response is tagged: `🟢 SUPPORTED` / `🟡 PARTIALLY SUPPORTED` / `🔴 NOT SUPPORTED`

### ⚖️ Tab 6 — Strategy Comparison
One question → all 5 strategies answer simultaneously in parallel columns with a live head-to-head leaderboard sorted by faithfulness score.

---

## 📊 Evaluation Metrics

Every response (except direct Self-RAG answers) includes a real-time evaluation panel:

| Metric | How it's computed |
|--------|------------------|
| **Latency** | Wall-clock time from query to final token |
| **Chunks Used** | Number of document chunks passed to the LLM |
| **Mean Retrieval Score** | Average cosine similarity between query embedding and retrieved chunk embeddings |
| **Per-chunk Scores** | Individual cosine score for each retrieved chunk |
| **Faithfulness** | Cosine similarity between the final answer embedding and the mean context embedding |
| **Source Pages** | Page numbers of all retrieved chunks |

---

## 🧠 Document Intelligence

After PDF processing, the app automatically generates a **Document Intelligence card** showing:

- 📄 Page count, 📝 word count, 🧩 chunk count
- 🏷️ Detected document type (Technical Report / Research Paper / etc.)
- One-paragraph auto-summary of the entire document
- Topic tags extracted by the LLM
- 3 key facts / figures from the document

---

## 📋 Session Analytics

The sidebar tracks a **live leaderboard** across your session:
- 🥇🥈🥉 Strategies ranked by average faithfulness score
- Average latency per strategy
- Total queries per strategy

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit 1.57.0 |
| **LLM** | Groq API — `llama-3.1-8b-instant` |
| **Orchestration** | LangChain 1.x (`langchain-classic==1.0.4`) |
| **Vector Store** | ChromaDB 1.5.8 (EphemeralClient — in-memory) |
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Sparse Retrieval** | BM25 via `rank-bm25` |
| **PDF Loading** | `pypdf` |
| **Key Rotation** | Automatic Groq key rotation on `RateLimitError` |

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/chatpdf-rag.git
cd chatpdf-rag
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure secrets
Create `.streamlit/secrets.toml` (never commit this file):
```toml
GROQ_API_KEY = "gsk_your_key_here"
HF_TOKEN = "hf_your_token_here"
```

Or use a `.env` file:
```env
GROQ_API_KEY=gsk_your_key_here
HF_TOKEN=hf_your_token_here
```

### 5. Run the app
```bash
streamlit run conversationalqna.py
```

---

## 🌐 Deployment on Streamlit Cloud

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, and set `conversationalqna.py` as the main file
4. Under **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "gsk_key1,gsk_key2"
HF_TOKEN = "hf_your_token"
```
5. Deploy — the app will be live at your Streamlit URL

> **Tip**: Add multiple Groq API keys separated by commas for automatic rate-limit rotation.

---

## 📁 Project Structure

```
chatpdf-rag/
├── conversationalqna.py        # Main Streamlit app — all 6 tabs
├── requirements.txt            # Pinned dependencies
├── .gitignore
├── .streamlit/
│   ├── config.toml             # Server config (fileWatcherType=poll)
│   └── secrets.toml            # Local secrets (gitignored)
└── rag/
    ├── __init__.py
    ├── analyzer.py             # Document Intelligence — LLM-based doc analysis
    ├── evaluator.py            # Per-response metrics (faithfulness, retrieval score)
    ├── vector_rag.py           # Hybrid + MultiQuery + Cross-encoder reranking
    ├── bm25_rag.py             # Pure BM25 + LLM contextual compression
    ├── tree_rag.py             # RAPTOR-style hierarchical tree index
    ├── hyde_rag.py             # Hypothetical Document Embeddings
    └── self_rag.py             # Self-reflective RAG with 4-step reasoning
```

---

## 🔑 API Keys Required

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| [Groq](https://console.groq.com) | LLM inference (LLaMA 3.1 8B) | ✅ Free — 6000 TPM |
| [Hugging Face](https://huggingface.co/settings/tokens) | Embedding model download | ✅ Free |

> **Rate limit tip**: Groq free tier allows 6000 tokens/minute. Add 2-3 keys for uninterrupted use on larger PDFs.

---

## ⚠️ Known Limitations

- **Tree RAG** on PDFs > 50 pages may hit Groq's TPM limit during tree construction — use with smaller PDFs or upgrade to Groq Dev tier
- **Streamlit Cloud free tier** has ~1 GB RAM — embedding model + 5 vector stores use ~600 MB peak
- **No persistent storage** — all indexes are rebuilt on every PDF upload (by design, using ChromaDB EphemeralClient)
- **Cold start** — first visitor after inactivity waits ~30s for Streamlit Cloud to wake the app

---

## 📚 References

| Paper | Authors | Year |
|-------|---------|------|
| [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | Gao et al. | 2022 |
| [Self-RAG: Learning to Retrieve, Generate and Critique](https://arxiv.org/abs/2310.11511) | Asai et al. | 2023 |
| [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | Sarthi et al. | 2024 |
| [RAG for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | Lewis et al. | 2020 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 |

---

<div align="center">

Built by **Priyanshu Verma**

[![Live Demo](https://img.shields.io/badge/🚀_Try_it_Live-chatpdf--rag--06.streamlit.app-FF4B4B?style=for-the-badge)](https://chatpdf-rag-06.streamlit.app/)

</div>
