<div align="center">

```
  ____                          ____      _    ____
 |  _ \ __ _ _ __   ___ _ __  |  _ \    / \  / ___|
 | |_) / _` | '_ \ / _ \ '__| | |_) |  / _ \| |  _
 |  __/ (_| | |_) |  __/ |    |  _ <  / ___ \ |_| |
 |_|   \__,_| .__/ \___|_|    |_| \_\/_/   \_\____|
             |_|
```

# Paper RAG

**A production-grade Retrieval-Augmented Generation system for academic research.**
Drop in your PDFs, ask questions in any language, get precise answers grounded in your papers.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)
![Jina](https://img.shields.io/badge/Jina-Reranker-E94B5C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

## Overview

Paper RAG is a local-first research assistant that lets you query a collection of academic PDFs using natural language. It combines **semantic vector search**, **BM25 keyword search**, and **neural reranking** into a hybrid retrieval pipeline — then feeds the most relevant context to GPT-4o-mini to produce grounded, cited answers.

Unlike generic RAG wrappers, this system makes deliberate retrieval architecture decisions tuned for scientific text: exact terminology matching via BM25, section-aware chunking that preserves paper structure, and optional GPT-4o Vision figure extraction that makes graphs and diagrams fully searchable.

Built for researchers and engineers who want precise answers from their own document collections without relying on third-party knowledge bases.

---

## Features

- **Hybrid Search** — Reciprocal Rank Fusion (RRF) over vector search and BM25, capturing both semantic meaning and exact scientific terminology in parallel
- **Neural Reranking** — Jina `rerank-multilingual-v2-base` as a final cross-encoder precision pass over retrieved candidates
- **Figure Extraction** — GPT-4o Vision describes figures, graphs, and diagrams; their descriptions are embedded and searchable alongside text
- **Section-aware Chunking** — Detects academic paper structure (Abstract, Methods, Results…) before splitting, preserving semantic boundaries
- **Multilingual** — Works with papers and queries in any language
- **Zero re-ingestion** — Papers already embedded are skipped automatically on subsequent runs
- **Dual interface** — Interactive terminal chat with streaming output, or REST API with `/ask` and `/ask/stream` endpoints
- **Local vector store** — ChromaDB runs in Docker, data persists in a named volume with no external dependencies

---

## Architecture

### System Layout

```
paper_rag/
├── chat.py              # Interactive terminal chat (entry point)
├── ingest_pdfs.py       # Bulk PDF ingestion script
├── debug_retrieve.py    # Pipeline inspection tool
├── app/
│   ├── api.py           # Flask REST API (POST /ask, POST /ask/stream)
│   ├── ask.py           # Retrieval → prompt → LLM orchestration
│   ├── retrieve.py      # Hybrid search pipeline
│   ├── ingest.py        # PDF extraction + figure extraction + embedding + storage
│   ├── chunking.py      # Section-aware text splitting
│   ├── prompts.py       # System prompt + RAG prompt builder
│   ├── models.py        # Pydantic v2 schemas
│   ├── env.py           # dotenv loader
│   └── logging_config.py
└── data/
    ├── pdfs/            # Place your PDF papers here
    └── processed/
        └── metadata/    # Per-paper ingestion records (duplicate guard)
```

### Ingestion Pipeline

```
data/pdfs/*.pdf
    │
    ▼
PyMuPDF ──── extracts raw text (page by page)
    │
    ├── [if EXTRACT_FIGURES=true]
    │       PyMuPDF extracts image crops (skips < 100×100 px)
    │       → GPT-4o Vision → detailed figure description
    │       → Figure chunks with section = "Figure N (page P)"
    │
    ▼
chunking.py ── detect_sections()  →  regex splits on section headers
             └─ split_section()   →  800-token sliding window, 100-token overlap
    │
    ▼
OpenAI text-embedding-3-small  (batched, 100 texts/request)
    │
    ▼
ChromaDB  ──── vectors + metadata persisted to Docker volume
    │
    ▼
data/processed/metadata/<paper_id>.json  ──── duplicate guard
```

### Query Pipeline

```
User question
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
OpenAI embed query                      BM25Okapi
    │                                   (in-memory index,
    ▼                                   rebuilt on count change)
ChromaDB cosine search                         │
top 20 candidates                       Keyword scoring
    │                                   top 20 candidates
    └──────────────┬───────────────────────────┘
                   ▼
      Reciprocal Rank Fusion (k=60)
      Merged + deduplicated ranked list
                   │
                   ▼
      Jina rerank-multilingual-v2-base
      Cross-encoder re-scoring → top 5
                   │
                   ▼
      GPT-4o-mini  (streaming, max 2048 tokens)
                   │
                   ▼
      Grounded answer with inline source citations
```

### Infrastructure

```
┌────────────────────────┐        HTTP :8000         ┌───────────────────────┐
│  Local machine         │ ◄────────────────────────►│  Docker               │
│                        │                           │  chromadb/chroma      │
│  Flask API  :5000      │                           └───────────────────────┘
│  chat.py               │
└────────────────────────┘
           ▲
           │  HTTPS
           ▼
  ┌─────────────────────────────────────────────────┐
  │  OpenAI API   embeddings (text-embedding-3-small)│
  │               generation (gpt-4o-mini)           │
  │               figures    (gpt-4o vision)         │
  │  Jina AI      reranking  (rerank-multilingual)   │
  └─────────────────────────────────────────────────┘
```

### External Services

| Role | Provider | Model | Pricing |
|---|---|---|---|
| Embeddings | OpenAI | `text-embedding-3-small` | $0.020 / 1M tokens |
| Generation | OpenAI | `gpt-4o-mini` | $0.15 input / $0.60 output / 1M tokens |
| Figure extraction | OpenAI | `gpt-4o` Vision | $2.50 input / $10.00 output / 1M tokens |
| Reranking | Jina AI | `rerank-multilingual-v2-base` | Free 1M tokens/mo, then $0.018/1M |
| Vector store | ChromaDB (Docker) | — | Free |

### Estimated Cost — 150 papers/month, 900 queries/month

| | Without figures | With figures |
|---|---|---|
| Ingestion | ~$0.03 | ~$5.84 |
| Queries | ~$0.83 | ~$0.83 |
| **Total** | **~$0.86/mo** | **~$6.67/mo** |

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker
- OpenAI API key
- Jina AI API key (free tier available)

### 1. Clone and install

```bash
git clone https://github.com/your-username/paper-rag
cd paper-rag/paper_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Fill in your API keys in `.env`:

```env
OPENAI_API_KEY=your_openai_key
JINA_API_KEY=your_jina_key

CHROMA_HOST=localhost
CHROMA_PORT=8000

EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

TOP_K_CANDIDATES=20
TOP_K_FINAL=5

EXTRACT_FIGURES=false
```

### 3. Start ChromaDB

```bash
docker compose up -d
```

### 4. Add your PDFs and ingest

```bash
# Place PDFs in data/pdfs/, then:
python ingest_pdfs.py
```

```
2026-03-24 10:00:01 [INFO] app.ingest: Found 6 PDFs to ingest
2026-03-24 10:00:01 [INFO] app.ingest: Ingesting kumar_2024.pdf
...
Done. 6 paper(s) ingested.
  - kumar_2024 (36 chunks)
  - lum_2024 (43 chunks)
  - silva_2016 (21 chunks)
```

To enable figure extraction:

```bash
# In .env:
EXTRACT_FIGURES=true

# Re-ingest a single paper:
rm data/processed/metadata/kumar_2024.json
python ingest_pdfs.py
```

---

## Usage

### Terminal Chat

```bash
python chat.py
```

```
  ____                          ____      _    ____
 |  _ \ __ _ _ __   ___ _ __  |  _ \    / \  / ___|
 ...

  You › What was the optimal MOI used for macrophage infection?

  Assistant › According to [felipe2020, Section: 2.5. In vitro virus infections],
  monocytes were infected at MOI 10 and MDMs at MOI 5 in serum-free RPMI-1640...
```

### REST API

Start the server:

```bash
flask --app app.api run --port 5000
```

**Standard response:**

```bash
curl -s -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution?"}' | jq .
```

**Streaming response:**

```bash
curl -sN -X POST http://localhost:5000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What dataset was used for evaluation?"}'
```

Both endpoints accept:

```json
{
  "question": "your question",
  "top_k": 5,
  "paper_filter": "paper-id"
}
```

### Debug retrieval pipeline

```bash
python debug_retrieve.py
```

Prints chunk counts, top-5 vector results, top-5 BM25 results, and final reranked results — useful for diagnosing why a question returns poor answers.

---

## How Hybrid Search Works

Standard RAG relies solely on vector similarity, which misses exact technical terms, acronyms, and specific methodology names common in academic papers.

**Vector search** captures semantic meaning — useful for conceptual questions where the wording differs between query and paper.

**BM25** captures lexical matches — critical for scientific text where terms like `p-value < 0.05`, `MOI 10`, or `CHIKV` must be found verbatim.

Results from both are merged using **Reciprocal Rank Fusion** — rank-based, not score-based, requiring no weight tuning. The merged list is then passed to **Jina's neural reranker**, a cross-encoder that re-scores each chunk against the query — the highest-precision step in the pipeline.

This three-stage approach measurably outperforms standard single-stage RAG on scientific retrieval benchmarks.

---

## Why Not NotebookLM?

| | NotebookLM | Paper RAG |
|---|---|---|
| Max sources | 50 per notebook | Unlimited |
| Search | Vector only | BM25 + vector + neural rerank |
| Figure understanding | No | GPT-4o Vision |
| Data privacy | Google servers | Your machine |
| Programmable | No | REST API |
| Cost at scale | Subscription | Pay-per-token |
| Paper filter | Manual notebook switching | `paper_filter` param |
| Multilingual | Partial | Full |

---

## Project Structure Reference

| File | Responsibility |
|---|---|
| `models.py` | Pydantic v2 schemas shared across all modules |
| `chunking.py` | Section detection + sliding window splitting with tiktoken |
| `ingest.py` | PDF text extraction, GPT-4o Vision figure extraction, embedding, ChromaDB storage |
| `retrieve.py` | BM25 + vector search, RRF fusion, Jina reranking |
| `prompts.py` | System prompt and RAG prompt templates |
| `ask.py` | End-to-end Q&A orchestration, streaming support |
| `api.py` | Flask endpoints `/ask` and `/ask/stream` |
| `chat.py` | Styled interactive terminal interface |
| `ingest_pdfs.py` | Bulk ingestion entry point |
| `debug_retrieve.py` | Retrieval pipeline inspection tool |

---

<div align="center">

Built with Python · OpenAI · Jina AI · ChromaDB · Flask · Docker

</div>
