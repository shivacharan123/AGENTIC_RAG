# 🤖 Agentic RAG

A production-style **Agentic Retrieval-Augmented Generation** system built with **LangGraph**. Instead of a single "retrieve → generate" pass, this system *plans*, *routes*, *retrieves in parallel from multiple sources*, *grades its own evidence*, *self-corrects*, and *verifies its answers for hallucinations* before responding.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Graph Workflow](#-graph-workflow)
- [Project Structure](#-project-structure)
- [Component Deep Dive](#-component-deep-dive)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Ingesting Documents](#-ingesting-documents)
- [Running the API](#-running-the-api)
- [Memory System](#-memory-system)
- [Design Decisions](#-design-decisions)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Query Planning** | A query analyzer routes each question and decomposes complex multi-hop queries into sub-queries. |
| **Parallel Multi-Source Retrieval** | Fans out to **vector search**, **web search**, and **SQL retrieval** simultaneously and merges results. |
| **HyDE** | Hypothetical Document Embeddings improve dense retrieval for short or vague queries. |
| **Cross-Encoder Reranking** | Cohere Rerank or a local BGE reranker re-scores merged candidates. |
| **Self-Corrective Loop** | A relevance grader filters weak evidence; failures trigger query rewriting and re-retrieval (with a hard retry limit). |
| **Hallucination Checking** | Generated answers are verified against context; unsupported claims trigger regeneration. |
| **Multiple Generation Modes** | `rag`, `regenerate`, `conversational`, and `fallback`. |
| **Two-Tier Memory** | Short-term per-thread checkpointing + long-term summaries and fact store. |
| **Streaming** | Token streaming via `stream_generate()` for real-time UIs. |
| **Special Routes** | Chitchat and out-of-scope questions short-circuit retrieval entirely. |
| **Config-Driven** | Models, retrievers, rerankers, memory, and thresholds all live in `config.yaml`. |

---

## 🏗 Architecture

```
                         ┌──────────────────┐
        User Query ────▶ │  FastAPI (api/)  │
                         └────────┬─────────┘
                                  │  thread_id scoped
                                  ▼
                    ┌───────────────────────────┐
                    │   LangGraph Workflow      │
                    │   (graph/builder.py)      │
                    └───────────────────────────┘
                                  │
        ┌─────────────┬───────────┴───────────┬──────────────┐
        ▼             ▼                       ▼              ▼
   Agents layer   Retrieval layer        Prompts layer   Memory layer
   (planning,     (vector, web,          (templates &    (checkpointer +
    grading,       SQL, HyDE,             chains)         long-term store)
    generation)    reranker)
```

The codebase is split into clear layers:

- **Agents** – decision-making LLM components (analyze, grade, generate, verify).
- **Retrieval** – everything that fetches or scores documents.
- **Prompts** – pure prompt engineering and LangChain chain factories (never imports from `retrieval/`).
- **Graph** – state, nodes, edges, and workflow wiring.
- **Memory** – session persistence and cross-session knowledge.
- **API** – the FastAPI serving layer.

---

## 🔀 Graph Workflow

```
query_analyzer
    → router_node
        ├─▶ chitchat        → END      (greetings / small talk)
        ├─▶ out_of_scope    → END      (outside the domain)
        └─▶ [vector_search ‖ web_search_tool ‖ sql_graph_tool]   ← parallel fan-out
                    ↓
            context_aggregator   (merge → dedup → rerank)
                    ↓
            relevance_grader
              ├─ pass  → answer_generator → hallucination_checker
              │                               ├─ grounded   → END
              │                               └─ unsupported → regenerate
              └─ retry → query_rewriter → router_node   ← self-corrective loop
                         (retries exhausted → fallback answer)
```

**Parallel fan-out.** `decide_route()` in `graph/edges.py` returns a `List[str]` of node names. LangGraph (≥ 0.2) dispatches to all of them concurrently. Route values include `web_only`, `sql_only`, and `parallel_all`.

**Safe merging.** `retrieved_documents` in `GraphState` uses a custom reducer (`merge_documents`) so parallel branches merge cleanly, and an explicit empty-list write from `query_rewriter` acts as a **reset** so stale documents never leak into a new retrieval round.

---

## 📂 Project Structure

```
agentic_rag/
├── ingestion/                  # Offline data pipeline
│   ├── loader.py               # Load TXT / MD / PDF with metadata extraction
│   ├── chunker.py              # Configurable chunking with overlap
│   └── embedder.py             # Embeddings + vector index creation
│
├── graph/                      # LangGraph orchestration
│   ├── state.py                # GraphState + custom reducers (merge_documents)
│   ├── nodes.py                # Node implementations
│   ├── edges.py                # Conditional routing (decide_route, grade edges)
│   └── builder.py              # build_workflow() / build_app()
│
├── retrieval/                  # Retrieval layer
│   ├── vector_store.py         # FAISS / Chroma / Pinecone wrapper + MMR + HyDE entry
│   ├── hyde.py                 # Hypothetical Document Embeddings
│   ├── web_search.py           # Tavily / SerpAPI retriever
│   ├── sql_retriever.py        # NL → SQL → structured results
│   └── reranker.py             # Cohere / local BGE cross-encoder reranking
│
├── prompts/                    # Prompt templates & chain factories
│   ├── generator.py            # RAG / regenerate / conversational / fallback prompts
│   ├── grader.py               # Relevance, hallucination & answer-quality prompts
│   └── query_transform.py      # Rewrite, step-back, sub-query decomposition
│
├── agent/                      # LLM-powered agents
│   ├── query_analyzer.py       # Routing + query decomposition (planning layer)
│   ├── relevancer_grader.py    # Filters irrelevant documents
│   ├── generator.py            # Answer synthesis orchestrator
│   └── hallusination_checker.py# Grounding / hallucination verification
│
├── memory/                     # Persistence
│   ├── checkepointer.py        # Short-term session memory (LangGraph checkpointer)
│   └── long_term.py            # Summaries + cross-session fact store
│
├── api/
│   └── main.py                 # FastAPI entry point
│
├── data/                       # Raw docs, indexes, SQLite DBs
├── config.py                   # Config loader
└── config.yaml                 # Central configuration
```

> **Note:** a few filenames (`checkepointer.py`, `hallusination_checker.py`) contain typos from the original implementation. Consider renaming them (and updating imports) for polish.

---

## 🔍 Component Deep Dive

### Ingestion (`ingestion/`)

1. **`loader.py`** – Loads `.txt`, `.md`, and `.pdf` files from `data.raw_docs_dir`, extracting metadata and applying basic cleaning. `load_from_config()` reads the path automatically; `load_from_directory()` accepts an explicit one.
2. **`chunker.py`** – Splits documents using `chunk_size`, `chunk_overlap`, and separators from the `ingestion` config section, attaching chunk metadata.
3. **`embedder.py`** – Embeds chunks with **`google/embeddinggemma-300m`** (768-dim, runs locally via `HuggingFaceEmbeddings`) and saves the index (FAISS by default).

### Retrieval (`retrieval/`)

- **`vector_store.py`** – One interface over **FAISS, Chroma, and Pinecone**. Provides `similarity_search()`, async MMR search for diversity, and `retrieve()` — the HyDE-aware entry point used by the graph.
- **`hyde.py`** – Generates a hypothetical expert answer and uses *that* for similarity search. Falls back to the raw query on failure.
- **`web_search.py`** – **Tavily** or **SerpAPI** with timeouts, retries, and graceful fallbacks. Returns the same `Document` schema as the other retrievers.
- **`sql_retriever.py`** – Converts natural language to SQL, validates it (rejects `DROP`, `DELETE`, etc.), and executes it against a SQLite database.
- **`reranker.py`** – Scores `(query, document)` pairs with a cross-encoder. Backends: **Cohere** (`rerank-v3.5`), **local BGE** (`BAAI/bge-reranker-base`), or **passthrough**. Annotates `metadata["rerank_score"]` and never crashes the graph on failure.

### Agents (`agent/`)

| Agent | Role |
|---|---|
| `query_analyzer.py` | Routes queries and decomposes complex questions (planning layer). |
| `relevancer_grader.py` | Filters irrelevant retrieved documents; drives the self-corrective loop. |
| `generator.py` | Orchestrates generation across four modes (below). |
| `hallusination_checker.py` | Verifies the answer is grounded and flags unsupported claims. |

**Generation modes**

| Mode | When it's used |
|---|---|
| `rag` | Primary synthesis from aggregated multi-source context. |
| `regenerate` | Fixes a hallucinated answer using flagged unsupported claims. |
| `conversational` | Multi-turn chat grounded in retrieved context. |
| `fallback` | All retrieval retries exhausted — polite baseline answer. |

Every prompt enforces strict **context-only** answering to avoid fabrication.

### Prompts (`prompts/`)

Pure prompt engineering with LangChain chain factories (`get_generator_chain`, `get_fallback_chain`, `get_conversational_chain`, `get_regeneration_chain`), the canonical `format_docs()` context formatter, and `stream_answer()` for streaming. `query_transform.py` supports **query rewriting**, **step-back prompting**, and **sub-query decomposition**.

---

## 🧰 Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLMs:** OpenRouter (e.g. `anthropic/claude-3.5-sonnet` for generation, `gpt-4o-mini` for lightweight tasks)
- **Embeddings:** `google/embeddinggemma-300m` (HuggingFace / sentence-transformers)
- **Vector stores:** FAISS · Chroma · Pinecone
- **Reranking:** Cohere Rerank · BAAI/bge-reranker-base
- **Web search:** Tavily · SerpAPI
- **Structured data:** SQLite
- **Persistence:** LangGraph checkpointers (memory / SQLite / Postgres)
- **API:** FastAPI + Uvicorn

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key
- (Optional) Tavily / SerpAPI, Cohere, and Pinecone keys depending on the backends you enable
- A HuggingFace account (for EmbeddingGemma)

### Installation

```bash
git clone https://github.com/<your-username>/agentic-rag.git
cd agentic-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### EmbeddingGemma setup (one-time)

1. Accept the license at <https://huggingface.co/google/embeddinggemma-300m>.
2. Install the required transformers build:
   ```bash
   pip install -U sentence-transformers
   pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
   ```
   Without it, the model silently uses causal instead of bidirectional attention and produces poor embeddings.
3. Authenticate with HuggingFace (`huggingface-cli login`) or set `HUGGINGFACE_TOKEN` in `.env`.

### Environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
TAVILY_API_KEY=tvly-...        # if using Tavily
SERPAPI_API_KEY=...            # if using SerpAPI
COHERE_API_KEY=...             # if using Cohere reranker
PINECONE_API_KEY=...           # if using Pinecone
```

---

## ⚙️ Configuration

All behavior is controlled by `config.yaml`. Representative sections:

```yaml
data:
  raw_docs_dir: "data/raw"

models:
  embedding:
    model_name: "google/embeddinggemma-300m"
    dimensions: 768
    device: "cpu"              # "cuda" if available

agents:
  generator:
    provider: "openrouter"
    model_name: "anthropic/claude-3.5-sonnet"
    temperature: 0.0
    max_tokens: 2048
    streaming: true
    openrouter_base_url: "https://openrouter.ai/api/v1"

retrieval:
  vector_store:
    type: "faiss"              # faiss | chroma | pinecone
    index_path: "data/index.faiss"
    k: 5
    mmr_fetch_k: 20
    mmr_lambda: 0.5
  web_search:
    engine: "tavily"           # tavily | serpapi
    k: 3
    timeout: 10
    max_retries: 2
  reranker:
    backend: "cohere"          # cohere | local | passthrough
    model: "rerank-v3.5"
    local_model: "BAAI/bge-reranker-base"
    top_k: 5

query:
  transform:
    max_subqueries: 4
    rewrite_model: "gpt-4o-mini"
    decompose_model: "gpt-4o-mini"

memory:
  checkpointer:
    backend: "sqlite"          # memory | sqlite | postgres
    sqlite_path: "data/checkpoints.sqlite"
  long_term:
    db_path: "data/memory/long_term.db"
    summary_frequency: 5
    max_facts_per_thread: 100
    fact_retrieval_top_k: 5
```

---

## 📥 Ingesting Documents

Place `.txt`, `.md`, or `.pdf` files in `data/raw/` and run the ingestion pipeline:

```bash
python -m agentic_rag.ingestion.embedder
```

Data flow: `loader → chunker → embedder (EmbeddingGemma) → FAISS index on disk`.

You can also trigger ingestion through the API's `/ingest` endpoint (the embedder is lazy-imported so the server starts fast).

---

## 🌐 Running the API

```bash
uvicorn agentic_rag.api.main:app --reload --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

Example request (replace the endpoint/fields with your actual schema):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about retrieval?", "thread_id": "session-1"}'
```

Each `thread_id` has its own isolated conversation state.

---

## 🧠 Memory System

**Short-term (`memory/checkepointer.py`)** – Persists `GraphState` snapshots across nodes and across turns within a `thread_id`. Backends: in-process `memory`, durable `sqlite`, or shared `postgres`. Swapping backends never requires changing nodes, edges, or the builder.

**Long-term (`memory/long_term.py`)**
- **Compression:** every N turns (`summary_frequency`), chat history is summarized by an LLM and replaces the raw turns so the context window never overflows.
- **Fact store:** durable cross-session facts (preferences, entities, decisions) in SQLite via `store_fact()` / `retrieve_facts()`.
- **Context injection:** `build_memory_context()` injects the stored summary and relevant facts at the start of each turn.
- **Persistence:** summaries survive restarts, stored in a separate DB to keep concerns isolated.

---

## 🧩 Design Decisions

- **Bounded self-correction.** Retry loops have hard limits and end in a graceful `fallback` mode instead of looping forever.
- **Fail-open grading.** If a grader or reranker fails, the pipeline degrades gracefully (original ordering / pass-through) rather than crashing.
- **State-reset reducer.** `merge_documents` distinguishes `None` (keep), `[]` (reset), and non-empty (append), so each retry round sees only fresh documents.
- **Layer independence.** `web_search.py` and `reranker.py` have no cross-imports inside `retrieval/`; `prompts/` never imports from `retrieval/`.
- **Uncompiled workflow.** `build_workflow()` returns an uncompiled graph for testability; `build_app()` attaches the checkpointer.
- **Lazy imports.** Heavy dependencies (embedder) are loaded only where needed.

---

## 🧪 Testing

A standalone harness validates the SQL retriever in isolation (DB connection, schema, NL→SQL, execution, async wrapper, and safety validators):

```bash
export OPENROUTER_API_KEY=sk-...
python test_sql_retriever.py
```

Set `SKIP_LLM_TESTS = True` inside the script to test DB/schema/safety logic without API calls.

---

## 🗺 Roadmap

- [ ] Dockerize the API and ingestion pipeline
- [ ] Kubernetes manifests and CI/CD
- [ ] Evaluation suite (RAGAS / faithfulness and answer relevancy metrics)
- [ ] Web UI with streaming responses
- [ ] Observability with LangSmith tracing
- [ ] Additional document loaders (DOCX, HTML, CSV)

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, create a feature branch, and open a pull request with a clear description of the change.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👤 Author

**Shiva** — B.Tech CSE (AI/ML), IIIT Nagpur
Open to AI Engineer roles · [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
