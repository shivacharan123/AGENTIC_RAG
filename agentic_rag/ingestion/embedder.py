"""
Embedding and vector store ingestion for Agentic RAG.
Embedding model: google/embeddinggemma-300m

Complete data flow:
    config.yaml
        ├── data.raw_docs_dir           → loader.py
        ├── ingestion.*                 → chunker.py
        ├── models.embedding.*          → HuggingFaceEmbeddings (this file)
        └── retrieval.vector_store.*    → FAISS save path (this file)

    loader.load_from_config()
            ↓  List[Document]
    chunker.chunk_documents()
            ↓  List[Document] (chunks)
    HuggingFaceEmbeddings  (google/embeddinggemma-300m, local)
            ↓  768-dim vectors
    FAISS index  →  saved to disk

PRE-REQUISITES (run once before using this file):
──────────────────────────────────────────────────
1. Accept Google's license at https://huggingface.co/google/embeddinggemma-300m
   (click "Acknowledge license" — processed immediately)

2. Install the special transformers build EmbeddingGemma requires:
       pip install -U sentence-transformers
       pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
   Without the preview transformers, the model silently uses causal attention
   instead of bidirectional attention, producing bad embeddings.

3. Add your HuggingFace token to .env:
       HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   Or run  huggingface-cli login  once in your terminal.

Config keys used (config.yaml):
    models.embedding.model_name  → default: google/embeddinggemma-300m
    models.embedding.dimensions  → default: 768  (max for this model)
    models.embedding.device      → default: cpu   (set "cuda" if you have GPU)
"""

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from ..prompts.config import load_config
from .loader import load_documents, load_from_directory, load_from_config
from .chunker import chunk_documents
import time
from langchain_community.vectorstores import FAISS

class Embedder:
    """
    Orchestrates: load → chunk → embed → store.
    Uses google/embeddinggemma-300m locally via sentence-transformers.
    All configuration is read from config.yaml.
    """
    def __init__(self):
        self.config = load_config()
        print("CONFIG =", self.config)

        # ── Embedding model config (Gemini, API-based) ───────────────────────
        emb_cfg = self.config.get("models", {}).get("embedding", {})
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing — required for Embedder to call "
                "Google's embeddings endpoint."
            )

        model_name = emb_cfg.get("model_name", "models/gemini-embedding-001")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

        print(f"[embedder] ✅ Embeddings ready via Gemini: {model_name}")

        # ── Vector store path (from config) ──────────────────────────────────
        self.vector_store_path = Path(
            self.config
            .get("retrieval", {})
            .get("vector_store", {})
            .get("index_path", "vector_store/faiss_index")
        )
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Core pipeline ────────────────────────────────────────────────────────

    def run_full_pipeline(self, index_name: str = "agentic_rag") -> FAISS:
        """
        PRIMARY METHOD — reads everything from config.yaml, no arguments needed.

        Flow:
            config.yaml (data.raw_docs_dir)
                → loader.load_from_config()   →  List[Document]
                → chunker.chunk_documents()   →  List[Document] (chunks)
                → HuggingFaceEmbeddings        →  768-dim vectors
                → FAISS.from_documents()       →  index
                → save to disk
        """
        print("[embedder] Starting full ingestion pipeline...")
        documents = load_from_config()
        return self.embed_and_store(documents, index_name)

    

    def embed_and_store(
        self,
        documents: list[Document],
        index_name: str = "agentic_rag",
    ) -> FAISS:
        chunks = chunk_documents(documents)
        print(f"[embedder] Embedding {len(chunks)} chunks via Gemini (rate-limited)...")

        batch_size = 95
        vector_store: Optional[FAISS] = None
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(chunks), batch_size), start=1):
            batch = chunks[i:i + batch_size]
            print(f"[embedder] Batch {batch_num}/{total_batches} "
                  f"({i + 1}-{min(i + batch_size, len(chunks))} of {len(chunks)})...")

            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    vector_store.add_documents(batch)
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Daily quota is unrecoverable — don't retry, fail fast
                    if "PerDay" in err_str or "per_day" in err_str.lower():
                        raise RuntimeError(
                            "Daily Gemini embedding quota exhausted. "
                            "Wait until tomorrow or use a different API key. "
                            "Your existing index is still valid — don't re-ingest."
                        ) from exc
                    # Per-minute quota — wait and retry
                    print("[embedder] Rate limited (per-minute) — waiting 60s then retrying...")
                    time.sleep(60)
                    # retry logic here...
                else:
                    raise

        self.save_vector_store(vector_store, index_name)
        print(f"[embedder] ✅ Done. {len(chunks)} chunks stored in index '{index_name}'.")
        return vector_store
    
    def _embed_batch_with_retry(self, vector_store, batch, is_first, max_retries=3):
        for attempt in range(max_retries):
            try:
                if is_first:
                    return FAISS.from_documents(batch, self.embeddings)
                else:
                    vector_store.add_documents(batch)
                    return vector_store
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    wait = 60
                    print(f"[embedder] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Embedding batch failed after max retries.")
    # ── Vector store helpers ─────────────────────────────────────────────────

    def save_vector_store(self, vector_store: FAISS, index_name: str = "agentic_rag"):
        """Persist FAISS index to disk."""
        index_path = self.vector_store_path / index_name
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
        print(f"[embedder] Vector store saved → {index_path}")

    def load_vector_store(self, index_name: str = "agentic_rag") -> Optional[FAISS]:
        """Load an existing FAISS index from disk. Returns None if not found."""
        index_path = self.vector_store_path / index_name
        if index_path.exists():
            print(f"[embedder] Loading existing index from {index_path}")
            return FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        print(f"[embedder] No existing index at {index_path}.")
        return None

    def add_documents(
        self,
        documents: list[Document],
        index_name: str = "agentic_rag",
    ) -> FAISS:
        """Add new documents to an existing index (creates it if absent)."""
        vector_store = self.load_vector_store(index_name)
        if vector_store is None:
            return self.embed_and_store(documents, index_name)

        chunks = chunk_documents(documents)
        vector_store.add_documents(chunks)
        self.save_vector_store(vector_store, index_name)
        print(f"[embedder] Added {len(chunks)} chunks to index '{index_name}'.")
        return vector_store


# ── Convenience functions ────────────────────────────────────────────────────

def run_ingestion(index_name: str = "agentic_rag") -> FAISS:
    """One-liner entry point. Reads everything from config.yaml."""
    return Embedder().run_full_pipeline(index_name)


def ingest_documents(file_paths: list[str], index_name: str = "agentic_rag") -> FAISS:
    """Ingest an explicit list of file paths."""
    embedder = Embedder()
    documents = load_documents(file_paths)
    return embedder.embed_and_store(documents, index_name)


def ingest_directory(directory: str, index_name: str = "agentic_rag") -> FAISS:
    """Ingest all files from a specific directory."""
    embedder = Embedder()
    documents = load_from_directory(directory)
    return embedder.embed_and_store(documents, index_name)