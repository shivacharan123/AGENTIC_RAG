"""
vector_store.py — Dense Vector Store wrapper for Agentic RAG

Responsibilities:
    • Wrap three backends (FAISS, Chroma, Pinecone) behind a single interface.
    • Store / upsert Document embeddings produced at ingestion time.
    • Expose similarity_search()  — standard dense retrieval.
    • Expose async_amax_marginal_relevance_search() — MMR-based diversity retrieval.
    • Expose retrieve()           — HyDE-aware entry point used by graph/nodes.py.

Relationship to other retrieval files:
    ┌──────────────┐   use_hyde=True    ┌──────────┐
    │  graph/nodes │ ──────────────────▶│  VectorStore.retrieve()  │
    └──────────────┘                    └──────────┘
                                             │
                          ┌──────────────────┼──────────────────────┐
                          ▼                  ▼                       ▼
                    hyde.generate()   similarity_search()   MMR search
                          │
                          └──▶  similarity_search(hypothetical_doc)

    reranker.py receives the List[Document] output of this file
    alongside results from web_search.py and sql_retriever.py.

Config block (config.yaml):
    retrieval:
      vector_store:
        type: "faiss"          # "faiss" | "chroma" | "pinecone"
        index_path: "data/index.faiss"   # faiss only
        collection_name: "rag_docs"      # chroma only
        persist_directory: "data/chroma" # chroma only
        pinecone_index: "rag-index"      # pinecone only
        pinecone_namespace: "default"    # pinecone only
        embedding_model: "text-embedding-3-small"
        k: 5
        mmr_fetch_k: 20        # candidate pool for MMR
        mmr_lambda: 0.5        # diversity weight (0 = max diversity, 1 = max relevance)
"""

from __future__ import annotations
import os
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import SecretStr
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class StoreBackend(str, Enum):
    FAISS    = "faiss"
    CHROMA   = "chroma"
    PINECONE = "pinecone"


@dataclass
class VectorStoreConfig:
    """Mirrors the `retrieval.vector_store` block in config.yaml."""
    type: str = "faiss"                          # faiss | chroma | pinecone

    # FAISS
    index_path: str = "vector_store/faiss_index"
    index_name: str = "agentic_rag"

    # Chroma
    collection_name: str = "rag_docs"
    persist_directory: str = "data/chroma"

    # Pinecone
    pinecone_index: str = "rag-index"
    pinecone_namespace: str = "default"
    pinecone_api_key: Optional[str] = None       # falls back to env var PINECONE_API_KEY
    pinecone_environment: Optional[str] = None   # legacy; ignored for serverless

    # Shared
    embedding_model: str = "models/gemini-embedding-001"
    k: int = 5
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.5

    # Extra kwargs forwarded to the embeddings constructor
    embedding_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal backend protocol
# ---------------------------------------------------------------------------

class _BaseBackend:
    """
    Minimal protocol every backend wrapper must satisfy.
    Concrete classes implement _similarity_search_with_scores and upsert.
    """

    def similarity_search_with_scores(
        self, query_text: str, k: int
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    async def async_mmr_search(
        self,
        query_text: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> List[Document]:
        raise NotImplementedError

    def upsert(self, documents: List[Document]) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class _FAISSBackend(_BaseBackend):
    """
    Wraps LangChain's FAISS vector store.
    Index is loaded from disk on init; a new index is created if absent.
    """

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings) -> None:
        from langchain_community.vectorstores import FAISS  # lazy import

        self._embeddings = embeddings
        self._index_path = str(
            Path(config.index_path) / config.index_name)
        self._store: Optional[FAISS] = None
        self._FAISS = FAISS
        self._load_or_create()

    def _load_or_create(self) -> None:
        import os
        from langchain_community.vectorstores import FAISS

        if os.path.exists(self._index_path):
            logger.info("FAISS: loading existing index from %s", self._index_path)
            try:
                self._store = FAISS.load_local(
                    self._index_path,
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                return
            except Exception as exc:
                logger.warning("FAISS: failed to load index (%s); creating new.", exc)

        logger.info("FAISS: creating empty index (no existing index found).")
        # Bootstrap with a sentinel document so the index is valid
        sentinel = Document(
            page_content="__sentinel__",
            metadata={"source": "system", "sentinel": True},
        )
        self._store = FAISS.from_documents([sentinel], self._embeddings)

    def similarity_search_with_scores(
        self, query_text: str, k: int
    ) -> List[Tuple[Document, float]]:
        assert self._store is not None
        results = self._store.similarity_search_with_score(query_text, k=k)
        # Filter out the bootstrap sentinel
        return [
            (doc, float(score))
            for doc, score in results
            if not doc.metadata.get("sentinel")
        ]

    async def async_mmr_search(
        self,
        query_text: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> List[Document]:
        assert self._store is not None
        # FAISS MMR is synchronous; run in executor so it doesn't block event loop
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: self._store.max_marginal_relevance_search(  # type: ignore[union-attr]
                query_text, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            ),
        )
        return [d for d in docs if not d.metadata.get("sentinel")]

    def upsert(self, documents: List[Document]) -> int:
        assert self._store is not None
        if not documents:
            return 0
        self._store.add_documents(documents)
        self._store.save_local(self._index_path)
        logger.info("FAISS: upserted %d docs and saved index to %s", len(documents), self._index_path)
        return len(documents)


# ---------------------------------------------------------------------------
# Chroma backend
# ---------------------------------------------------------------------------

class _ChromaBackend(_BaseBackend):
    """Wraps LangChain's Chroma vector store (local persistent mode)."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings) -> None:
        from langchain_community.vectorstores import Chroma  # lazy import

        logger.info(
            "Chroma: connecting to collection '%s' at %s",
            config.collection_name,
            config.persist_directory,
        )
        self._store = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embeddings,
        )

    def similarity_search_with_scores(
        self, query_text: str, k: int
    ) -> List[Tuple[Document, float]]:
        return [
            (doc, float(score))
            for doc, score in self._store.similarity_search_with_relevance_scores(
                query_text, k=k
            )
        ]

    async def async_mmr_search(
        self,
        query_text: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._store.max_marginal_relevance_search(
                query_text, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            ),
        )

    def upsert(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        self._store.add_documents(documents)
        logger.info("Chroma: upserted %d docs.", len(documents))
        return len(documents)


# ---------------------------------------------------------------------------
# Pinecone backend
# ---------------------------------------------------------------------------

class _PineconeBackend(_BaseBackend):
    """Wraps LangChain's PineconeVectorStore (serverless / pod-based)."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings) -> None:
        import os
        from langchain_pinecone import PineconeVectorStore  # lazy import
        from pinecone import Pinecone                       # lazy import

        api_key = config.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Pinecone API key not found. Set PINECONE_API_KEY env var or "
                "config.pinecone_api_key."
            )

        pc = Pinecone(api_key=api_key)
        index = pc.Index(config.pinecone_index)
        logger.info(
            "Pinecone: connected to index '%s', namespace '%s'.",
            config.pinecone_index,
            config.pinecone_namespace,
        )
        self._store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=config.pinecone_namespace,
        )

    def similarity_search_with_scores(
        self, query_text: str, k: int
    ) -> List[Tuple[Document, float]]:
        return [
            (doc, float(score))
            for doc, score in self._store.similarity_search_with_score(query_text, k=k)
        ]

    async def async_mmr_search(
        self,
        query_text: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._store.max_marginal_relevance_search(
                query_text, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            ),
        )

    def upsert(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        self._store.add_documents(documents)
        logger.info("Pinecone: upserted %d docs.", len(documents))
        return len(documents)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(config: VectorStoreConfig, embeddings: Embeddings) -> _BaseBackend:
    backend = StoreBackend(config.type.lower())
    if backend == StoreBackend.FAISS:
        return _FAISSBackend(config, embeddings)
    if backend == StoreBackend.CHROMA:
        return _ChromaBackend(config, embeddings)
    if backend == StoreBackend.PINECONE:
        return _PineconeBackend(config, embeddings)
    raise ValueError(f"Unknown vector store backend: {config.type!r}")


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _content_hash(doc: Document) -> str:
    """SHA-256 of the document's text content — used to drop exact duplicates."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _deduplicate(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        h = _content_hash(doc)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    removed = len(docs) - len(unique)
    if removed:
        logger.debug("VectorStore: removed %d duplicate doc(s).", removed)
    return unique


# ---------------------------------------------------------------------------
# Public VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Unified dense vector store for the Agentic RAG retrieval layer.

    Supports FAISS (local), Chroma (local / server), Pinecone (cloud).
    All public methods return LangChain Document objects with
    metadata["source"] = "vector" set automatically.

    Public API expected by other retrieval files
    ─────────────────────────────────────────────
    • similarity_search(query, k)               → List[Document]
        Called by hyde.py with either original query or hypothetical doc.

    • async_amax_marginal_relevance_search(query, k, fetch_k, lambda_mult)
        → List[Document]
        MMR search that balances relevance with diversity.

    • retrieve(query, k, use_hyde, hyde_client)  → List[Document]
        HyDE-aware entry point called from graph/nodes.py.

    • upsert(documents)                          → int
        Ingestion-time method to add/update embeddings.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """
        Args:
            config:     VectorStoreConfig instance.
            embeddings: Optional pre-built Embeddings; built from config if None.
        """
        self.config = config
        self._embeddings: Embeddings = embeddings or self._build_embeddings()
        self._backend: _BaseBackend = _build_backend(config, self._embeddings)
        logger.info(
            "VectorStore initialised (backend=%s, embedding_model=%s).",
            config.type,
            config.embedding_model,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self):
        
        emb_config = getattr(self.config, "embedding", None) or {}
        model_name = getattr(emb_config, "model_name", None) or "models/gemini-embedding-001"

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Missing GEMINI_API_KEY inside vector_store configuration!")

        logger.info(f"Vector Store online query embedder ready via Gemini: {model_name}")

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    @staticmethod
    def _annotate(docs: List[Document], extra_meta: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Stamp metadata["source"] = "vector" on every document."""
        for doc in docs:
            doc.metadata.setdefault("source", "vector")
            if extra_meta:
                doc.metadata.update(extra_meta)
        return docs

    # ------------------------------------------------------------------
    # Core retrieval — similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Dense similarity search.

        Called by hyde.py:
            docs = vector_store.similarity_search(hypothetical_doc, k=5)

        Also called by retrieve() when use_hyde=False.

        Args:
            query: Query text (plain user query OR hypothetical document from HyDE).
            k:     Number of top documents to return. Defaults to config.k.

        Returns:
            List[Document] sorted by relevance (most relevant first),
            each with metadata["source"] = "vector" and metadata["score"].
        """
        if not query or not query.strip():
            raise ValueError("VectorStore.similarity_search: query must be non-empty.")

        k = k or self.config.k
        logger.info("VectorStore.similarity_search | k=%d | query=%r…", k, query[:100])

        try:
            results: List[Tuple[Document, float]] = (
                self._backend.similarity_search_with_scores(query, k=k)
            )
        except Exception as exc:
            logger.error("VectorStore similarity search failed: %s", exc)
            raise

        # Attach raw similarity score into metadata
        docs = []
        for doc, score in results:
            doc.metadata["score"] = score
            docs.append(doc)

        docs = _deduplicate(docs)
        self._annotate(docs)
        logger.debug("VectorStore.similarity_search: returning %d docs.", len(docs))
        return docs

    # ------------------------------------------------------------------
    # MMR search (async)
    # ------------------------------------------------------------------

    async def async_amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ) -> List[Document]:
        """
        Async Maximum Marginal Relevance search.

        Balances relevance with diversity to avoid returning near-duplicate
        chunks from the same document.

        Args:
            query:       Query text.
            k:           Final number of documents to return.
            fetch_k:     Candidate pool size before MMR filtering.
            lambda_mult: 0 = max diversity, 1 = max relevance (default 0.5).

        Returns:
            Diverse, relevant List[Document] with source metadata.
        """
        if not query or not query.strip():
            raise ValueError("async_amax_marginal_relevance_search: query must be non-empty.")

        k          = k          or self.config.k
        fetch_k    = fetch_k    or self.config.mmr_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else self.config.mmr_lambda

        logger.info(
            "VectorStore.MMR | k=%d | fetch_k=%d | λ=%.2f | query=%r…",
            k, fetch_k, lambda_mult, query[:100],
        )

        try:
            docs = await self._backend.async_mmr_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        except Exception as exc:
            logger.error("VectorStore MMR search failed: %s. Falling back to similarity search.", exc)
            # Graceful sync fallback
            docs = self.similarity_search(query, k=k)

        docs = _deduplicate(docs)
        self._annotate(docs, extra_meta={"retrieval_method": "mmr"})
        logger.debug("VectorStore.MMR: returning %d docs.", len(docs))
        return docs

    # ------------------------------------------------------------------
    # HyDE-aware retrieve (primary entry point for graph/nodes.py)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_hyde: bool = False,
        hyde_client: Optional[Any] = None,   # HyDE instance from hyde.py
    ) -> List[Document]:
        """
        HyDE-aware retrieval entry point.

        Workflow:
            1. If use_hyde=True and hyde_client is provided:
               a. hyde_client.generate_hypothetical_answer(query) → hypothetical_doc
               b. similarity_search(hypothetical_doc, k)          → docs
            2. Otherwise: similarity_search(query, k)             → docs

        Falls back to plain similarity search if HyDE generation fails.

        Args:
            query:       Original user query string.
            k:           Number of documents to return.
            use_hyde:    Whether to apply HyDE enhancement.
            hyde_client: Instantiated HyDE object from hyde.py
                         (must expose generate_hypothetical_answer(str) → str).

        Returns:
            List[Document] annotated with:
                metadata["source"]         = "vector"
                metadata["hyde"]           = True / False
                metadata["original_query"] = query
        """
        k = k or self.config.k

        search_text = query   # default: search with original query
        used_hyde   = False

        if use_hyde and hyde_client is not None:
            try:
                logger.info("VectorStore.retrieve: invoking HyDE for query %r…", query[:80])
                search_text = hyde_client.generate_hypothetical_answer(query)
                used_hyde   = True
                logger.debug("HyDE text (%d chars): %s…", len(search_text), search_text[:150])
            except Exception as exc:
                logger.warning(
                    "VectorStore.retrieve: HyDE failed (%s); using original query.", exc
                )
                search_text = query
                used_hyde   = False
        elif use_hyde and hyde_client is None:
            logger.warning(
                "VectorStore.retrieve: use_hyde=True but no hyde_client provided; "
                "falling back to plain search."
            )

        docs = self.similarity_search(search_text, k=k)
        self._annotate(
            docs,
            extra_meta={
                "hyde": used_hyde,
                "original_query": query,
            },
        )
        return docs

    # ------------------------------------------------------------------
    # Ingestion — upsert
    # ------------------------------------------------------------------

    def upsert(self, documents: List[Document]) -> int:
        """
        Add or update documents in the vector store.

        Deduplicates by content hash before inserting to avoid redundancy.

        Args:
            documents: List of LangChain Document objects to embed and store.

        Returns:
            Number of documents actually written.
        """
        if not documents:
            logger.debug("VectorStore.upsert called with empty list; nothing to do.")
            return 0

        unique_docs = _deduplicate(documents)
        logger.info(
            "VectorStore.upsert: %d docs in → %d unique docs after dedup.",
            len(documents),
            len(unique_docs),
        )

        try:
            written = self._backend.upsert(unique_docs)
            logger.info("VectorStore.upsert: %d docs written to %s.", written, self.config.type)
            return written
        except Exception as exc:
            logger.error("VectorStore.upsert failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VectorStore(backend={self.config.type!r}, "
            f"embedding={self.config.embedding_model!r}, "
            f"k={self.config.k})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def build_vector_store_from_config(
    config_dict: Dict[str, Any],
    embeddings: Optional[Embeddings] = None,
) -> VectorStore:
    """
    Construct a VectorStore from the `retrieval.vector_store` section of config.yaml.

    Example:
        vs_cfg = yaml_config["retrieval"]["vector_store"]
        vs = build_vector_store_from_config(vs_cfg)
    """
    cfg = VectorStoreConfig(
        type=config_dict.get("type", "faiss"),
        index_path=config_dict.get("index_path", "vector_store/faiss_index"),
        index_name=config_dict.get("index_name", "agentic_rag"),
        collection_name=config_dict.get("collection_name", "rag_docs"),
        persist_directory=config_dict.get("persist_directory", "data/chroma"),
        pinecone_index=config_dict.get("pinecone_index", "rag-index"),
        pinecone_namespace=config_dict.get("pinecone_namespace", "default"),
        pinecone_api_key=config_dict.get("pinecone_api_key"),
        embedding_model=config_dict.get("embedding_model", "models/gemini-embedding-001"),
        k=int(config_dict.get("k", 5)),
        mmr_fetch_k=int(config_dict.get("mmr_fetch_k", 20)),
        mmr_lambda=float(config_dict.get("mmr_lambda", 0.5)),
        embedding_kwargs=config_dict.get("embedding_kwargs", {}),
    )
    return VectorStore(cfg, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Standalone smoke-test  (python -m retrieval.vector_store)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.DEBUG)

    cfg = VectorStoreConfig(
        type="faiss",
        index_path="data/test_index.faiss",
        index_name="test_rag",
        embedding_model=os.getenv("EMBED_MODEL", "models/gemini-embedding-001"),
        k=3,
    )

    vs = VectorStore(cfg)

    # --- Upsert sample docs ---
    sample_docs = [
        Document(page_content="Product X sold 1.2 million units in Q1 2025 in North America.",
                 metadata={"source": "vector", "doc_id": "rpt-001"}),
        Document(page_content="Q1 2025 earnings showed a 12% YoY increase for product line X.",
                 metadata={"source": "vector", "doc_id": "rpt-002"}),
        Document(page_content="Supply chain disruptions impacted product X availability in Q1.",
                 metadata={"source": "vector", "doc_id": "rpt-003"}),
    ]
    written = vs.upsert(sample_docs)
    print(f"\nUpserted {written} documents.")

    # --- Plain similarity search ---
    query = "What were the total sales of product X in Q1 2025?"
    results = vs.similarity_search(query, k=2)
    print(f"\nSimilarity search results ({len(results)}):")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] score={doc.metadata.get('score', 'N/A'):.4f}  {doc.page_content[:80]}")

    # --- HyDE-aware retrieve (no real HyDE client in smoke-test) ---
    retrieved = vs.retrieve(query, k=2, use_hyde=False)
    print(f"\nretrieve() results ({len(retrieved)}):")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] hyde={doc.metadata.get('hyde')}  {doc.page_content[:80]}")