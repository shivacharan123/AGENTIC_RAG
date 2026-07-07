"""
reranker.py — Cross-Encoder Reranking Layer for Agentic RAG

Responsibilities:
    • Accept the merged List[Document] from all three retrievers
      (vector_store, web_search, sql_retriever) after deduplication.
    • Score every (query, document) pair using a cross-encoder model.
    • Sort by relevance score descending and truncate to top_k.
    • Annotate each Document with its reranker score in metadata["rerank_score"].
    • Fall back to original ordering on model failure — never crash the graph.

Position in the retrieval pipeline:
    ┌──────────────────────────────────────────────────────────────┐
    │           context_aggregator node  (graph/nodes.py)         │
    └──────────────────────────────┬───────────────────────────────┘
                                   │
              merged + deduplicated List[Document]
              (vector + web + sql results combined)
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   DocumentReranker        │
                    │                          │
                    │  Backend A: Cohere API   │
                    │  Backend B: Local BGE    │  ← cross-encoder models
                    │  Backend C: Pass-through │  ← if no key / no model
                    └──────────────────────────┘
                                   │
                    top_k scored + sorted Documents
                                   │
                                   ▼
                          relevance_grader node

Independence:
    reranker.py has NO imports from other retrieval/ files.
    It is the last stage of retrieval — pure filter, no fetching.

Config block (config.yaml):
    retrieval:
      reranker:
        backend: "cohere"                   # "cohere" | "local" | "passthrough"
        model: "rerank-v3.5"                # Cohere model name
        local_model: "BAAI/bge-reranker-base"  # HuggingFace model for local backend
        top_k: 5
        score_threshold: null               # optional float: drop docs below threshold
        batch_size: 32                      # local backend only
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class RerankerBackend(str, Enum):
    COHERE      = "cohere"
    LOCAL       = "local"
    PASSTHROUGH = "passthrough"


@dataclass
class RerankerConfig:
    """Mirrors the `retrieval.reranker` block in config.yaml."""
    backend: str           = "cohere"
    model: str             = "rerank-v3.5"          # Cohere model
    local_model: str       = "BAAI/bge-reranker-base"
    top_k: int             = 5
    score_threshold: Optional[float] = None         # None = no threshold filtering
    batch_size: int        = 32                      # local inference batch size
    cohere_api_key: Optional[str] = None             # falls back to COHERE_API_KEY env
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scored result — internal normalised form
# ---------------------------------------------------------------------------

@dataclass
class _ScoredDocument:
    document: Document
    score: float
    original_index: int    # position before reranking — for tie-breaking


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------

class _BaseRerankerBackend:
    """Protocol every backend must implement."""

    def score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Return (document, relevance_score) pairs.
        Score range: 0.0–1.0 (or unbounded for local cross-encoders).
        Higher = more relevant.
        """
        raise NotImplementedError

    async def async_score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Default: run sync score() in thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.score(query, documents)
        )


# ---------------------------------------------------------------------------
# Cohere backend
# ---------------------------------------------------------------------------

class _CohereBackend(_BaseRerankerBackend):

    def __init__(self, config: RerankerConfig) -> None:
        try:
            from langchain_cohere import CohereRerank
            from pydantic import SecretStr
        except ImportError as exc:
            raise ImportError(
                "langchain-cohere is required for the Cohere backend. "
                "Install with: pip install langchain-cohere cohere"
            ) from exc

        api_key = config.cohere_api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Cohere API key not found. Set COHERE_API_KEY environment variable "
                "or config.cohere_api_key."
            )

        self._compressor = CohereRerank(
            cohere_api_key=SecretStr(api_key),
            model=config.model,
            top_n=1000,
        )
        self._top_k = config.top_k
        logger.info("CohereBackend initialised (model=%s).", config.model)

    # ← THIS METHOD WAS COMPLETELY MISSING
    def score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Call Cohere rerank API and return (document, score) pairs."""
        if not documents:
            return []

        # CohereRerank.compress_documents returns docs with relevance_score in metadata
        compressed = self._compressor.compress_documents(
            documents=documents,
            query=query,
        )

        # Build a lookup from content → score
        score_map: Dict[str, float] = {}
        for doc in compressed:
            score_map[doc.page_content] = float(
                doc.metadata.get("relevance_score", 0.0)
            )

        # Return original docs with their scores (preserves docs not returned by Cohere)
        return [
            (doc, score_map.get(doc.page_content, 0.0))
            for doc in documents
        ]

# ---------------------------------------------------------------------------
# Local cross-encoder backend (HuggingFace / sentence-transformers)
# ---------------------------------------------------------------------------

class _LocalCrossEncoderBackend(_BaseRerankerBackend):
    """
    Uses sentence-transformers CrossEncoder for fully local, offline reranking.
    Install: pip install sentence-transformers
    Recommended model: BAAI/bge-reranker-base  (fast, strong)
    """

    def __init__(self, config: RerankerConfig) -> None:
        try:
            from sentence_transformers import CrossEncoder  # lazy import
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the local backend. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "LocalCrossEncoderBackend: loading model %r (this may take a moment)…",
            config.local_model,
        )
        self._model = CrossEncoder(config.local_model)
        self._batch_size = config.batch_size
        logger.info("LocalCrossEncoderBackend: model loaded.")

    def score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Run cross-encoder inference in batches.
        Returns raw logit scores (unbounded; higher = more relevant).
        """
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        return [(doc, float(s)) for doc, s in zip(documents, scores)]


# ---------------------------------------------------------------------------
# Pass-through backend (no reranking — preserves original retriever order)
# ---------------------------------------------------------------------------

class _PassthroughBackend(_BaseRerankerBackend):
    """
    No-op backend. Returns documents in original order with score=0.0.
    Used when:
        • backend="passthrough" is explicitly configured.
        • API key is missing and we fall back gracefully.
        • Model loading fails at init time.
    """

    def __init__(self) -> None:
        logger.info(
            "PassthroughBackend active — documents will not be reranked."
        )

    def score(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        return [(doc, 0.0) for doc in documents]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(config: RerankerConfig) -> _BaseRerankerBackend:
    """
    Attempt to build the requested backend; fall back to passthrough on failure.
    This ensures the pipeline never hard-crashes due to a missing key or model.
    """
    backend = RerankerBackend(config.backend.lower())

    if backend == RerankerBackend.PASSTHROUGH:
        return _PassthroughBackend()

    try:
        if backend == RerankerBackend.COHERE:
            return _CohereBackend(config)
        if backend == RerankerBackend.LOCAL:
            return _LocalCrossEncoderBackend(config)
    except (ImportError, EnvironmentError, Exception) as exc:
        logger.warning(
            "Reranker: could not initialise %r backend (%s). "
            "Falling back to passthrough — documents will not be reranked.",
            config.backend, exc,
        )
        return _PassthroughBackend()

    raise ValueError(f"Unknown reranker backend: {config.backend!r}")


# ---------------------------------------------------------------------------
# Deduplication helper (content hash)
# ---------------------------------------------------------------------------

def _deduplicate(docs: List[Document]) -> List[Document]:
    import hashlib
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        h = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    removed = len(docs) - len(unique)
    if removed:
        logger.debug("Reranker: removed %d duplicate doc(s) before scoring.", removed)
    return unique


# ---------------------------------------------------------------------------
# Public DocumentReranker class
# ---------------------------------------------------------------------------

class DocumentReranker:
    """
    Cross-encoder reranking layer for the Agentic RAG pipeline.

    Takes the merged output of all three retrievers and returns the
    top_k most relevant documents, scored against the original query.

    Supports three backends:
        • Cohere API  (rerank-v3.5)           — cloud, best quality
        • Local BGE   (bge-reranker-base)     — fully offline, no API cost
        • Passthrough                         — no-op, preserves original order

    All backends use the same interface. The active backend is chosen at
    init time from config and falls back to Passthrough on any failure.

    Public API
    ──────────
    rerank(query, documents)         → List[Document]   (sync)
    async_rerank(query, documents)   → List[Document]   (async)

    Both methods:
        • Deduplicate by content hash before scoring.
        • Annotate metadata["rerank_score"] and metadata["rerank_rank"] on each doc.
        • Apply score_threshold filtering if configured.
        • Truncate to top_k.
        • Fall back to passthrough order if scoring fails.
        • Never raise — always return a (possibly empty) list.
    """

    def __init__(
        self,
        config: RerankerConfig,
        backend: Optional[_BaseRerankerBackend] = None,
    ) -> None:
        self.config   = config
        self._backend = backend or _build_backend(config)
        logger.info(
            "DocumentReranker initialised (backend=%s, top_k=%d, threshold=%s).",
            config.backend, config.top_k, config.score_threshold,
        )

    # ------------------------------------------------------------------
    # Internal scoring & sorting
    # ------------------------------------------------------------------

    def _apply_scores(
        self,
        scored_pairs: List[Tuple[Document, float]],
    ) -> List[_ScoredDocument]:
        """Convert raw (doc, score) pairs into sorted _ScoredDocument list."""
        scored = [
            _ScoredDocument(document=doc, score=s, original_index=i)
            for i, (doc, s) in enumerate(scored_pairs)
        ]
        # Sort by score descending; use original_index as deterministic tie-breaker
        scored.sort(key=lambda x: (-x.score, x.original_index))
        return scored

    def _filter_and_truncate(
        self, scored: List[_ScoredDocument]
    ) -> List[_ScoredDocument]:
        """Apply score threshold and top_k cap."""
        if self.config.score_threshold is not None:
            before = len(scored)
            scored = [s for s in scored if s.score >= self.config.score_threshold]
            removed = before - len(scored)
            if removed:
                logger.debug(
                    "Reranker: dropped %d doc(s) below threshold %.3f.",
                    removed, self.config.score_threshold,
                )
        return scored[: self.config.top_k]

    @staticmethod
    def _annotate(scored: List[_ScoredDocument]) -> List[Document]:
        """Stamp rerank_score and rerank_rank into metadata and return docs."""
        docs: List[Document] = []
        for rank, item in enumerate(scored, 1):
            item.document.metadata["rerank_score"] = round(item.score, 6)
            item.document.metadata["rerank_rank"]  = rank
            docs.append(item.document)
        return docs

    # ------------------------------------------------------------------
    # Public sync method
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Score, sort, and truncate documents by relevance to the query.

        Called by context_aggregator in graph/nodes.py after all three
        retrievers have returned and their results have been merged.

        Args:
            query:     Original user query (not rewritten — the question
                       the documents should actually answer).
            documents: Merged list from vector_store + web_search + sql_retriever.

        Returns:
            Top-k List[Document] sorted by rerank_score descending.
            Each document has:
                metadata["rerank_score"] = float
                metadata["rerank_rank"]  = int (1 = most relevant)
            Returns documents[:top_k] in original order if scoring fails.
        """
        if not documents:
            logger.debug("Reranker.rerank: received empty document list.")
            return []

        if not query or not query.strip():
            logger.warning("Reranker.rerank: empty query; returning top_k unranked docs.")
            return documents[: self.config.top_k]

        # Dedup before sending to the scoring API
        unique_docs = _deduplicate(documents)
        logger.info(
            "Reranker.rerank | backend=%s | docs=%d (after dedup) | query=%r",
            self.config.backend, len(unique_docs), query[:80],
        )

        try:
            scored_pairs = self._backend.score(query, unique_docs)
        except Exception as exc:
            logger.error(
                "Reranker.rerank: scoring failed (%s). "
                "Returning original order (top %d).",
                exc, self.config.top_k,
            )
            return unique_docs[: self.config.top_k]

        scored   = self._apply_scores(scored_pairs)
        filtered = self._filter_and_truncate(scored)
        result   = self._annotate(filtered)

        logger.info(
            "Reranker.rerank: %d → %d doc(s) after scoring + filtering.",
            len(unique_docs), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Public async method
    # ------------------------------------------------------------------

    async def async_rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Async version of rerank() — used when context_aggregator is an
        async LangGraph node.

        Wraps the sync pipeline in a thread executor for non-blocking execution.
        """
        if not documents:
            return []

        unique_docs = _deduplicate(documents)
        logger.info(
            "Reranker.async_rerank | backend=%s | docs=%d | query=%r",
            self.config.backend, len(unique_docs), query[:80],
        )

        loop = asyncio.get_event_loop()
        try:
            scored_pairs: List[Tuple[Document, float]] = await (
                self._backend.async_score(query, unique_docs)
            )
        except Exception as exc:
            logger.error(
                "Reranker.async_rerank: scoring failed (%s). "
                "Returning original order.",
                exc,
            )
            return unique_docs[: self.config.top_k]

        scored   = self._apply_scores(scored_pairs)
        filtered = self._filter_and_truncate(scored)
        result   = self._annotate(filtered)

        logger.info(
            "Reranker.async_rerank: %d → %d doc(s) after scoring.",
            len(unique_docs), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DocumentReranker(backend={self.config.backend!r}, "
            f"model={self.config.model!r}, "
            f"top_k={self.config.top_k}, "
            f"threshold={self.config.score_threshold})"
        )


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def build_reranker_from_config(config_dict: Dict[str, Any]) -> DocumentReranker:
    """
    Construct a DocumentReranker from the `retrieval.reranker` config block.

    Example:
        rr_cfg = yaml_config["retrieval"]["reranker"]
        reranker = build_reranker_from_config(rr_cfg)
    """
    cfg = RerankerConfig(
        backend=config_dict.get("backend", "cohere"),
        model=config_dict.get("model", "rerank-v3.5"),
        local_model=config_dict.get("local_model", "BAAI/bge-reranker-base"),
        top_k=int(config_dict.get("top_k", 5)),
        score_threshold=(
            float(config_dict["score_threshold"])
            if config_dict.get("score_threshold") is not None
            else None
        ),
        batch_size=int(config_dict.get("batch_size", 32)),
        cohere_api_key=config_dict.get("cohere_api_key"),
        extra_kwargs=config_dict.get("extra_kwargs", {}),
    )
    return DocumentReranker(cfg)


# ---------------------------------------------------------------------------
# Standalone smoke-test  (python -m retrieval.reranker)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Smoke-test with passthrough backend (no API key needed)
    cfg = RerankerConfig(
        backend="passthrough",
        top_k=3,
        score_threshold=None,
    )
    reranker = DocumentReranker(cfg)

    sample_docs = [
        Document(
            page_content="Product X sold 1.2 million units in Q1 2025 in North America.",
            metadata={"source": "sql", "score": 0.9},
        ),
        Document(
            page_content="Supply chain disruptions impacted product X in early 2025.",
            metadata={"source": "web", "score": 0.5},
        ),
        Document(
            page_content="Q1 2025 earnings call transcript for Company ABC.",
            metadata={"source": "vector", "score": 0.7},
        ),
        Document(
            page_content="Unrelated document about the weather forecast for June.",
            metadata={"source": "web", "score": 0.1},
        ),
    ]

    query  = "What were the total sales of Product X in Q1 2025?"
    result = reranker.rerank(query, sample_docs)

    print(f"\n=== Reranker smoke-test (backend=passthrough, top_k={cfg.top_k}) ===")
    for doc in result:
        print(
            f"  rank={doc.metadata['rerank_rank']} "
            f"score={doc.metadata['rerank_score']:.4f}  "
            f"src={doc.metadata['source']}  "
            f"{doc.page_content[:80]}"
        )