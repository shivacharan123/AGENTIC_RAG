"""
web_search.py — Web Search Retriever for Agentic RAG

Responsibilities:
    • Integrate two search engine backends: Tavily and SerpAPI.
    • Expose a single web_search(query, num_results) → List[Document] interface
      that the retrieval pipeline calls in parallel with vector_store and sql_retriever.
    • Return results in the shared RetrievedDocument / LangChain Document schema
      so the merge + reranker step works seamlessly.
    • Handle timeouts, empty results, and API errors gracefully with fallbacks.

Position in the retrieval pipeline:
    ┌─────────────────────────────────────────────┐
    │         retrieve_node  (graph/nodes.py)     │
    └──────────┬──────────────────────────────────┘
               │  parallel fan-out
       ┌───────┼────────────────────┐
       ▼       ▼                    ▼
  vector   web_search          sql_retriever
  _store   .web_search()       .get_structured_data()
       │       │                    │
       └───────┴────────────────────┘
               │  merge + dedup
               ▼
           reranker.rerank()

Independence:
    web_search.py has NO imports from other retrieval/ files.
    It only produces Documents that feed into the merge step.

Config block (config.yaml):
    retrieval:
      web_search:
        engine: "tavily"          # "tavily" | "serpapi"
        k: 3                      # default number of results
        timeout: 10               # seconds before giving up
        max_retries: 2
        retry_delay: 1.0
        safe_search: true         # passed to SerpAPI if supported
        tavily_search_depth: "basic"   # "basic" | "advanced"
        tavily_include_domains: []     # optional whitelist
        tavily_exclude_domains: []     # optional blacklist
        serpapi_engine: "google"       # "google" | "bing" | "duckduckgo"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class SearchEngine(str, Enum):
    TAVILY  = "tavily"
    SERPAPI = "serpapi"


@dataclass
class WebSearchConfig:
    """Mirrors the `retrieval.web_search` block in config.yaml."""
    engine: str = "tavily"                    # "tavily" | "serpapi"
    k: int = 3                                # default result count
    timeout: float = 10.0                     # seconds
    max_retries: int = 2
    retry_delay: float = 1.0

    # Tavily-specific
    tavily_api_key: Optional[str] = None      # falls back to TAVILY_API_KEY env var
    tavily_search_depth: str = "basic"        # "basic" | "advanced"
    tavily_include_domains: List[str] = field(default_factory=list)
    tavily_exclude_domains: List[str] = field(default_factory=list)
    tavily_include_raw_content: bool = False  # fetch full page text

    # SerpAPI-specific
    serpapi_api_key: Optional[str] = None     # falls back to SERPAPI_API_KEY env var
    serpapi_engine: str = "google"            # "google" | "bing" | "duckduckgo"
    safe_search: bool = True


# ---------------------------------------------------------------------------
# Result schema — internal, normalised before converting to Document
# ---------------------------------------------------------------------------

@dataclass
class _RawResult:
    """Normalised result from any search engine before Document conversion."""
    title: str
    url: str
    snippet: str
    score: float = 0.0          # relevance score if provided by the API (0–1)
    raw_content: str = ""       # full page text, only populated for Tavily advanced


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------

class _BaseSearchBackend:
    """Protocol every search backend must satisfy."""

    def search(self, query: str, num_results: int) -> List[_RawResult]:
        raise NotImplementedError

    async def async_search(self, query: str, num_results: int) -> List[_RawResult]:
        """Default async wrapper — runs sync search in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.search(query, num_results)
        )


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------

class _TavilyBackend(_BaseSearchBackend):
    """
    Uses the official tavily-python SDK (TavilyClient).
    Install: pip install tavily-python
    """

    def __init__(self, config: WebSearchConfig) -> None:
        try:
            from tavily import TavilyClient  # lazy import
        except ImportError as exc:
            raise ImportError(
                "tavily-python is required for the Tavily backend. "
                "Install it with: pip install tavily-python"
            ) from exc

        api_key = config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Tavily API key not found. Set the TAVILY_API_KEY environment "
                "variable or config.tavily_api_key."
            )

        self._client = TavilyClient(api_key=api_key)
        self._config = config
        logger.info("TavilyBackend initialised (depth=%s).", config.tavily_search_depth)

    def search(self, query: str, num_results: int) -> List[_RawResult]:
        kwargs: Dict[str, Any] = {
            "query": query,
            "max_results": num_results,
            "search_depth": self._config.tavily_search_depth,
            "include_raw_content": self._config.tavily_include_raw_content,
        }
        if self._config.tavily_include_domains:
            kwargs["include_domains"] = self._config.tavily_include_domains
        if self._config.tavily_exclude_domains:
            kwargs["exclude_domains"] = self._config.tavily_exclude_domains

        logger.debug("Tavily search | query=%r | kwargs=%s", query[:80], kwargs)
        response = self._client.search(**kwargs)

        results: List[_RawResult] = []
        for item in response.get("results", []):
            results.append(
                _RawResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                    raw_content=item.get("raw_content", ""),
                )
            )
        return results


# ---------------------------------------------------------------------------
# SerpAPI backend
# ---------------------------------------------------------------------------

class _SerpAPIBackend(_BaseSearchBackend):
    """
    Uses google-search-results (SerpAPI Python SDK).
    Install: pip install google-search-results
    """

    def __init__(self, config: WebSearchConfig) -> None:
        try:
            from serpapi import GoogleSearch  # lazy import  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "google-search-results is required for the SerpAPI backend. "
                "Install it with: pip install google-search-results"
            ) from exc

        api_key = config.serpapi_api_key or os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "SerpAPI key not found. Set the SERPAPI_API_KEY environment "
                "variable or config.serpapi_api_key."
            )

        self._api_key = api_key
        self._config = config
        logger.info("SerpAPIBackend initialised (engine=%s).", config.serpapi_engine)

    def search(self, query: str, num_results: int) -> List[_RawResult]:
        from serpapi import GoogleSearch  # lazy import

        params: Dict[str, Any] = {
            "q": query,
            "num": num_results,
            "api_key": self._api_key,
            "engine": self._config.serpapi_engine,
        }
        if self._config.safe_search:
            params["safe"] = "active"

        logger.debug("SerpAPI search | query=%r | engine=%s", query[:80], self._config.serpapi_engine)
        raw = GoogleSearch(params).get_dict()

        results: List[_RawResult] = []
        for item in raw.get("organic_results", [])[:num_results]:
            results.append(
                _RawResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    score=0.0,  # SerpAPI does not return relevance scores
                )
            )
        return results


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(config: WebSearchConfig) -> _BaseSearchBackend:
    engine = SearchEngine(config.engine.lower())
    if engine == SearchEngine.TAVILY:
        return _TavilyBackend(config)
    if engine == SearchEngine.SERPAPI:
        return _SerpAPIBackend(config)
    raise ValueError(f"Unknown search engine: {config.engine!r}. Choose 'tavily' or 'serpapi'.")


# ---------------------------------------------------------------------------
# Document conversion helpers
# ---------------------------------------------------------------------------

def _result_to_document(result: _RawResult, rank: int) -> Document:
    """
    Convert a normalised _RawResult into a LangChain Document.

    page_content  = snippet (+ raw_content if available)
    metadata      = source, url, title, score, rank
    """
    # Prefer full page text when available (Tavily advanced depth)
    raw_content = result.raw_content or ""
    snippet = result.snippet or ""

    content = (
        raw_content.strip()
        if raw_content.strip()
        else snippet.strip()
    )
    # Prepend title so the reranker has context
    if result.title:
        content = f"{result.title}\n\n{content}"

    return Document(
        page_content=content,
        metadata={
            "source": "web",
            "url": result.url,
            "title": result.title,
            "score": result.score,
            "rank": rank,                    # position in raw search results
        },
    )


def _content_hash(doc: Document) -> str:
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
        logger.debug("WebSearch: removed %d duplicate result(s).", removed)
    return unique


# ---------------------------------------------------------------------------
# Public WebSearch class
# ---------------------------------------------------------------------------

class WebSearch:
    """
    Web search retriever for the Agentic RAG pipeline.

    Supports Tavily and SerpAPI backends behind a unified interface.
    All results are returned as LangChain Documents with
    metadata["source"] = "web" so they merge cleanly with vector and SQL results.

    Public API
    ──────────
    web_search(query, num_results)          → List[Document]   (sync)
    async_web_search(query, num_results)    → List[Document]   (async)

    Both methods:
        • Retry on transient failures (configurable).
        • Return an empty list on timeout / exhausted retries (never raise
          in the retrieval pipeline — caller logs the warning and moves on).
        • Deduplicate results by content hash.
        • Annotate every Document with metadata["source"] = "web".
    """

    def __init__(
        self,
        config: WebSearchConfig,
        backend: Optional[_BaseSearchBackend] = None,
    ) -> None:
        """
        Args:
            config:  WebSearchConfig instance.
            backend: Optional pre-built backend (useful for testing / mocking).
        """
        self.config  = config
        self._backend: _BaseSearchBackend = backend or _build_backend(config)
        logger.info(
            "WebSearch initialised (engine=%s, k=%d, timeout=%.1fs).",
            config.engine, config.k, config.timeout,
        )

    # ------------------------------------------------------------------
    # Internal retry wrapper
    # ------------------------------------------------------------------

    def _search_with_retry(self, query: str, num_results: int) -> List[_RawResult]:
        """
        Call the backend with retry logic.

        Returns empty list after exhausting retries instead of raising,
        consistent with the error-handling spec in the work structure.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    "WebSearch attempt %d/%d | engine=%s | query=%r",
                    attempt, self.config.max_retries, self.config.engine, query[:80],
                )
                results = self._backend.search(query, num_results)
                logger.info(
                    "WebSearch: got %d results on attempt %d.", len(results), attempt
                )
                return results

            except TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "WebSearch timeout (attempt %d/%d): %s",
                    attempt, self.config.max_retries, exc,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "WebSearch error (attempt %d/%d): %s",
                    attempt, self.config.max_retries, exc,
                )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay)

        logger.error(
            "WebSearch: all %d attempts failed. Last error: %s. "
            "Returning empty list.",
            self.config.max_retries, last_exc,
        )
        return []

    # ------------------------------------------------------------------
    # Public sync method
    # ------------------------------------------------------------------

    def web_search(
        self,
        query: str,
        num_results: Optional[int] = None,
    ) -> List[Document]:
        """
        Perform a web search and return results as LangChain Documents.

        Called by the retrieval pipeline in parallel with:
            • vector_store.similarity_search()
            • sql_retriever.get_structured_data()

        Args:
            query:       Search query string (original user query).
            num_results: Number of results to retrieve. Defaults to config.k.

        Returns:
            List[Document] where each document has:
                page_content  = title + snippet (or full page if Tavily advanced)
                metadata = {
                    "source": "web",
                    "url":    "https://...",
                    "title":  "Article title",
                    "score":  0.87,   # API relevance score (0 if not provided)
                    "rank":   1,      # position in raw search results
                }
            Returns [] on timeout or API failure (never raises).
        """
        if not query or not query.strip():
            logger.warning("WebSearch.web_search: received empty query; returning [].")
            return []

        num_results = num_results or self.config.k
        logger.info(
            "WebSearch.web_search | engine=%s | k=%d | query=%r",
            self.config.engine, num_results, query[:120],
        )

        raw_results = self._search_with_retry(query, num_results)

        if not raw_results:
            logger.warning("WebSearch: no results returned for query %r.", query[:80])
            return []

        # Convert → Document, then deduplicate
        docs = [_result_to_document(r, rank=i + 1) for i, r in enumerate(raw_results)]
        docs = _deduplicate(docs)

        logger.debug("WebSearch.web_search: returning %d deduplicated docs.", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Public async method
    # ------------------------------------------------------------------

    async def async_web_search(
        self,
        query: str,
        num_results: Optional[int] = None,
    ) -> List[Document]:
        """
        Async version of web_search — used when the retrieval pipeline
        fans out all three retrievers concurrently with asyncio.gather().

        Internally runs the sync backend in a thread executor so it
        doesn't block the event loop.

        Args / Returns: identical to web_search().
        """
        if not query or not query.strip():
            logger.warning("WebSearch.async_web_search: received empty query; returning [].")
            return []

        num_results = num_results or self.config.k
        logger.info(
            "WebSearch.async_web_search | engine=%s | k=%d | query=%r",
            self.config.engine, num_results, query[:120],
        )

        loop = asyncio.get_event_loop()
        try:
            raw_results: List[_RawResult] = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._search_with_retry(query, num_results),  # type: ignore[arg-type]
                ),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearch.async_web_search: timed out after %.1fs for query %r. "
                "Returning [].",
                self.config.timeout, query[:80],
            )
            return []

        if not raw_results:
            return []

        docs = [_result_to_document(r, rank=i + 1) for i, r in enumerate(raw_results)]
        docs = _deduplicate(docs)
        logger.debug("WebSearch.async_web_search: returning %d deduplicated docs.", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WebSearch(engine={self.config.engine!r}, "
            f"k={self.config.k}, "
            f"timeout={self.config.timeout}s)"
        )


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def build_web_search_from_config(config_dict: Dict[str, Any]) -> WebSearch:
    """
    Construct a WebSearch instance from the `retrieval.web_search` section
    of config.yaml (already parsed into a plain dict).

    Example:
        ws_cfg = yaml_config["retrieval"]["web_search"]
        ws = build_web_search_from_config(ws_cfg)
    """
    cfg = WebSearchConfig(
        engine=config_dict.get("engine", "tavily"),
        k=int(config_dict.get("k", 3)),
        timeout=float(config_dict.get("timeout", 10.0)),
        max_retries=int(config_dict.get("max_retries", 2)),
        retry_delay=float(config_dict.get("retry_delay", 1.0)),
        tavily_api_key=config_dict.get("tavily_api_key"),
        tavily_search_depth=config_dict.get("tavily_search_depth", "basic"),
        tavily_include_domains=config_dict.get("tavily_include_domains", []),
        tavily_exclude_domains=config_dict.get("tavily_exclude_domains", []),
        tavily_include_raw_content=bool(config_dict.get("tavily_include_raw_content", False)),
        serpapi_api_key=config_dict.get("serpapi_api_key"),
        serpapi_engine=config_dict.get("serpapi_engine", "google"),
        safe_search=bool(config_dict.get("safe_search", True)),
    )
    return WebSearch(cfg)


# ---------------------------------------------------------------------------
# Standalone smoke-test  (python -m retrieval.web_search)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from typing import Dict, Any

    logging.basicConfig(level=logging.DEBUG)

    engine = os.environ.get("SEARCH_ENGINE", "tavily")   # override via env
    cfg = WebSearchConfig(
        engine=engine,
        k=3,
        timeout=10.0,
        tavily_search_depth="basic",
    )

    ws = WebSearch(cfg)
    query = "What were the total sales of product X in Q1 2025?"

    print(f"\n=== WebSearch smoke-test (engine={engine}) ===")
    results = ws.web_search(query, num_results=3)

    if not results:
        print("[WARN] No results returned — check your API key.")
    else:
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.metadata['title']}")
            print(f"    URL  : {doc.metadata['url']}")
            print(f"    Score: {doc.metadata['score']}")
            print(f"    Text : {doc.page_content[:120]}…")