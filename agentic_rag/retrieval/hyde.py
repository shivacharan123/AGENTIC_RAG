"""
hyde.py — Hypothetical Document Embeddings (HyDE) for Agentic RAG

Workflow:
    1. Receive the original user query from VectorStore.retrieve().
    2. Use an LLM to generate a hypothetical expert answer.
    3. Return the hypothetical answer as a plain string.
    4. VectorStore passes that text to similarity_search() to retrieve
       semantically similar real documents.
    5. If generation fails, VectorStore falls back to the original query.

Architecture:
    graph/nodes.py
          │
          ▼
    VectorStore.retrieve()
          │
          ▼
    HyDE.generate_hypothetical_answer()
          │
          ▼
    VectorStore.similarity_search()
          │
          ▼
    List[Document]

Dependencies (retrieval-internal):
    • HyDE does NOT retrieve documents itself during the normal pipeline.
    • HyDE is used internally by VectorStore.retrieve() when use_hyde=True.
    • graph/nodes.py never calls HyDE directly.
"""

from __future__ import annotations
import os
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr
if TYPE_CHECKING:
    # Avoid circular import at runtime; vector_store imports hyde
    from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (populated from config.yaml at module level by the caller)
# ---------------------------------------------------------------------------

@dataclass
class HyDEConfig:
    """Mirrors the `retrieval.hyde` block in config.yaml."""
    enabled: bool          = True
    model: str             = "llama-3.3-70b-versatile"
    temperature: float     = 0.7
    max_tokens: int        = 512
    max_retries: int       = 2
    retry_delay: float     = 1.0
    system_prompt: str     = (
        "You are an expert assistant. "
        "Answer the following question concisely and factually, "
        "as if you were writing a passage in a reference document. "
        "Do NOT say 'I don't know'; always provide a plausible, detailed answer."
    )
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Core HyDE class
# ---------------------------------------------------------------------------

class HyDE:
    """
    Hypothetical Document Embeddings generator.

    Usage (standalone):
        hyde = HyDE(config)
        hyp_doc: str = hyde.generate_hypothetical_answer(query)

    Usage (integrated with VectorStore):
    hyde = HyDE(config)

    hypothetical_doc = hyde.generate_hypothetical_answer(query)

    # VectorStore.retrieve() internally calls this method
    # and then performs similarity_search(hypothetical_doc).
    """

    def __init__(self, config: HyDEConfig, llm: Optional[BaseChatModel] = None) -> None:
        """
        Args:
            config:  HyDEConfig instance.
            llm:     Optional pre-built LLM; if None, one is constructed from config.
        """
        self.config = config
        self._llm: BaseChatModel = llm or self._build_llm()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_llm(self):
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found — required for HyDE LLM.")
        return ChatGroq(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=SecretStr(api_key),
        )

    def _call_llm(self, query: str) -> str:
        """
        Send the generation prompt to the LLM with retry logic.

        Returns the generated text, or raises RuntimeError after exhausting retries.
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user",   "content": query},
        ]

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug("HyDE LLM call — attempt %d/%d", attempt, self.config.max_retries)
                response = self._llm.invoke(messages)
                # LangChain returns an AIMessage; extract string content.
                if hasattr(response, "content"):
                    return str(response.content).strip()
                return str(response).strip()

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "HyDE LLM call failed (attempt %d/%d): %s",
                    attempt, self.config.max_retries, exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)

        raise RuntimeError(
            f"HyDE: LLM failed after {self.config.max_retries} attempt(s)."
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical expert answer for *query*.

        This synthetic text is NOT shown to the user; it is only used as an
        embedding proxy to improve vector retrieval quality.

        Args:
            query: The original user question / search query.

        Returns:
            A plausible hypothetical answer as a plain string.

        Raises:
            RuntimeError: If all LLM retry attempts fail (caller should fall back).
        """
        if not query or not query.strip():
            raise ValueError("HyDE: query must be a non-empty string.")

        logger.info("HyDE generating hypothetical answer for query: %r", query[:120])
        hypothetical = self._call_llm(query)
        logger.debug("HyDE hypothetical answer (%d chars): %s…", len(hypothetical), hypothetical[:200])
        return hypothetical

    def retrieve(
        self,
        query: str,
        vector_store: "VectorStore",
        k: int = 5,
    ) -> List[Document]:
        """
        Full HyDE pipeline: generate hypothetical doc → vector search.

        Args:
            query:        Original user query.
            vector_store: Initialised VectorStore instance.
            k:            Number of documents to return.

        Returns:
            List of retrieved LangChain Document objects, each with
            metadata["source"] = "vector" and metadata["hyde"] = True.
            Falls back to a plain similarity search on *query* if generation fails.
        """
        # --- Step 1: Generate hypothetical answer --------------------------------
        try:
            hypothetical_doc = self.generate_hypothetical_answer(query)
            used_hyde = True
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "HyDE: falling back to original query for vector search. Reason: %s", exc
            )
            hypothetical_doc = query
            used_hyde = False

        # --- Step 2: Retrieve real documents using the hypothetical text ----------
        logger.info(
            "HyDE vector search (used_hyde=%s, k=%d).", used_hyde, k
        )
        docs: List[Document] = vector_store.similarity_search(hypothetical_doc, k=k)

        # --- Step 3: Annotate metadata -------------------------------------------
        for doc in docs:
            doc.metadata.setdefault("source", "vector")
            doc.metadata["hyde"] = used_hyde
            doc.metadata["original_query"] = query

        return docs


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def build_hyde_from_config(config_dict: Dict[str, Any]) -> HyDE:
    """
    Construct a HyDE instance from the `retrieval.hyde` section of config.yaml
    (already parsed into a plain dict).

    Example:
        hyde_cfg = yaml_config["retrieval"]["hyde"]
        hyde = build_hyde_from_config(hyde_cfg)
    """
    cfg = HyDEConfig(
        enabled=config_dict.get("enabled", True),
        model=config_dict.get("model", "llama-3.3-70b-versatile"),
        temperature=float(config_dict.get("temperature", 0.7)),
        max_tokens=int(config_dict.get("max_tokens", 512)),
        max_retries=int(config_dict.get("max_retries", 2)),
        retry_delay=float(config_dict.get("retry_delay", 1.0)),
        system_prompt=config_dict.get(       # ← replace HyDEConfig.system_prompt
            "system_prompt",
            (
                "You are an expert assistant. "
                "Answer the following question concisely and factually, "
                "as if you were writing a passage in a reference document. "
                "Do NOT say 'I don't know'; always provide a plausible, detailed answer."
            ),
        ),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    return HyDE(cfg)


# ---------------------------------------------------------------------------
# Standalone smoke-test (python -m retrieval.hyde)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    

    logging.basicConfig(level=logging.DEBUG)

    # Minimal config for local testing — requires OPENAI_API_KEY in env.
    # Optional — update smoke-test too:
    test_config = HyDEConfig(
        enabled=True,
        model=os.getenv("HYDE_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.7,
    )
    hyde = HyDE(test_config)
    sample_query = "What were the total sales of product X in Q1 2025?"

    try:
        answer = hyde.generate_hypothetical_answer(sample_query)
        print("\n=== Hypothetical Answer ===")
        print(answer)
    except RuntimeError as e:
        print(f"[ERROR] {e}")