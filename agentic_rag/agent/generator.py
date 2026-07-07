"""
agents/generator.py — Answer Synthesis Agent

Responsibilities:
    • Orchestrate the full answer generation lifecycle inside the LangGraph node.
    • Select the correct generation mode based on GraphState:
        - "rag"            → primary: synthesise from aggregated multi-source context.
        - "regenerate"     → fix a hallucinated answer using flagged unsupported claims.
        - "conversational" → multi-turn chat grounded in retrieved context.
        - "fallback"       → all retrieval retries exhausted; polite baseline answer.
    • Support both sync (.generate()) and streaming (.stream_generate()) for real-time UI.
    • Never fabricate — enforce strict context-only instructions in every prompt.

Relationship to prompts/generator.py:
    prompts/generator.py   → raw prompt templates + format_docs() utility +
                             individual chain factories (get_generator_chain, etc.)
    agents/generator.py    → the ORCHESTRATOR that chooses which chain to invoke,
                             handles mode routing, exposes the unified .generate()
                             interface consumed by nodes.py, and adds generation
                             metadata to GraphState.

Position in the Agentic RAG graph:
    relevance_grader returns "pass"
           │
           │  relevant_documents: List[Document]
           │  generation_mode: "rag" | "regenerate" | "conversational" | "fallback"
           ▼
    ┌─────────────────────────────────────────────┐
    │           generator node                    │  ← THIS FILE
    │                                            │
    │  GeneratorAgent.generate()                 │
    │    ├─ mode="rag"            → rag chain    │
    │    ├─ mode="regenerate"     → regen chain  │
    │    ├─ mode="conversational" → conv chain   │
    │    └─ mode="fallback"       → fallback     │
    └─────────────────────────────────────────────┘
           │
           │  generated_answer: str
           │  generation_metadata: Dict
           ▼
    hallucination_checker node

Interfaces consumed by graph/nodes.py:
    • GeneratorAgent.generate(state)       → GenerationResult
    • GeneratorAgent.stream_generate(state) → Iterator[str]
    • get_generator_agent(llm, config)     → GeneratorAgent
    • GenerationResult                     → typed output schema
    • GENERATION_MODES                     → valid mode strings

Config block (config.yaml):
    agents:
      generator:
        provider: "openrouter"
        model_name: "anthropic/claude-3.5-sonnet"
        temperature: 0.0
        max_tokens: 2048
        streaming: true
        openrouter_base_url: "https://openrouter.ai/api/v1"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr

# Import canonical prompts and utilities from the prompts layer
from agentic_rag.prompts.generator import (
    RAG_SYNTHESIS_PROMPT,
    NO_DOCS_FALLBACK_PROMPT,
    CONVERSATIONAL_PROMPT,
    REGENERATION_PROMPT,
    format_docs,
    format_conversation_history,
    get_generator_chain,
    get_fallback_chain,
    get_conversational_chain,
    get_regeneration_chain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation mode constants — single source of truth for nodes.py
# ---------------------------------------------------------------------------

MODE_RAG            = "rag"
MODE_REGENERATE     = "regenerate"
MODE_CONVERSATIONAL = "conversational"
MODE_FALLBACK       = "fallback"

GENERATION_MODES = {MODE_RAG, MODE_REGENERATE, MODE_CONVERSATIONAL, MODE_FALLBACK}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GeneratorAgentConfig:
    provider: str          = "groq"
    model_name: str        = "llama-3.3-70b-versatile"
    temperature: float     = 0.0
    max_tokens: int        = 2048
    streaming: bool        = True
    groq_api_key: Optional[str] = None
    max_context_docs: int  = 10
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Typed generation result — stored in GraphState
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    """
    Structured output from the generator agent.

    answer is the only field the final user sees.
    All other fields are stored in GraphState for downstream nodes
    (hallucination_checker, answer_quality_grader) and for logging.
    """

    answer: str = Field(
        description="The final generated answer. May be empty string on hard failure."
    )

    mode_used: str = Field(
        description=(
            "Which generation mode was actually used: "
            "'rag', 'regenerate', 'conversational', or 'fallback'."
        )
    )

    context_doc_count: int = Field(
        default=0,
        description="Number of context documents that were fed to the generator.",
    )

    sources_cited: List[str] = Field(
        default_factory=list,
        description=(
            "Unique source types present in the context that the generator had access to. "
            "E.g. ['vector', 'web', 'sql']."
        ),
    )

    fallback_used: bool = Field(
        default=False,
        description="True if the generator had to fall back to a different mode than requested.",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if generation failed; None on success.",
    )

    @property
    def succeeded(self) -> bool:
        return bool(self.answer) and self.error is None


# ---------------------------------------------------------------------------
# LLM builder  ← module-level function, NOT inside GenerationResult
# ---------------------------------------------------------------------------

def _build_llm(config: GeneratorAgentConfig) -> BaseChatModel:
    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found.")
    return ChatGroq(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=SecretStr(api_key),
    )

    


# ---------------------------------------------------------------------------
# Helper: extract sources from document list
# ---------------------------------------------------------------------------

def _extract_sources(documents: List[Document]) -> List[str]:
    return sorted({doc.metadata.get("source", "unknown") for doc in documents})


# ---------------------------------------------------------------------------
# Stateful generator agent
# ---------------------------------------------------------------------------

class GeneratorAgent:
    """
    Orchestrates the full answer generation lifecycle for the LangGraph node.

    Mode selection logic (applied automatically inside .generate()):

        state has no relevant_documents OR mode = "fallback"
            → fallback chain  (polite "no data found" response)

        mode = "regenerate"  (hallucination_checker returned "regenerate")
            → regeneration chain (uses unsupported_claims to target corrections)

        state has conversation_history AND mode = "conversational"
            → conversational chain  (multi-turn grounded chat)

        default
            → RAG synthesis chain  (primary path)

    All modes fall back to the RAG synthesis chain if their specific
    chain fails, and fall back to the fallback chain if RAG also fails.

    Usage in nodes.py:
        agent = GeneratorAgent(llm, config)
        result: GenerationResult = agent.generate(state)
        state["generated_answer"]     = result.answer
        state["generation_metadata"]  = result..model_dump()
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[GeneratorAgentConfig] = None,
    ) -> None:
        self.config = config or GeneratorAgentConfig()
        self._llm   = llm or _build_llm(self.config)

        # Pre-build all chains at init — not per-call
        self._rag_chain   = get_generator_chain(self._llm)
        self._regen_chain = get_regeneration_chain(self._llm)
        self._conv_chain  = get_conversational_chain(self._llm)
        self._fall_chain  = get_fallback_chain(self._llm)

        logger.info(
            "GeneratorAgent initialised (provider=%s, model=%s, max_context_docs=%d).",
            self.config.provider, self.config.model_name, self.config.max_context_docs,
        )

    # ------------------------------------------------------------------
    # Public: sync generation
    # ------------------------------------------------------------------

    def generate(self, state: Dict[str, Any]) -> GenerationResult:
        """
        Generate a final answer from GraphState.

        Reads from state:
            state["user_query"]           str   — original question
            state["relevant_documents"]   List[Document] — graded context
            state["generation_mode"]      str   — "rag"|"regenerate"|"conversational"|"fallback"
            state["generated_answer"]     str   — previous answer (for regenerate mode)
            state["hallucination_grade"]  GradeHallucinations — (for regenerate mode)
            state["chat_history"]         List[Dict] — (for conversational mode)
            state["regeneration_count"]   int   — how many times we've already regenerated

        Returns:
            GenerationResult with answer, mode_used, context_doc_count, sources_cited.

        Falls back gracefully:
            regenerate → rag → fallback (each step only if the previous fails)
        """
        question     = state.get("user_query", "")
        documents    = state.get("relevant_documents", [])
        mode         = state.get("generation_mode", MODE_RAG)
        chat_history = state.get("chat_history", [])

        if not question:
            return GenerationResult(
                answer="I'm sorry, I couldn't understand your question.",
                mode_used=MODE_FALLBACK,
                fallback_used=True,
                error="Empty question in GraphState.",
            )

        # Truncate context if it exceeds max_context_docs
        if len(documents) > self.config.max_context_docs:
            logger.debug(
                "GeneratorAgent: truncating %d docs to max_context_docs=%d.",
                len(documents), self.config.max_context_docs,
            )
            documents = documents[: self.config.max_context_docs]

        sources = _extract_sources(documents)
        context = format_docs(documents)

        # ------------------------------------------------------------------
        # Mode: FALLBACK — no documents available
        # ------------------------------------------------------------------
        if mode == MODE_FALLBACK:
            logger.info("GeneratorAgent: explicit fallback mode requested.")
            return self._run_fallback(question)
        if not documents:
            logger.info("GeneratorAgent: no documents → fallback.")
            return self._run_fallback(question)

        # ------------------------------------------------------------------
        # Mode: REGENERATE — fix a hallucinated answer
        # ------------------------------------------------------------------
        if mode == MODE_REGENERATE:
            return self._run_regenerate(question, context, state, sources, len(documents))

        # ------------------------------------------------------------------
        # Mode: CONVERSATIONAL — multi-turn chat
        # ------------------------------------------------------------------
        if mode == MODE_CONVERSATIONAL and chat_history:
            return self._run_conversational(
                question, context, chat_history, sources, len(documents)
            )

        # ------------------------------------------------------------------
        # Mode: RAG (default primary path)
        # ------------------------------------------------------------------
        return self._run_rag(question, context, sources, len(documents))

    # ------------------------------------------------------------------
    # Public: streaming generation
    # ------------------------------------------------------------------

    def stream_generate(self, state: Dict[str, Any]) -> Iterator[str]:
        """
        Stream the generated answer token-by-token for real-time UI delivery.

        Reads same state keys as generate().
        Falls back to non-streaming generate() if streaming fails.

        Usage in FastAPI / Streamlit:
            for token in agent.stream_generate(state):
                print(token, end="", flush=True)
        """
        question  = state.get("user_query", "")
        documents = state.get("relevant_documents", [])
        mode      = state.get("generation_mode", MODE_RAG)

        if not documents or mode == MODE_FALLBACK:
            result = self._run_fallback(question)
            yield result.answer
            return

        context = format_docs(documents[: self.config.max_context_docs])

        # Choose chain
        if mode == MODE_REGENERATE:
            chain   = self._regen_chain
            payload = self._regen_payload(question, context, state)
        elif mode == MODE_CONVERSATIONAL and state.get("chat_history"):
            chain   = self._conv_chain
            payload = {
                "question":             question,
                "context":              context,
                "conversation_history": format_conversation_history(
                    state.get("chat_history", [])
                ),
            }
        else:
            chain   = self._rag_chain
            payload = {"question": question, "context": context}

        try:
            yield from chain.stream(payload)
        except Exception as exc:
            logger.error("GeneratorAgent.stream_generate: streaming failed (%s); falling back.", exc)
            result = self.generate(state)
            yield result.answer

    # ------------------------------------------------------------------
    # Private mode runners
    # ------------------------------------------------------------------

    def _run_rag(
        self,
        question: str,
        context: str,
        sources: List[str],
        doc_count: int,
    ) -> GenerationResult:
        logger.info("GeneratorAgent: RAG synthesis | sources=%s | docs=%d", sources, doc_count)
        try:
            answer: str = self._rag_chain.invoke({
                "question": question,
                "context":  context,
            })
            return GenerationResult(
                answer=answer,
                mode_used=MODE_RAG,
                context_doc_count=doc_count,
                sources_cited=sources,
            )
        except Exception as exc:
            logger.error("GeneratorAgent._run_rag failed (%s) → fallback.", exc)
            result = self._run_fallback(question)
            return result.model_copy(update={
                "fallback_used": True,
                "error": str(exc),
            })

    def _run_regenerate(
        self,
        question: str,
        context: str,
        state: Dict[str, Any],
        sources: List[str],
        doc_count: int,
    ) -> GenerationResult:
        logger.info(
            "GeneratorAgent: regeneration mode | attempt=%d",
            state.get("regeneration_count", 0),
        )
        payload = self._regen_payload(question, context, state)
        try:
            answer: str = self._regen_chain.invoke(payload)
            return GenerationResult(
                answer=answer,
                mode_used=MODE_REGENERATE,
                context_doc_count=doc_count,
                sources_cited=sources,
            )
        except Exception as exc:
            logger.error(
                "GeneratorAgent._run_regenerate failed (%s) → falling back to RAG.", exc
            )
            result = self._run_rag(question, context, sources, doc_count)
            return result.model_copy(update={
                "fallback_used": True,
                "error": str(exc),
            })

    def _run_conversational(
        self,
        question: str,
        context: str,
        chat_history: List[Dict[str, str]],
        sources: List[str],
        doc_count: int,
    ) -> GenerationResult:
        logger.info(
            "GeneratorAgent: conversational mode | history_turns=%d", len(chat_history)
        )
        try:
            answer: str = self._conv_chain.invoke({
                "question":             question,
                "context":              context,
                "conversation_history": format_conversation_history(chat_history),
            })
            return GenerationResult(
                answer=answer,
                mode_used=MODE_CONVERSATIONAL,
                context_doc_count=doc_count,
                sources_cited=sources,
            )
        except Exception as exc:
            logger.error(
                "GeneratorAgent._run_conversational failed (%s) → falling back to RAG.", exc
            )
            result = self._run_rag(question, context, sources, doc_count)
            return result.model_copy(update={
                "fallback_used": True,
                "error": str(exc),
            })

    def _run_fallback(self, question: str) -> GenerationResult:
        logger.info("GeneratorAgent: fallback mode (no relevant documents).")
        try:
            answer: str = self._fall_chain.invoke({"question": question})
            return GenerationResult(
                answer=answer,
                mode_used=MODE_FALLBACK,
                context_doc_count=0,
                sources_cited=[],
                fallback_used=True,
            )
        except Exception as exc:
            logger.error("GeneratorAgent._run_fallback also failed (%s).", exc)
            return GenerationResult(
                answer=(
                    "I'm sorry, I was unable to find relevant information to answer "
                    "your question, and the fallback response also failed. "
                    "Please try rephrasing your query."
                ),
                mode_used=MODE_FALLBACK,
                fallback_used=True,
                error=str(exc),
            )

    @staticmethod
    def _regen_payload(
        question: str,
        context: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the payload for the regeneration chain from GraphState."""
        previous_answer     = state.get("generated_answer", "")
        hallucination_grade = state.get("hallucination_grade")

        # prompts/grader.py defines GradeHallucinations:
        #   - binary_score: 'yes'|'no'
        #   - unsupported_claims: list[str]
        unsupported_claims_list: List[str] = []
        if hallucination_grade is not None:
            unsupported_claims_list = getattr(hallucination_grade, "unsupported_claims", []) or []

        unsupported_claims = (
            "\n".join(f"- {c}" for c in unsupported_claims_list)
            if unsupported_claims_list
            else "No specific claims identified."
        )

        return {
            "question":            question,
            "context":             context,
            "previous_answer":     previous_answer,
            "unsupported_claims":  unsupported_claims,
        }


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def build_generator_agent_from_config(
    config_dict: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
) -> GeneratorAgent:
    cfg = GeneratorAgentConfig(
        provider=config_dict.get("provider", "groq"),           # ← was "gemini"
        model_name=config_dict.get("model_name", "llama-3.3-70b-versatile"),  # ← was "gemini-2.0-flash"
        temperature=float(config_dict.get("temperature", 0.0)),
        max_tokens=int(config_dict.get("max_completion_tokens", 2048)),
        streaming=bool(config_dict.get("streaming", True)),
        groq_api_key=config_dict.get("groq_api_key"),           # ← was gemini_api_key
        max_context_docs=int(config_dict.get("max_context_docs", 10)),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    return GeneratorAgent(llm=llm, config=cfg)


# Quick chain access for nodes.py that prefers direct chain usage
def get_generator_agent(
    llm: Optional[BaseChatModel] = None,
    config: Optional[GeneratorAgentConfig] = None,
) -> GeneratorAgent:
    """Convenience alias matching the blueprint's get_generator() pattern."""
    return GeneratorAgent(llm=llm, config=config)


# ---------------------------------------------------------------------------
# Standalone smoke-test  (python -m agents.generator)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    _llm = ChatGroq(                              # ← was ChatGoogleGenerativeAI
        model="llama-3.3-70b-versatile",             # ← was "gemini-2.0-flash"
        temperature=0.0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),   # ← was GEMINI_API_KEY
    )
    _agent = GeneratorAgent(
        _llm,
        GeneratorAgentConfig(
            provider="groq",                      # ← was "gemini"
            model_name="llama-3.3-70b-versatile",    # ← was "gemini-2.0-flash"
            streaming=False,
        ),
    )
    # rest stays the same

    _docs = [
        Document(
            page_content="Product X sold 1.2 million units in Q1 2025, generating $4.8M revenue.",
            metadata={"source": "sql", "rerank_rank": 1, "rerank_score": 0.97},
        ),
        Document(
            page_content="Analysts praised Product X's North America momentum in early 2025.",
            metadata={"source": "web", "rerank_rank": 2, "rerank_score": 0.81},
        ),
    ]

    print("\n=== generator.py (agent) smoke-test ===")

    # Mode: RAG
    state_rag = {
        "user_query":         "What were the total sales of Product X in Q1 2025?",
        "relevant_documents": _docs,
        "generation_mode":    MODE_RAG,
    }
    result = _agent.generate(state_rag)
    print(f"\n[RAG MODE]\n  mode_used={result.mode_used}  sources={result.sources_cited}")
    print(f"  answer: {result.answer[:200]}")
    
    # Mode: FALLBACK
    state_fallback = {
        "user_query":         "What is the market cap of XYZ Corp?",
        "relevant_documents": [],
        "generation_mode":    MODE_FALLBACK,
    }
    result_fb = _agent.generate(state_fallback)
    print(f"\n[FALLBACK MODE]\n  mode_used={result_fb.mode_used}")
    print(f"  answer: {result_fb.answer[:200]}")