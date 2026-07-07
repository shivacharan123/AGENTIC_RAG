"""
prompts/generator.py — Final Answer Generation Layer for Agentic RAG

Responsibilities:
    • Provide LangChain runnable chain factories for every generation mode.
    • Expose format_docs() — the canonical context formatter used by the
      entire graph (nodes.py, agent/generator.py, hallucination_checker.py).
    • Support streaming via stream_answer().
    • Never import from retrieval/ — pure prompt engineering only.

Public API used by nodes.py and agent/generator.py:
    get_generator_chain(llm)          → chain: {question, context} → str
    get_fallback_chain(llm)           → chain: {question} → str
    get_conversational_chain(llm)     → chain: {question, context, conversation_history} → str
    get_regeneration_chain(llm)       → chain: {question, context, previous_answer, unsupported_claims} → str
    format_docs(documents)            → str
    format_conversation_history(hist) → str
    stream_answer(question, context, llm, history) → Iterator[str]

Public aliases used by nodes.py imports:
    rag_generator_prompt    → RAG_SYNTHESIS_PROMPT
    no_docs_fallback_prompt → NO_DOCS_FALLBACK_PROMPT
    streaming_rag_prompt    → RAG_SYNTHESIS_PROMPT
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional
from pydantic import SecretStr
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    provider: str          = "groq"
    model_name: str        = "llama-3.3-70b-versatile"
    temperature: float     = 0.0
    max_tokens: int        = 2048
    streaming: bool        = True
    groq_api_key: Optional[str] = None
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Context formatters — used across the entire graph
# ---------------------------------------------------------------------------

def format_docs(documents: List[Any]) -> str:
    """
    Format a mixed list of Documents (vector + web + SQL) into a single
    readable context block for injection into generator prompts.

    This is the canonical context formatter for the entire graph.
    Imported by nodes.py, agent/generator.py, and agent/hallucination_checker.py.

    Args:
        documents: Reranked, relevance-graded List[Document] (or any object
                   with page_content / content / metadata attributes).

    Returns:
        Formatted multi-source context string. Returns "" if empty.
    """
    if not documents:
        return ""

    formatted: List[str] = []
    for i, doc in enumerate(documents, 1):
        # Extract content — handle LangChain Document, custom Document, dict
        if hasattr(doc, "page_content"):
            content = doc.page_content
        elif hasattr(doc, "content"):
            content = doc.content
        elif isinstance(doc, dict):
            content = doc.get("page_content") or doc.get("content") or str(doc)
        else:
            content = str(doc)

        # Extract metadata
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            meta = doc.metadata
        elif isinstance(doc, dict):
            meta = doc.get("metadata", {})
        else:
            meta = {}

        source = meta.get("source", "unknown")

        # Build a source-aware label
        if source == "web":
            label = f"web | {meta.get('url', 'no-url')}"
        elif source == "sql":
            label = f"sql | table: {meta.get('table', '?')} | query: {meta.get('sql_query', '?')[:60]}"
        elif source == "vector":
            label = f"vector | id: {meta.get('doc_id', meta.get('id', '?'))}"
        else:
            label = source

        # Rerank info for transparency
        rank_info = ""
        if "rerank_rank" in meta:
            rank_info = f" | rank: {meta['rerank_rank']} score: {meta.get('rerank_score', 0.0):.3f}"

        header  = f"[Source {i} — {label}{rank_info}]"
        formatted.append(f"{header}\n{content.strip()}")

    return "\n\n---\n\n".join(formatted)


def format_conversation_history(
    history: List[Dict[str, str]],
    max_turns: int = 5,
) -> str:
    """
    Format a chat history list into a plain text block for the conversational prompt.

    Args:
        history:   List of {"role": "user"|"assistant"|"system", "content": str}.
        max_turns: Only include the last N turns to avoid context overflow.

    Returns:
        Formatted conversation string, or "No prior conversation." if empty.
    """
    if not history:
        return "No prior conversation."

    recent = history[-(max_turns * 2):]
    lines: List[str] = []
    for msg in recent:
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert synthesis AI operating inside a self-corrective "
            "Agentic RAG system.\n\n"
            "You have been given aggregated context retrieved from three sources:\n"
            "  • Dense Vector Database  (internal documents, reports, knowledge base)\n"
            "  • Web Search Engine      (live external information)\n"
            "  • SQL Database           (structured internal data and metrics)\n\n"
            "Answer generation rules:\n"
            "  1. Answer using ONLY the provided context. Do not use your internal "
            "     training knowledge to add facts not present in the context.\n"
            "  2. If the context contains conflicting data across sources, explicitly "
            "     point out the discrepancy and present both data points.\n"
            "  3. If the context does not contain enough information to answer the "
            "     question fully, state exactly what is missing instead of guessing.\n"
            "  4. Cite sources implicitly by referencing 'according to [source type]'.\n"
            "  5. Use markdown formatting: headers, bullet points, bold for key figures.\n"
            "  6. Lead with a direct, concise answer, then provide supporting detail."
        ),
    ),
    (
        "human",
        "Question: {question}\n\nAggregated Context:\n{context}",
    ),
])

NO_DOCS_FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful AI assistant inside an Agentic RAG system.\n\n"
            "The retrieval system searched the vector database, web, and SQL database "
            "but could not find any relevant documents. Multiple query-rewriting attempts "
            "were also made without success.\n\n"
            "Your response must:\n"
            "  1. Politely inform the user that no specific data was found.\n"
            "  2. Provide a brief, high-level answer from your baseline knowledge "
            "     IF the topic is general enough.\n"
            "  3. Suggest the user refine their question or check that the relevant "
            "     data source is connected.\n"
            "  4. Never fabricate specific numbers, names, or dates."
        ),
    ),
    (
        "human",
        "Question: {question}",
    ),
])

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful, context-grounded conversational AI assistant.\n\n"
            "You are continuing an ongoing conversation. Use the retrieved context "
            "to answer the current question while maintaining natural conversation flow.\n\n"
            "Rules:\n"
            "  • Stay strictly grounded in the provided context for factual claims.\n"
            "  • Reference prior conversation turns where relevant for continuity.\n"
            "  • If the context does not address the current question, say so clearly.\n"
            "  • Keep responses concise and conversational unless depth is requested."
        ),
    ),
    (
        "human",
        (
            "Conversation history:\n{conversation_history}\n\n"
            "Current question: {question}\n\n"
            "Relevant context:\n{context}"
        ),
    ),
])

REGENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert synthesis AI. A previous answer you generated was "
            "flagged for containing unsupported claims.\n\n"
            "Flagged unsupported claims:\n{unsupported_claims}\n\n"
            "Regeneration rules:\n"
            "  1. Answer ONLY from the provided context. Remove or correct every "
            "     flagged claim.\n"
            "  2. If the context cannot support a complete answer, state the gap "
            "     explicitly rather than guessing.\n"
            "  3. Do not repeat the flagged claims even in modified form."
        ),
    ),
    (
        "human",
        (
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Previous (flawed) answer:\n{previous_answer}"
        ),
    ),
])


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------

def _build_llm(config: GeneratorConfig) -> BaseChatModel:
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
# Public chain factories — imported by nodes.py and agent/generator.py
# ---------------------------------------------------------------------------

def get_generator_chain(
    llm: Optional[BaseChatModel] = None,
    config: Optional[GeneratorConfig] = None,
):
    """
    Primary RAG synthesis chain.

    Input:  {"question": str, "context": str}
    Output: str (the final generated answer)

    Usage in nodes.py:
        chain  = get_generator_chain(llm)
        answer = chain.invoke({"question": q, "context": format_docs(docs)})
    """
    config = config or GeneratorConfig()
    llm    = llm    or _build_llm(config)
    return RAG_SYNTHESIS_PROMPT | llm | StrOutputParser()


def get_fallback_chain(
    llm: Optional[BaseChatModel] = None,
    config: Optional[GeneratorConfig] = None,
):
    """
    Fallback chain used when retrieval returns no relevant documents.

    Input:  {"question": str}
    Output: str (polite fallback answer)
    """
    config = config or GeneratorConfig()
    llm    = llm    or _build_llm(config)
    return NO_DOCS_FALLBACK_PROMPT | llm | StrOutputParser()


def get_conversational_chain(
    llm: Optional[BaseChatModel] = None,
    config: Optional[GeneratorConfig] = None,
):
    """
    Multi-turn conversational chain.

    Input:  {"question": str, "context": str, "conversation_history": str}
    Output: str
    """
    config = config or GeneratorConfig()
    llm    = llm    or _build_llm(config)
    return CONVERSATIONAL_PROMPT | llm | StrOutputParser()


def get_regeneration_chain(
    llm: Optional[BaseChatModel] = None,
    config: Optional[GeneratorConfig] = None,
):
    """
    Regeneration chain called when hallucination checker fires.

    Input:  {"question": str, "context": str,
             "previous_answer": str, "unsupported_claims": str}
    Output: str (corrected answer)
    """
    config = config or GeneratorConfig()
    llm    = llm    or _build_llm(config)
    return REGENERATION_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_answer(
    question: str,
    context: str,
    llm: BaseChatModel,
    conversation_history: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream the generated answer token-by-token.

    Uses LangChain's built-in .stream() — works with any streaming LLM.

    Args:
        question:             User query.
        context:              Formatted context string from format_docs().
        llm:                  Streaming-enabled LLM instance.
        conversation_history: Optional formatted history for conversational mode.

    Yields:
        str tokens as they arrive from the LLM.
    """
    if conversation_history:
        chain   = CONVERSATIONAL_PROMPT | llm | StrOutputParser()
        payload = {
            "question":             question,
            "context":              context,
            "conversation_history": conversation_history,
        }
    else:
        chain   = RAG_SYNTHESIS_PROMPT | llm | StrOutputParser()
        payload = {"question": question, "context": context}

    try:
        yield from chain.stream(payload)
    except Exception as exc:
        logger.error("stream_answer: streaming failed: %s", exc)
        yield chain.invoke(payload)   # fallback: non-streaming invoke


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def build_generator_from_config(config_dict, llm=None):
    cfg = GeneratorConfig(
        provider=config_dict.get("provider", "groq"),
        model_name=config_dict.get("model_name", "llama-3.3-70b-versatile"),
        temperature=float(config_dict.get("temperature", 0.0)),
        max_tokens=int(config_dict.get("max_completion_tokens", 2048)),
        groq_api_key=config_dict.get("groq_api_key"),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    _llm = llm or _build_llm(cfg)
    return {
        "primary":        get_generator_chain(_llm, cfg),
        "fallback":       get_fallback_chain(_llm, cfg),
        "conversational": get_conversational_chain(_llm, cfg),
        "regeneration":   get_regeneration_chain(_llm, cfg),
        "config":         cfg,
        "llm":            _llm,
    }


# ---------------------------------------------------------------------------
# Public aliases — match the import names used in nodes.py
# ---------------------------------------------------------------------------

rag_generator_prompt    = RAG_SYNTHESIS_PROMPT
no_docs_fallback_prompt = NO_DOCS_FALLBACK_PROMPT
streaming_rag_prompt    = RAG_SYNTHESIS_PROMPT