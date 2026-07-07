"""
prompts/query_transform.py — Query Transformation Layer for Agentic RAG

Responsibilities:
    • Supply prompt templates for every query mutation strategy the graph uses.
    • Expose LangChain runnable chains (not raw strings) so graph/nodes.py can
      call them with a single .invoke() / .stream().
    • Enforce structured output where the result must be machine-parseable
      (decomposition → List[str], rewrite → str).
    • Never import from retrieval/ — this layer is pure prompt engineering.

Query transformation strategies supported:
    1. Query Rewriting      — tighten + focus the original query for fan-out search.
    2. Step-Back Prompting  — generalise to a broader parent question first.
    3. Sub-Query Decomposition — split complex multi-hop questions into atomics.

Position in the Agentic RAG graph:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     LangGraph nodes.py                         │
    └─────────────────────────────────┬───────────────────────────────┘
                                      │
              relevance_grader returns "retry" / all docs failed
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   query_transform node   │
                        │                         │
                        │  rewrite_chain          │  ← primary: tighten query
                        │  stepback_chain         │  ← optional: broaden scope
                        │  decompose_chain        │  ← complex multi-hop queries
                        └─────────────────────────┘
                                      │
                   new query / sub-queries injected into GraphState
                                      │
                                      ▼
                        parallel retrieval fan-out (again)

Config block (config.yaml):
    query:
      transform:
        max_subqueries: 4
        rewrite_model: "gpt-4o-mini"
        rewrite_temperature: 0.0
        decompose_model: "gpt-4o-mini"
        decompose_temperature: 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QueryTransformConfig:
    """Mirrors the `query.transform` block in config.yaml."""
    max_subqueries: int        = 4
    rewrite_model: str         = "gpt-4o-mini"
    rewrite_temperature: float = 0.0      # deterministic — one best rewrite
    decompose_model: str       = "gpt-4o-mini"
    decompose_temperature: float = 0.3    # slight creativity for sub-questions
    stepback_model: str        = "gpt-4o-mini"
    stepback_temperature: float = 0.2
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic structured output schemas
# ---------------------------------------------------------------------------

class RewrittenQuery(BaseModel):
    """Structured output for query rewriting — guarantees a clean string."""
    rewritten_query: str = Field(
        description=(
            "The rewritten, optimised query. Must be a single focused question or "
            "search phrase with no explanations, quotes, or preamble."
        )
    )
    reasoning: str = Field(
        default="",
        description="Internal reasoning about what was changed and why (not shown to user).",
    )


class DecomposedQueries(BaseModel):
    """Structured output for sub-query decomposition."""
    sub_queries: List[str] = Field(
        description=(
            "List of independent sub-questions that together cover all aspects "
            "of the original query. Each is self-contained and directly searchable."
        ),
        min_length=1,
    )
    reasoning: str = Field(
        default="",
        description="Why the query was split this way.",
    )


class StepBackQuery(BaseModel):
    """Structured output for step-back prompting."""
    stepback_query: str = Field(
        description=(
            "A broader, more general version of the original question that "
            "retrieves high-level background knowledge useful for answering the specific query."
        )
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# --- 1. Query Rewriting ---
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert query optimiser operating inside a self-corrective "
            "retrieval loop for an Agentic RAG system.\n\n"
            "The previous retrieval attempt failed to find sufficiently relevant context. "
            "Your job is to rewrite the user's original question so it performs better "
            "across ALL THREE of the following search backends simultaneously:\n\n"
            "  1. Dense Vector Database  — needs semantic richness and domain terminology.\n"
            "  2. Web Search Engine      — needs concrete keywords, entities, dates, metrics.\n"
            "  3. SQL Database           — needs specific entity names, column values, time ranges.\n\n"
            "Rewriting rules:\n"
            "  • Strip conversational filler (e.g. 'Can you tell me...', 'I was wondering...').\n"
            "  • Surface key entities, proper nouns, dates, and numeric targets explicitly.\n"
            "  • Collapse multi-part questions into ONE highly focused core question.\n"
            "  • Preserve the original intent — do not change what is being asked.\n"
            "  • Return ONLY the rewritten query string in the `rewritten_query` field."
        ),
    ),
    (
        "human",
        (
            "Original question: {question}\n\n"
            "Retrieval attempt #{retry_count} failed. "
            "Failure reason hint: {failure_reason}"
        ),
    ),
])

# --- 2. Step-Back Prompting ---
STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert at abstract reasoning within an Agentic RAG system.\n\n"
            "Your task is to generate a STEP-BACK question — a broader, more general "
            "version of the user's specific question.\n\n"
            "The step-back question is used to first retrieve high-level background "
            "knowledge (principles, context, definitions) before attempting to answer "
            "the specific question. This improves the quality of the final answer for "
            "complex or technical queries.\n\n"
            "Rules:\n"
            "  • Make the step-back question more abstract / general than the original.\n"
            "  • It should retrieve foundational knowledge that HELPS answer the specific query.\n"
            "  • Keep it as a single question.\n"
            "  • Return ONLY the step-back question in `stepback_query`."
        ),
    ),
    (
        "human",
        "Specific question: {question}",
    ),
])

# --- 3. Sub-Query Decomposition ---
DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert at decomposing complex multi-hop questions for a "
            "parallel retrieval system.\n\n"
            "Break the user's question into {max_subqueries} or fewer INDEPENDENT "
            "sub-questions. These will each be sent to the retrieval fan-out separately "
            "(vector search, web search, SQL) and their results merged before answering.\n\n"
            "Decomposition rules:\n"
            "  • Each sub-question must be SELF-CONTAINED — answerable on its own.\n"
            "  • Collectively they must cover ALL important aspects of the original.\n"
            "  • NO overlap between sub-questions.\n"
            "  • Make each specific, concrete, and directly searchable.\n"
            "  • If the original is already simple and atomic, return it as a single-item list.\n"
            "  • Return the list in the `sub_queries` field."
        ),
    ),
    (
        "human",
        "Original question: {question}",
    ),
])


# ---------------------------------------------------------------------------
# Chain builder helpers
# ---------------------------------------------------------------------------

def _build_llm(
    model: str,
    temperature: float,
    extra_kwargs: Dict[str, Any],
) -> BaseChatModel:
    """
    Deprecated fallback.

    Kept only for backwards compatibility, but query chain factories should now
    require an explicit llm to prevent accidentally calling OpenAI directly.
    """
    return ChatOpenAI(model=model, temperature=temperature, **extra_kwargs)


# ---------------------------------------------------------------------------
# Public chain factories
# (return LangChain Runnables — graph/nodes.py calls .invoke() on them)
# ---------------------------------------------------------------------------

def get_query_rewriter_chain(
    llm: BaseChatModel,
    config: Optional[QueryTransformConfig] = None,
) -> Any:
    """
    Returns a runnable chain:
        Input : {"question": str, "retry_count": int, "failure_reason": str}
        Output: RewrittenQuery (Pydantic — access .rewritten_query)

    Called in the query_transform node when the relevance grader
    returns "retry" after a retrieval attempt.

    Usage in nodes.py:
        chain = get_query_rewriter_chain(llm)
        result: RewrittenQuery = chain.invoke({
            "question": state["user_query"],
            "retry_count": state["retry_count"],
            "failure_reason": "No documents passed relevance grading.",
        })
        new_query = result.rewritten_query
    """
    config = config or QueryTransformConfig()
    structured_llm = llm.with_structured_output(RewrittenQuery)
    return QUERY_REWRITE_PROMPT | structured_llm


def get_stepback_chain(
    llm: BaseChatModel,
    config: Optional[QueryTransformConfig] = None,
) -> Any:
    """
    Returns a runnable chain:
        Input : {"question": str}
        Output: StepBackQuery (Pydantic — access .stepback_query)

    Used as an optional pre-retrieval step for complex technical queries.
    The step-back query runs through the vector store first to build
    background context, then the specific query is sent to the full fan-out.

    Usage in nodes.py:
        chain = get_stepback_chain(llm)
        result: StepBackQuery = chain.invoke({"question": state["user_query"]})
        background_query = result.stepback_query
    """
    config = config or QueryTransformConfig()
    structured_llm = llm.with_structured_output(StepBackQuery)
    return STEP_BACK_PROMPT | structured_llm


def get_decomposition_chain(
    llm: BaseChatModel,
    config: Optional[QueryTransformConfig] = None,
) -> Any:
    """
    Returns a runnable chain:
        Input : {"question": str, "max_subqueries": int}
        Output: DecomposedQueries (Pydantic — access .sub_queries → List[str])

    Used for complex multi-hop questions that need multiple independent
    retrieval passes. The graph fans out each sub-query separately.

    Usage in nodes.py:
        chain = get_decomposition_chain(llm)
        result: DecomposedQueries = chain.invoke({
            "question": state["user_query"],
            "max_subqueries": 4,
        })
        for sub_q in result.sub_queries:
            # run retrieval fan-out for each
    """
    config = config or QueryTransformConfig()
    structured_llm = llm.with_structured_output(DecomposedQueries)
    return DECOMPOSITION_PROMPT | structured_llm


# ---------------------------------------------------------------------------
# Convenience: run all transformations in one call
# ---------------------------------------------------------------------------

def run_all_transforms(
    question: str,
    llm: BaseChatModel,
    config: Optional[QueryTransformConfig] = None,
    retry_count: int = 1,
    failure_reason: str = "No relevant documents retrieved.",
    decompose: bool = True,
    stepback: bool = False,
) -> Dict[str, Any]:
    """
    Run all active query transformation strategies and return a unified dict.

    This is the convenience function called by the query_transform node
    when it wants all transformations in a single step.

    Args:
        question:       Original user query from GraphState.
        llm:            Shared LLM instance from nodes.py.
        config:         QueryTransformConfig (optional; uses defaults if None).
        retry_count:    Which retry this is — injected into the rewrite prompt.
        failure_reason: Why the last retrieval attempt failed.
        decompose:      Whether to run sub-query decomposition.
        stepback:       Whether to run step-back broadening.

    Returns:
        {
            "rewritten":   str            — primary output, always populated
            "sub_queries": List[str]      — populated if decompose=True
            "stepback":    str | None     — populated if stepback=True
        }
    """
    config  = config or QueryTransformConfig()
    results: Dict[str, Any] = {
        "rewritten":   question,   # safe default: no change
        "sub_queries": [],
        "stepback":    None,
    }

    # 1. Rewrite (always runs)
    try:
        rewrite_chain = get_query_rewriter_chain(llm, config)
        rewrite_out: RewrittenQuery = rewrite_chain.invoke({
            "question":       question,
            "retry_count":    retry_count,
            "failure_reason": failure_reason,
        })
        results["rewritten"] = rewrite_out.rewritten_query
        logger.info(
            "QueryTransform: rewritten %r → %r",
            question[:60], results["rewritten"][:60],
        )
    except Exception as exc:
        logger.warning("QueryTransform: rewrite failed (%s); keeping original.", exc)

    # 2. Decomposition (optional)
    if decompose:
        try:
            decomp_chain = get_decomposition_chain(llm, config)
            decomp_out: DecomposedQueries = decomp_chain.invoke({
                "question":      question,
                "max_subqueries": config.max_subqueries,
            })
            results["sub_queries"] = decomp_out.sub_queries
            logger.info(
                "QueryTransform: decomposed into %d sub-queries: %s",
                len(decomp_out.sub_queries), decomp_out.sub_queries,
            )
        except Exception as exc:
            logger.warning("QueryTransform: decomposition failed (%s).", exc)

    # 3. Step-back (optional)
    if stepback:
        try:
            sb_chain = get_stepback_chain(llm, config)
            sb_out: StepBackQuery = sb_chain.invoke({"question": question})
            results["stepback"] = sb_out.stepback_query
            logger.info(
                "QueryTransform: step-back query: %r", results["stepback"][:80]
            )
        except Exception as exc:
            logger.warning("QueryTransform: step-back failed (%s).", exc)

    return results


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def build_query_transform_from_config(
    config_dict: Dict[str, Any],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Build all three chains from the `query.transform` config block.

    Returns a dict of ready-to-use chains so nodes.py can store them
    as globals and call them without re-instantiation per request.

    Example:
        transform_chains = build_query_transform_from_config(
            yaml_config["query"]["transform"], llm=shared_llm
        )
        rewritten = transform_chains["rewrite"].invoke({...}).rewritten_query
    """
    cfg = QueryTransformConfig(
        max_subqueries=int(config_dict.get("max_subqueries", 4)),
        rewrite_model=config_dict.get("rewrite_model", "gpt-4o-mini"),
        rewrite_temperature=float(config_dict.get("rewrite_temperature", 0.0)),
        decompose_model=config_dict.get("decompose_model", "gpt-4o-mini"),
        decompose_temperature=float(config_dict.get("decompose_temperature", 0.3)),
        stepback_model=config_dict.get("stepback_model", "gpt-4o-mini"),
        stepback_temperature=float(config_dict.get("stepback_temperature", 0.2)),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    return {
        "rewrite":    get_query_rewriter_chain(llm, cfg),
        "stepback":   get_stepback_chain(llm, cfg),
        "decompose":  get_decomposition_chain(llm, cfg),
        "config":     cfg,
    }


# ---------------------------------------------------------------------------
# Public aliases — match the import names used in nodes.py
# ---------------------------------------------------------------------------

# Raw prompt templates (for direct inspection / YAML export)
query_rewriter_prompt       = QUERY_REWRITE_PROMPT
step_back_prompt            = STEP_BACK_PROMPT
sub_query_decomposer_prompt = DECOMPOSITION_PROMPT


# ---------------------------------------------------------------------------
# Standalone smoke-test  (python -m prompts.query_transform)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    from langchain_openai import ChatOpenAI as _ChatOpenAI

    _llm = _ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    _q   = "What were the total revenue and top-selling product of Company X in Q1 2025, and how did that compare to Q1 2024?"

    print("\n=== query_transform smoke-test ===")
    out = run_all_transforms(
        question=_q,
        llm=_llm,
        retry_count=1,
        failure_reason="No documents passed relevance grading.",
        decompose=True,
        stepback=True,
    )
    print(f"\nOriginal : {_q}")
    print(f"Rewritten: {out['rewritten']}")
    print(f"Step-back: {out['stepback']}")
    print(f"Sub-queries ({len(out['sub_queries'])}):")
    for i, sq in enumerate(out["sub_queries"], 1):
        print(f"  {i}. {sq}")