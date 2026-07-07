"""
graph/edges.py — Conditional edge routing functions for Agentic RAG

These functions are passed to workflow.add_conditional_edges().
Each receives the current GraphState and returns a string (or list of strings
for fan-out) that maps to a node name via the path_map dict in builder.py.

Fan-out (parallel_all):
    decide_route() returns a List[str] of node names.
    LangGraph >= 0.2 supports this natively: when the condition function
    returns a list, the graph dispatches to ALL named nodes in parallel.
    The `retrieved_documents` field in GraphState uses
    Annotated[List[Document], merge_documents] so each parallel branch's
    output is safely merged before context_aggregator runs.
"""

from typing import List, Union
from agentic_rag.graph.state import GraphState


def decide_route(state: GraphState) -> Union[str, List[str]]:
    """
    Determine which retrieval node(s) to activate.

    Returns a list of node name strings for parallel execution,
    or a single string for targeted routing.

    LangGraph will fire all nodes in the returned list in parallel
    when a list is returned. Their outputs are merged via the
    Annotated[List[Document], merge_documents] reducer on retrieved_documents.
    """
    decision = state.get("route_decision", "parallel_all")
    classification = state.get("query_classification", "").lower()
    
    if "greeting" in classification or "conversational" in classification:
        return "parallel_all"

    if decision == "parallel_all":
        # Fire all three retrievers simultaneously
        return ["vector_search", "web_search_tool", "sql_graph_tool"]

    # Targeted single-retriever routing
    targeted = {
        "vector_only": "vector_search",
        "web_only":    "web_search_tool",
        "sql_only":    "sql_graph_tool",
        # Legacy key from nodes.py earlier versions
        "vector_search": "vector_search",
    }
    return targeted.get(decision, "vector_search")


def grade_relevance(state: GraphState) -> str:
    """
    Route after relevance grading.

    Returns:
        'retry'  → query_rewriter node (re-runs retrieval with improved query)
        'pass'   → answer_generator node

    The retry loop is capped at 3 attempts by checking retry_count.
    If retries are exhausted, force 'pass' so the graph always terminates.
    """
    status      = state.get("relevance_status", "pass")
    retry_count = state.get("retry_count", 0)

    if status == "retry" and retry_count < 3:
        print(f"[grade_relevance] Retry {retry_count + 1}/3 — rewriting query.")
        return "retry"

    if status == "retry" and retry_count >= 3:
        print("[grade_relevance] Max retries reached — proceeding to generation.")

    return "pass"


def check_hallucination(state: GraphState) -> str:
    """
    Route after hallucination checking.
    """
    status = state.get("hallucination_status", "grounded")
    retry_count = state.get("generation_retry_count", 0)

    if status == "regenerate" and retry_count < 2:
        print(f"[check_hallucination] Regeneration attempt {retry_count + 1}/2.")
        return "regenerate"

    if status == "regenerate" and retry_count >= 2:
        print("[check_hallucination] Max regeneration attempts reached — returning answer.")

    return "grounded"