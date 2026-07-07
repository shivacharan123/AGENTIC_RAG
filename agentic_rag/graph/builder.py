"""
graph/builder.py — LangGraph workflow definition for Agentic RAG

Graph structure:
    query_analyzer
        → router_node
        → [vector_search, web_search_tool, sql_graph_tool]  ← parallel fan-out
        → context_aggregator
        → relevance_grader
        → (pass)  → answer_generator → hallucination_checker → (grounded) → END
        → (retry) → query_rewriter   → router_node           ← self-corrective loop

Parallel fan-out:
    decide_route() in edges.py returns a List[str] of node names.
    LangGraph >= 0.2 natively dispatches to all named nodes in parallel
    when the condition function returns a list.
    retrieved_documents in GraphState uses Annotated[List, operator.add]
    so all parallel outputs are safely merged before context_aggregator runs.

Persistence:
    build_workflow() returns the *uncompiled* StateGraph so callers can
    compile it with whatever checkpointer they want (or none, e.g. in tests).
    build_app() is the convenience entry point that attaches the
    memory/checkpointer.py-backed checkpointer and is what the rest of the
    app (API layer, CLI, etc.) should actually import and call.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.graph.state import GraphState
from agentic_rag.graph.nodes import (
    query_analyzer,
    router_node,
    vector_search,
    web_search_tool,
    sql_graph_tool,
    context_aggregator,
    relevance_grader,
    query_rewriter,
    answer_generator,
    hallucination_checker,
)
from agentic_rag.graph.edges import decide_route, grade_relevance, check_hallucination


def build_workflow() -> StateGraph:
    """
    Assemble the graph structure (nodes + edges) only — does NOT compile.

    Kept separate from build_app() so tests / notebooks can compile this
    with checkpointer=None or a throwaway MemorySaver without going through
    config.yaml at all.
    """
    workflow = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    workflow.add_node("query_analyzer",       query_analyzer)
    workflow.add_node("router_node",          router_node)
    workflow.add_node("vector_search",        vector_search)
    workflow.add_node("web_search_tool",      web_search_tool)
    workflow.add_node("sql_graph_tool",       sql_graph_tool)
    workflow.add_node("context_aggregator",   context_aggregator)
    workflow.add_node("relevance_grader",     relevance_grader)
    workflow.add_node("query_rewriter",       query_rewriter)
    workflow.add_node("answer_generator",     answer_generator)
    workflow.add_node("hallucination_checker", hallucination_checker)

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "router_node")

    # ── Parallel fan-out ──────────────────────────────────────────────────
    # decide_route() returns either:
    #   • A List[str]  → LangGraph fires all listed nodes in parallel
    #   • A str        → LangGraph routes to that single node
    # path_map maps every possible string output to the matching node.
    workflow.add_conditional_edges(
        "router_node",
        decide_route,
        {
            "vector_search":   "vector_search",
            "web_search_tool": "web_search_tool",
            "sql_graph_tool":  "sql_graph_tool",
            "conversational":  "answer_generator",   # ← ADD THIS
        }
    )

    # ── Fan-in: all three retrievers converge on context_aggregator ───────
    # The merge_documents reducer on retrieved_documents (state.py) merges
    # the parallel lists automatically.
    workflow.add_edge("vector_search",   "context_aggregator")
    workflow.add_edge("web_search_tool", "context_aggregator")
    workflow.add_edge("sql_graph_tool",  "context_aggregator")

    # ── Post-retrieval pipeline ─────────────────────────────────────────
    workflow.add_edge("context_aggregator", "relevance_grader")

    # Self-corrective retrieval loop (capped at 3 retries in edges.py)
    workflow.add_conditional_edges(
        "relevance_grader",
        grade_relevance,
        {
            "retry": "query_rewriter",
            "pass":  "answer_generator",
        }
    )

    # Rewriter sends query back to router for a fresh parallel retrieval round
    workflow.add_edge("query_rewriter", "router_node")

    # ── Generation + grounding loop ────────────────────────────────────
    workflow.add_edge("answer_generator", "hallucination_checker")

    # Hallucination loop (capped at 2 regenerations in edges.py)
    workflow.add_conditional_edges(
        "hallucination_checker",
        check_hallucination,
        {
            "regenerate": "answer_generator",
            "grounded":   END,
        }
    )

    return workflow


def compile_workflow(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """
    Compile build_workflow()'s graph with an explicit checkpointer (or
    none). Useful for tests / notebooks that want to inject a throwaway
    MemorySaver without touching config.yaml.
    """
    return build_workflow().compile(checkpointer=checkpointer)


def build_app() -> CompiledStateGraph:
    """
    Production entry point: compiles the workflow with the checkpointer
    configured via config.yaml's `memory.checkpointer` block, so every
    invocation scoped to a thread_id persists GraphState (including
    chat_history) across turns.

        from agentic_rag.graph.builder import app
        from agentic_rag.memory.checkpointer import make_thread_config

        result = app.invoke(
            {"user_query": "What were Q3 sales?", "chat_history": []},
            config=make_thread_config("session-123"),
        )
    """
    # Local imports: keeps build_workflow()/compile_workflow() usable in
    # tests without requiring config.yaml or memory/checkpointer.py deps
    # to be present.
    from agentic_rag.prompts.config import load_config
    from agentic_rag.memory.checkpointer import build_checkpointer

    yaml_config: dict[str, Any] = load_config()
    checkpointer_config = yaml_config.get("memory", {}).get("checkpointer", {})
    checkpointer = build_checkpointer(checkpointer_config)

    return compile_workflow(checkpointer=checkpointer)


# Built once at import time, reused across requests — same "build once,
# module-level singleton" pattern used for llm / vector_store_client /
# reranker_client etc. in nodes.py.
app = build_app()