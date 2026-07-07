"""
api/main.py — FastAPI entry point for the Agentic RAG system

Fixes applied vs previous version:
    FIX 1  — Removed the dead `_graph_app = build_workflow()` before the try block.
    FIX 2  — Removed the stale unused imports inside the try block (GraphState, StateGraph, END).
    FIX 3  — Conditional edge keys now match what decide_route() actually returns:
              "web_only" → "web_search_tool" node
              "sql_only" → "sql_graph_tool" node
              "parallel_all" added (fans out to vector_search as default).
    FIX 4  — __main__ block now loads config fresh instead of reading empty _yaml_config.
    FIX 5  — Embedder is lazy-imported inside /ingest (not at module level).
    FIX 6  — BackgroundTasks removed from /ingest (was imported but never used).
    FIX 7  — long_term built with llm=None warning documented; LLM passed if available.
    FIX 8  — memory_context key removed from initial state (not in GraphState TypedDict).
"""
#uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
#streamlit run app.py


from __future__ import annotations
import os
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.runnables.config import RunnableConfig

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_rag.prompts.config import load_config
from agentic_rag.graph.builder import build_workflow
from agentic_rag.memory.checkpointer import build_checkpointer_from_config, RagCheckpointer
from agentic_rag.memory.long_term import build_long_term_from_config, LongTermMemory

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("agentic_rag.api")

# ── App-level singletons ──────────────────────────────────────────────────────
_yaml_config: Dict[str, Any] = {}
_checkpointer: Optional[RagCheckpointer] = None
_long_term: Optional[LongTermMemory] = None
_graph_app = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _yaml_config, _checkpointer, _long_term, _graph_app

    logger.info("═══ Agentic RAG API — startup ═══")
    t0 = time.time()

    _yaml_config = load_config()
    logger.info("Config loaded.")

    cp_config = _yaml_config.get("memory", {}).get("checkpointer", {})
    _checkpointer = build_checkpointer_from_config(cp_config)
    logger.info("Checkpointer ready.")

    lt_config = _yaml_config.get("memory", {}).get("long_term", {})
    # FIX 7: pass llm=None; LongTermMemory must handle None gracefully for
    # summarisation — if your LongTermMemory needs an LLM, build it first and
    # pass it here. Logged so the operator knows compression won't use an LLM.
    _long_term = build_long_term_from_config(lt_config, llm=None)
    if _long_term is None:
        logger.warning("Long-term memory not initialised — cross-session summarisation disabled.")
    else:
        logger.info("Long-term memory ready.")

    # FIX 1: build graph only once, inside the try/except
    try:
        _graph_app = _build_workflow_with_checkpointer(_checkpointer)
        logger.info("Graph compiled with checkpointer.")
    except Exception as exc:
        logger.warning(
            "Could not compile graph with checkpointer (%s). "
            "Running without persistence — conversations won't resume across restarts.", exc,
        )
        _graph_app = build_workflow()

    elapsed = time.time() - t0
    logger.info("═══ Startup complete in %.2fs ═══", elapsed)

    yield

    logger.info("═══ Agentic RAG API — shutdown ═══")


def _build_workflow_with_checkpointer(cp: RagCheckpointer):
    """
    Compile the LangGraph workflow with the checkpointer injected.

    FIX 3: Conditional edge maps now use the exact strings that
    decide_route() returns (matching router_node's _alias values):
        "vector_search" → node "vector_search"
        "web_only"      → node "web_search_tool"
        "sql_only"      → node "sql_graph_tool"
        "parallel_all"  → node "vector_search" (default fan-out)

    If you need true parallel fan-out, replace "parallel_all" with
    langgraph.types.Send() calls in decide_route() instead.
    """
    from langgraph.graph import StateGraph, END
    from agentic_rag.graph.state import GraphState
    from agentic_rag.graph.nodes import (
        query_analyzer, router_node,
        vector_search, web_search_tool, sql_graph_tool,
        context_aggregator, relevance_grader, query_rewriter,
        answer_generator, hallucination_checker,
    )
    from agentic_rag.graph.edges import decide_route, grade_relevance, check_hallucination

    workflow = StateGraph(GraphState)

    workflow.add_node("query_analyzer",        query_analyzer)
    workflow.add_node("router_node",           router_node)
    workflow.add_node("vector_search",         vector_search)
    workflow.add_node("web_search_tool",       web_search_tool)
    workflow.add_node("sql_graph_tool",        sql_graph_tool)
    workflow.add_node("context_aggregator",    context_aggregator)
    workflow.add_node("relevance_grader",      relevance_grader)
    workflow.add_node("query_rewriter",        query_rewriter)
    workflow.add_node("answer_generator",      answer_generator)
    workflow.add_node("hallucination_checker", hallucination_checker)

    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "router_node")

    # FIX 3: keys must match decide_route() return strings exactly
    workflow.add_conditional_edges(
        "router_node", decide_route,
        {
            "vector_search":   "vector_search",
            "web_search_tool": "web_search_tool",   # ← key is "web_search_tool"
            "sql_graph_tool":  "sql_graph_tool",
            "parallel_all":   "vector_search",
            "conversational": "answer_generator",   # ← ADD THIS — skips retrieval entirely
        },
    )

    workflow.add_edge("vector_search",      "context_aggregator")
    workflow.add_edge("web_search_tool",    "context_aggregator")
    workflow.add_edge("sql_graph_tool",     "context_aggregator")
    workflow.add_edge("context_aggregator", "relevance_grader")

    workflow.add_conditional_edges(
        "relevance_grader", grade_relevance,
        {"retry": "query_rewriter", "pass": "answer_generator"},
    )
    workflow.add_edge("query_rewriter",  "router_node")
    workflow.add_edge("answer_generator", "hallucination_checker")

    workflow.add_conditional_edges(
        "hallucination_checker", check_hallucination,
        {"regenerate": "answer_generator", "grounded": END},
    )

    return workflow.compile(checkpointer=cp.checkpointer)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG API",
    description=(
        "Self-corrective Retrieval-Augmented Generation system. "
        "Combines vector search, web search, and SQL retrieval "
        "with LLM grading and hallucination checking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend SPA.
# - /static -> static/index.html
# - /       -> static/index.html (so http://0.0.0.0:8000/ works)
try:
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
except Exception:
    logger.warning("No 'static' directory found — frontend will not be served.")


# ── Request / Response schemas ────────────────────────────────────────────────
class InvokeRequest(BaseModel):
    question: str = Field(..., description="The user's natural language question.")
    thread_id: Optional[str] = Field(default=None, description="Session ID for continuity.")


class InvokeResponse(BaseModel):
    answer: str
    thread_id: str
    sources: List[str] = Field(default_factory=list)
    retry_count: int   = Field(default=0)
    latency_ms: float


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None)
    index_name: str          = Field(default="agentic_rag")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    index_name: str
    latency_ms: float


class HistoryResponse(BaseModel):
    thread_id: str
    chat_history: List[Dict[str, str]]
    turn_count: int
    summary: Optional[str] = None


class ThreadInfo(BaseModel):
    thread_id: str
    created_at: Optional[str]
    updated_at: Optional[str]
    turn_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_initial_state(question: str, thread_id: str) -> Dict[str, Any]:
    """
    Build the initial GraphState for a new graph invocation.

    FIX 8: memory_context removed — it is not a declared GraphState key.
    If you need it, add it to graph/state.py first.
    """
    chat_history: List[Dict[str, str]] = []
    if _checkpointer:
        chat_history = _checkpointer.get_chat_history(thread_id)

    return {
        "user_query":             question,
        "current_query":          question,
        "sub_queries":            [],
        "query_classification":   "",
        "route_decision":         "",
        "retrieved_documents":    [],
        "aggregated_context":     [],
        "relevance_status":       "",
        "hallucination_status":   "",
        "retry_count":            0,
        "generation_retry_count": 0,
        "generation":             "",
        "citations":              [],
        "chat_history":           chat_history,
    }


def _extract_sources(final_state: Dict[str, Any]) -> List[str]:
    docs = final_state.get("aggregated_context", [])
    return sorted({doc.metadata.get("source", "unknown") for doc in docs})


def _update_memory_after_turn(
    thread_id: str,
    question: str,
    answer: str,
    current_history: List[Dict[str, str]],
    turn_count: int,
) -> List[Dict[str, str]]:
    updated = current_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    if _long_term is None:
        return updated
    try:
        _long_term.auto_extract_and_store_facts(
            thread_id=thread_id, user_message=question,
            assistant_answer=answer, source_turn=turn_count,
        )
    except Exception as exc:
        logger.warning("fact extraction failed: %s", exc)
    try:
        return _long_term.maybe_compress(
            thread_id=thread_id, chat_history=updated, turn_index=turn_count,
        )
    except Exception as exc:
        logger.warning("history compression failed: %s", exc)
        return updated


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    return {
        "status":       "healthy",
        "graph":        _graph_app is not None,
        "checkpointer": _checkpointer is not None,
        "long_term":    _long_term is not None,
    }


@app.post("/invoke", response_model=InvokeResponse, tags=["RAG"])
async def invoke(request: InvokeRequest) -> InvokeResponse:
    if _graph_app is None:
        raise HTTPException(status_code=503, detail="Graph not initialised.")

    thread_id = request.thread_id or str(uuid.uuid4())
    question  = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info("POST /invoke | thread=%s | q=%r", thread_id, question[:80])
    t0 = time.time()

    initial_state = _build_initial_state(question, thread_id)
    try:
        final_state = _graph_app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.error("Graph error thread=%s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail=f"Graph error: {exc}")

    answer      = final_state.get("generation", "No answer generated.")
    retry_count = final_state.get("retry_count", 0)
    sources     = _extract_sources(final_state)

    prior_history = final_state.get("chat_history", initial_state["chat_history"])
    _update_memory_after_turn(thread_id, question, answer, prior_history, len(prior_history) // 2)

    latency_ms = (time.time() - t0) * 1000
    logger.info("POST /invoke done | thread=%s | latency=%.0fms", thread_id, latency_ms)

    return InvokeResponse(
        answer=answer, thread_id=thread_id,
        sources=sources, retry_count=retry_count,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/stream", tags=["RAG"])
async def stream(request: InvokeRequest) -> StreamingResponse:
    if _graph_app is None:
        raise HTTPException(status_code=503, detail="Graph not initialised.")

    thread_id = request.thread_id or str(uuid.uuid4())
    question  = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info("POST /stream | thread=%s | q=%r", thread_id, question[:80])
    initial_state = _build_initial_state(question, thread_id)
    graph_config  = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        import json
        final_generation = ""
        final_state_ref: Dict[str, Any] = {}
        try:
            for event in _graph_app.stream(initial_state, config=graph_config):
                for node_name, node_output in event.items():
                    payload: Dict[str, Any] = {"node": node_name, "status": "done"}
                    if node_name == "context_aggregator":
                        payload["doc_count"] = len(node_output.get("aggregated_context", []))
                    elif node_name == "relevance_grader":
                        payload["relevance_status"] = node_output.get("relevance_status", "")
                    elif node_name == "answer_generator":
                        gen = node_output.get("generation", "")
                        payload["generation"] = gen
                        final_generation = gen
                        final_state_ref  = node_output
                    elif node_name == "hallucination_checker":
                        payload["hallucination_status"] = node_output.get("hallucination_status", "")
                    elif node_name in ("vector_search", "web_search_tool", "sql_graph_tool"):
                        payload["doc_count"] = len(node_output.get("retrieved_documents", []))
                    elif node_name == "query_rewriter":
                        payload["rewritten_query"] = node_output.get("current_query", "")
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Stream error thread=%s: %s", thread_id, exc, exc_info=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            if final_generation and _long_term:
                try:
                    prior = final_state_ref.get("chat_history", [])
                    _update_memory_after_turn(thread_id, question, final_generation, prior, len(prior) // 2)
                except Exception as exc:
                    logger.warning("Stream memory update failed: %s", exc)
            else:
                logger.warning("Stream ended without long_term — turn not saved to memory.")
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    FIX 5 + 6: Embedder is lazy-imported here; BackgroundTasks removed.
    """
    logger.info("POST /ingest | dir=%s | index=%s", request.directory, request.index_name)
    t0 = time.time()
    try:
        from agentic_rag.ingestion.embedder import Embedder  # lazy — loads HF model on demand
        embedder = Embedder()
        if request.directory:
            from agentic_rag.ingestion.loader import load_from_directory
            docs = load_from_directory(request.directory)
            vs   = embedder.embed_and_store(docs, index_name=request.index_name)
        else:
            vs = embedder.run_full_pipeline(index_name=request.index_name)
        chunks = vs.index.ntotal if hasattr(vs, "index") else -1
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    latency_ms = (time.time() - t0) * 1000
    return IngestResponse(
        status="success", chunks_indexed=chunks,
        index_name=request.index_name, latency_ms=round(latency_ms, 1),
    )


@app.get("/history/{thread_id}", response_model=HistoryResponse, tags=["Memory"])
async def get_history(thread_id: str) -> HistoryResponse:
    if _checkpointer is None:
        raise HTTPException(status_code=503, detail="Checkpointer not initialised.")
    chat_history = _checkpointer.get_chat_history(thread_id)
    summary = _long_term.get_thread_summary(thread_id) if _long_term else None
    return HistoryResponse(
        thread_id=thread_id, chat_history=chat_history,
        turn_count=len(chat_history) // 2, summary=summary,
    )


@app.delete("/session/{thread_id}", tags=["Memory"])
async def clear_session(thread_id: str) -> Dict[str, Any]:
    cp_deleted = _checkpointer.delete_thread(thread_id) if _checkpointer else False
    lt_deleted = _long_term.delete_thread_memory(thread_id) if _long_term else {}
    logger.info("DELETE /session/%s | cp=%s lt=%s", thread_id, cp_deleted, lt_deleted)
    return {
        "thread_id": thread_id, "checkpoints_cleared": cp_deleted,
        "long_term_cleared": lt_deleted, "status": "cleared",
    }


@app.get("/threads", response_model=List[ThreadInfo], tags=["Memory"])
async def list_threads() -> List[ThreadInfo]:
    if _checkpointer is None:
        raise HTTPException(status_code=503, detail="Checkpointer not initialised.")
    return [
        ThreadInfo(
            thread_id=t["thread_id"],
            created_at=t["created_at"].isoformat() if t.get("created_at") else None,
            updated_at=t["updated_at"].isoformat() if t.get("updated_at") else None,
            turn_count=t.get("turn_count", 0),
        )
        for t in _checkpointer.list_threads()
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # FIX 4: load config fresh here — _yaml_config is still {} at this point
    _cli_cfg = load_config()
    _api_cfg = _cli_cfg.get("api", {})

    uvicorn.run(
        "agentic_rag.api.main:app",
        host=_api_cfg.get("host", "0.0.0.0"),
        port=int(_api_cfg.get("port", 8000)),
        reload=bool(_api_cfg.get("debug", True)),
        log_level="info",
    )