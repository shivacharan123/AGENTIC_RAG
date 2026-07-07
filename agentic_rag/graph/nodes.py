from typing import Any
import os

from dotenv import load_dotenv

load_dotenv()
from pydantic import SecretStr
from agentic_rag.graph.state import GraphState
from langchain_core.documents import Document
from agentic_rag.prompts.config import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
from agentic_rag.prompts.generator import (
    format_docs,
    get_fallback_chain,
    get_generator_chain,
)
from agentic_rag.prompts.grader import (
    GradeDocuments,
    GradeHallucinations,
    hallucination_checker_prompt,
    relevance_grader_prompt,
)
from agentic_rag.prompts.query_transform import get_query_rewriter_chain
import traceback
from agentic_rag.retrieval.hyde import build_hyde_from_config
from agentic_rag.retrieval.reranker import build_reranker_from_config
from agentic_rag.retrieval.sql_retriever import build_sql_retriever_from_config
from agentic_rag.retrieval.vector_store import build_vector_store_from_config
from agentic_rag.retrieval.web_search import build_web_search_from_config
from agentic_rag.agent.generator import GeneratorAgent, get_generator_agent
from agentic_rag.agent.hallucination_checker import (
    HallucinationCheckerAgent,
    GradeHallucinations as AgentGradeHallucinations,
    GRADE_GROUNDED,
    GRADE_REGENERATE,
    SEVERITY_MINOR,
)
from agentic_rag.agent.query_analyzer import QueryAnalyzer
from agentic_rag.agent.relevance_grader import RelevanceGrader

yaml_config = load_config()

from langchain_groq import ChatGroq

_model_cfg = yaml_config.get("models", {}).get("llm", {})

api_key = os.environ.get("GROQ_API_KEY")
llm = None
if api_key:
    llm = ChatGroq(
        model=_model_cfg.get("model_name", "llama-3.3-70b-versatile"),
        temperature=float(_model_cfg.get("temperature", 0.0)),
        max_tokens=int(_model_cfg.get("max_completion_tokens", 2048)),
        api_key=api_key,
    )

rt_config = yaml_config.get("retrieval", {})
vs_config_dict = rt_config.get("vector_store", {})
hyde_config_dict = rt_config.get("hyde", {})
ws_config_dict = rt_config.get("web_search", {})
sql_config_dict = yaml_config.get("sql", {})
rerank_config_dict = rt_config.get("reranker", {})

try:
    vector_store_client = build_vector_store_from_config(vs_config_dict)
    print("[startup] Vector store initialized successfully.")
except Exception as e:
    print(f"[startup] Vector store init FAILED: {e}")
    traceback.print_exc()
    vector_store_client = None

try:
    hyde_client = build_hyde_from_config(hyde_config_dict)
    print("[startup] HyDE client initialized successfully.")
except Exception as e:
    print(f"[startup] HyDE client init FAILED: {e}")
    traceback.print_exc()
    hyde_client = None

try:
    web_search_client = build_web_search_from_config(ws_config_dict)
    print("[startup] Web search client initialized successfully.")
except Exception as e:
    print(f"[startup] Web search client init FAILED: {e}")
    traceback.print_exc()
    web_search_client = None

try:
    sql_client = build_sql_retriever_from_config(sql_config_dict)
    print("[startup] SQL client initialized successfully.")
except Exception as e:
    print(f"[startup] SQL client init FAILED: {e}")
    traceback.print_exc()
    sql_client = None


try:
    reranker_client = build_reranker_from_config(rerank_config_dict)
    print("[startup] Reranker client initialized successfully.")
except Exception as e:
    print(f"[startup] Reranker client init FAILED: {e}")
    traceback.print_exc()
    reranker_client = None

_generator_chain = None
_fallback_chain = None
_generator_agent = None
_relevance_chain = None
_hallucination_chain = None
_rewrite_chain = None
_query_analyzer_agent = None

_hallucination_checker_agent = None
_relevance_agent = None

if llm is not None:
    _generator_agent = get_generator_agent(llm)
    _relevance_chain = relevance_grader_prompt | llm.with_structured_output(GradeDocuments)
    _hallucination_chain = hallucination_checker_prompt | llm.with_structured_output(GradeHallucinations)
    _rewrite_chain = get_query_rewriter_chain(llm)
    _query_analyzer_agent = QueryAnalyzer(llm)
    _hallucination_checker_agent = HallucinationCheckerAgent(llm)
    _relevance_agent = RelevanceGrader(llm)


def query_analyzer(state: GraphState) -> dict[str, Any]:
    print("---QUERY ANALYZER---")
    user_query = state.get("user_query", "")
    current_query = user_query
    route_decision = "parallel_all"

    if _query_analyzer_agent is not None:
        try:
            analysis = _query_analyzer_agent.analyze(user_query)
        except Exception as e:
            print(f"[query_analyzer] analyze() raised: {e}")
            analysis = None

        if analysis:
            rewritten = analysis.get("rewritten_query", user_query)

            # ── FIX: reject rewrites that are absurdly longer than the original ──
            if len(rewritten.split()) > max(len(user_query.split()) * 5, 30):
                print(f"[query_analyzer] Rewrite rejected (too long), keeping original.")
                rewritten = user_query

            current_query = rewritten
            route_decision = analysis.get("datasource", "parallel_all")
            print(f"[query_analyzer] Route: {route_decision}")
        else:
            print("[query_analyzer] analyze() returned None — using defaults (parallel_all).")

    return {
        "current_query": current_query,
        "route_decision": route_decision,
        "query_classification": "informational",
        "sub_queries": [current_query],
        "retry_count": 0,
        "generation_retry_count": 0,
        "retrieved_documents": [],
        "aggregated_context": [],
    }


def router_node(state: GraphState) -> dict[str, Any]:
    print("---ROUTER NODE---")
    raw = state.get("route_decision", "parallel_all")
    
    # ── FIX: detect conversational/greeting queries ──
    # If the original query is very short and has no retrieval-worthy content,
    # route it directly. The query_analyzer should ideally return "conversational"
    # but as a safety net we catch it here too.
    user_query = state.get("user_query", "").strip().lower()
    CONVERSATIONAL_SIGNALS = {"hey", "hi", "hello", "thanks", "thank you", "bye", 
                               "ok", "okay", "cool", "great", "sure", "yes", "no"}
    words = set(user_query.split())
    if words and words.issubset(CONVERSATIONAL_SIGNALS):
        print(f"[router_node] Conversational input detected → skipping retrieval")
        return {"route_decision": "conversational"}   # edges.py must handle this
    
    _alias = {
        "vectorstore":    "vector_search",
        "vector_store":   "vector_search",
        "websearch":      "web_only",
        "web_search":     "web_only",
        "sql":            "sql_only",
        "vector_only":    "vector_search",
        "vector_search":  "vector_search",
        "web_only":       "web_only",
        "sql_only":       "sql_only",
        "parallel_all":   "parallel_all",
        "conversational": "conversational",   # pass-through
    }
    normalized = _alias.get(raw, "parallel_all")
    print(f"[router_node] Route: {raw!r} → normalized: {normalized!r}")
    return {"route_decision": normalized}


def vector_search(state: GraphState) -> dict[str, Any]:
    print("---VECTOR SEARCH---")
    if vector_store_client is None:
        print("Vector store unavailable. Skipping.")
        return {"retrieved_documents": []}

    query = state.get("current_query") or state.get("user_query", "")
    use_hyde = hyde_config_dict.get("enabled", True)
    docs = vector_store_client.retrieve(
        query=query,
        k=rt_config.get("k", 8),
        use_hyde=use_hyde,
        hyde_client=hyde_client,
    )
    return {"retrieved_documents": docs if docs is not None else []}


def web_search_tool(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    query = state.get("current_query") or state.get("user_query", "")
    if web_search_client is None:
        print("WebSearch client unavailable. Skipping web retrieval.")
        return {"retrieved_documents": []}
    docs = web_search_client.web_search(query=query, num_results=ws_config_dict.get("k", 3))
    return {"retrieved_documents": docs if docs is not None else []}


def sql_graph_tool(state: GraphState) -> dict[str, Any]:
    print("---SQL TOOL---")
    query = state.get("current_query") or state.get("user_query", "")
    if sql_client is None:
        return {"retrieved_documents": []}
    docs = sql_client.get_structured_data(query)
    return {"retrieved_documents": docs if docs is not None else []}


def context_aggregator(state: GraphState) -> dict[str, Any]:
    print("---CONTEXT AGGREGATOR & RERANKER---")
    raw_docs = state.get("retrieved_documents", [])
    query = state.get("current_query") or state.get("user_query", "")

    if not raw_docs:
        return {"aggregated_context": []}

    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in raw_docs:
        key = (doc.page_content or "").strip()
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    if reranker_client is None:
        print("Reranker unavailable. Returning deduplicated docs as-is.")
        return {"aggregated_context": unique_docs}

    refined_docs = reranker_client.rerank(query=query, documents=unique_docs)
    return {"aggregated_context": refined_docs}


def relevance_grader(state: GraphState) -> dict[str, Any]:
    docs = state.get("aggregated_context", [])
    query = state.get("current_query") or state.get("user_query", "")

    if not docs:
        return {"aggregated_context": [], "relevance_status": "retry"}

    if _relevance_agent is None:
        return {"aggregated_context": docs, "relevance_status": "pass"}

    filtered = _relevance_agent.grade(docs, query)
    status = "pass" if filtered else "retry"
    return {"aggregated_context": filtered, "relevance_status": status}


def query_rewriter(state: GraphState) -> dict[str, Any]:
    print("---QUERY REWRITER---")
    query = state.get("current_query") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)

    rewritten = query
    if _rewrite_chain is not None:
        try:
            result = _rewrite_chain.invoke({
                "question": query,
                "retry_count": retry_count,
                "failure_reason": "No documents passed relevance grading.",
            })
            candidate = getattr(result, "rewritten_query", query)
            
            # ── FIX: reject runaway rewrites ──
            if len(candidate.split()) > max(len(query.split()) * 5, 30):
                print(f"[query_rewriter] Rewrite rejected (too long), keeping original.")
            else:
                rewritten = candidate
                print(f"[query_rewriter] {query!r} → {rewritten!r}")
                
        except Exception as e:
            print(f"[query_rewriter] Rewrite failed ({e}), keeping original.")

    return {
        "current_query": rewritten,
        "retry_count": retry_count + 1,
        "retrieved_documents": [],
        "aggregated_context": [],
    }


def answer_generator(state: GraphState) -> dict[str, Any]:
    if _generator_agent is None:
        return {"generation": "", "generation_retry_count": state.get("generation_retry_count", 0)}

    hallucination_status = state.get("hallucination_status")
    chat_history = state.get("chat_history", [])

    if hallucination_status == "regenerate":
        mode = "regenerate"
    elif chat_history:
        mode = "conversational"
    else:
        mode = "rag"

    agent_state = {
        "user_query": state.get("current_query") or state.get("user_query", ""),
        "relevant_documents": state.get("aggregated_context", []),
        "generation_mode": mode,
        "generated_answer": state.get("generation", ""),
        "hallucination_grade": state.get("hallucination_grade"),
        "chat_history": chat_history,
    }
    
    result = _generator_agent.generate(agent_state)
    
    # Bug Fix: Ensure answer falls back to an empty string if result.answer is None (e.g. on 429 errors)
    answer = result.answer or ""
    
    return {
        "generation": answer,
        "generation_metadata": result.model_dump(),
        "generation_retry_count": state.get("generation_retry_count", 0),
    }


def hallucination_checker(state: GraphState) -> dict[str, Any]:
    """Check generation against retrieved context."""
    print("---HALLUCINATION CHECKER---")

    # Bug Fix: Added 'or ""' guard to ensure generation is never None when .strip() is called
    generation = state.get("generation", "") or ""
    documents = state.get("aggregated_context", []) or state.get("retrieved_documents", [])
    gen_retries = state.get("generation_retry_count", 0)
    query = state.get("current_query") or state.get("user_query", "")

    if _hallucination_checker_agent is None:
        return {"hallucination_status": "regenerate", "generation_retry_count": gen_retries}

    if not generation.strip():
        return {"hallucination_status": "regenerate", "generation_retry_count": gen_retries}

    if not documents:
        return {"hallucination_status": "grounded", "generation_retry_count": gen_retries}

    status = GRADE_GROUNDED
    try:
        grade = _hallucination_checker_agent.check(
            question=query,
            documents=documents,
            generation=generation,
            # gen_retries = number of regenerations already completed.
            # The agent guard fires when this reaches max_regeneration_attempts.
            regeneration_attempt=gen_retries,
        )

        # agent/hallucination_checker.py uses binary_score: "grounded" | "regenerate"
        # — use GRADE_* constants directly; no translation needed.
        status = grade.binary_score
        if status == GRADE_REGENERATE:
            print(
                f"[hallucination_checker] Unsupported claims: {grade.unsupported_claims}"
            )
    except Exception as e:
        print(f"[hallucination_checker] Check failed ({e}) — treating as grounded.")
        status = GRADE_GROUNDED
        # Bug 3 fix: always write a grade object back to state so that a stale
        # "regenerate" grade from a prior loop iteration cannot bleed through and
        # trick answer_generator into entering regeneration mode on the next cycle.
        grade = AgentGradeHallucinations(
            binary_score=GRADE_GROUNDED,
            severity=SEVERITY_MINOR,
            grounding_score=0.5,
            reasoning=f"[nodes.py exception fallback] {e}",
        )

    new_retries = gen_retries + 1 if status == GRADE_REGENERATE else gen_retries

    result: dict[str, Any] = {
        "hallucination_status": status,
        "generation_retry_count": new_retries,
        # grade is always set here: either from .check() or from the exception-path
        # sentinel above.  Writing it unconditionally ensures no stale grade lingers
        # in state across loop iterations (Bug 3 fix).
        "hallucination_grade": grade,
    }


    return result