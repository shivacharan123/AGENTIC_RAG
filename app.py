"""
app.py — Streamlit frontend for the Agentic RAG system
--------------------------------------------------------
Talks to the FastAPI backend (agentic_rag/api/main.py) via HTTP.

Run backend first:
    uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

Then run this frontend:
    streamlit run app.py

The frontend calls /stream for live node-by-node progress, with
/invoke as a fallback, and exposes /ingest, /history, and /threads
through the sidebar.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"   # change if backend is on another host/port
STREAM_TIMEOUT = 120                 # seconds to wait for the streaming response


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall dark feel */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #181c27; border-right: 1px solid #2a3045; }

/* Header */
.rag-header {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0 18px;
    border-bottom: 1px solid #2a3045;
    margin-bottom: 20px;
}
.rag-logo {
    background: linear-gradient(135deg, #4f8ef7, #7c5cfc);
    border-radius: 10px;
    width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.rag-title { font-size: 22px; font-weight: 700; color: #e8eaf0; }
.rag-sub   { font-size: 12px; color: #7b84a0; }

/* Health badge */
.badge {
    display: inline-block;
    padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}
.badge-ok   { background: #0d3326; color: #3ecf8e; border: 1px solid #3ecf8e; }
.badge-fail { background: #3b1212; color: #f76b6b; border: 1px solid #f76b6b; }
.badge-warn { background: #3b2c00; color: #f7b84b; border: 1px solid #f7b84b; }

/* Pipeline node badges */
.pipeline { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
.node-done {
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: #0d3326; color: #3ecf8e; border: 1px solid #3ecf8e;
    font-family: monospace;
}
.node-active {
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: #0d2050; color: #4f8ef7; border: 1px solid #4f8ef7;
    font-family: monospace;
}

/* Source pills */
.src-pill {
    display: inline-block;
    font-size: 10px; padding: 1px 8px; border-radius: 20px;
    background: #1e2333; color: #7b84a0; border: 1px solid #2a3045;
    font-family: monospace; margin: 2px;
}

/* Latency chip */
.latency {
    font-size: 11px; color: #7b84a0; font-family: monospace;
}

/* Message bubbles */
.user-bubble {
    background: rgba(79,142,247,.10);
    border: 1px solid rgba(79,142,247,.25);
    border-radius: 10px; padding: 12px 16px;
    margin: 4px 0;
}
.ai-bubble {
    background: #181c27;
    border: 1px solid #2a3045;
    border-radius: 10px; padding: 12px 16px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ───────────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "thread_id":   str(uuid.uuid4()),
        "messages":    [],     # list of {"role": "user"|"ai", "content": str, "meta": dict}
        "sessions":    {},     # thread_id → {"turns": int}
        "health":      None,
        "last_health": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────
def _get(path: str, **kwargs) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, payload: Dict) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=STREAM_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _delete(path: str) -> bool:
    try:
        requests.delete(f"{API_BASE}{path}", timeout=10).raise_for_status()
        return True
    except Exception:
        return False


def _health() -> Dict[str, Any]:
    now = time.time()
    if now - st.session_state.last_health > 15:
        data = _get("/health") or {}
        st.session_state.health     = data
        st.session_state.last_health = now
    return st.session_state.health or {}


def _stream_answer(question: str, thread_id: str):
    """
    Generator: yields (nodes_so_far: list, generation: str, done: bool).
    Falls back to /invoke if streaming fails.
    """
    nodes: List[str] = []
    generation = ""

    try:
        with requests.post(
            f"{API_BASE}/stream",
            json={"question": question, "thread_id": thread_id},
            stream=True,
            timeout=STREAM_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                buffer += chunk
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            yield nodes, generation, True
                            return
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if ev.get("error"):
                            generation = f"⚠ Backend error: {ev['error']}"
                            yield nodes, generation, True
                            return

                        node = ev.get("node")
                        if node and node not in nodes:
                            nodes.append(node)

                        if node == "answer_generator" and ev.get("generation"):
                            generation = ev["generation"]

                        yield nodes, generation, False

    except Exception as e:
        # Fallback to /invoke
        data = _post("/invoke", {"question": question, "thread_id": thread_id})
        if data and not data.get("error"):
            generation = data.get("answer", "No answer returned.")
        else:
            generation = f"⚠ Error: {data.get('error', 'Unknown error')}"
        yield nodes, generation, True


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚡ Agentic RAG")

        # ── Health ────────────────────────────────────────────────────────
        h = _health()
        if h:
            graph_ok = h.get("graph", False)
            cp_ok    = h.get("checkpointer", False)
            lt_ok    = h.get("long_term", False)

            overall = graph_ok and cp_ok
            badge   = "badge-ok" if overall else "badge-fail"
            label   = "connected" if overall else "partial / offline"
            st.markdown(
                f'<span class="badge {badge}">{label}</span>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Graph",        "✓" if graph_ok else "✗")
            c2.metric("Checkpointer", "✓" if cp_ok    else "✗")
            c3.metric("Long-term",    "✓" if lt_ok    else "✗")
        else:
            st.markdown('<span class="badge badge-fail">offline</span>', unsafe_allow_html=True)

        st.divider()

        # ── Session controls ──────────────────────────────────────────────
        st.markdown("### Session")
        st.code(st.session_state.thread_id[:24] + "…", language=None)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("＋ New", use_container_width=True):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages  = []
                st.rerun()
        with col2:
            if st.button("🗑 Clear", use_container_width=True):
                _delete(f"/session/{st.session_state.thread_id}")
                st.session_state.messages = []
                st.toast("Session cleared.", icon="🗑")
                st.rerun()

        if st.button("↺ Load history", use_container_width=True):
            data = _get(f"/history/{st.session_state.thread_id}")
            if data:
                st.session_state.messages = []
                hist = data.get("chat_history", [])
                for i in range(0, len(hist) - 1, 2):
                    u = hist[i]
                    a = hist[i + 1] if i + 1 < len(hist) else None
                    if u:
                        st.session_state.messages.append(
                            {"role": "user", "content": u["content"], "meta": {}}
                        )
                    if a:
                        st.session_state.messages.append(
                            {"role": "ai", "content": a["content"], "meta": {}}
                        )
                st.toast(f"Loaded {data.get('turn_count', 0)} turns.", icon="↺")
                st.rerun()
            else:
                st.warning("No history found for this session.")

        st.divider()

        # ── Ingest ────────────────────────────────────────────────────────
        st.markdown("### Ingest Documents")
        ingest_dir   = st.text_input("Directory (optional)", placeholder="D:/docs")
        ingest_index = st.text_input("Index name", value="agentic_rag")

        if st.button("▶ Run ingestion", use_container_width=True):
            with st.spinner("Ingesting…"):
                payload: Dict[str, Any] = {"index_name": ingest_index}
                if ingest_dir.strip():
                    payload["directory"] = ingest_dir.strip()
                result = _post("/ingest", payload)
            if result and not result.get("error"):
                st.success(
                    f"✓ {result.get('chunks_indexed', '?')} chunks "
                    f"in {result.get('latency_ms', '?')} ms"
                )
            else:
                st.error(f"Ingestion failed: {result.get('error', 'unknown')}")

        st.divider()

        # ── Thread list ───────────────────────────────────────────────────
        st.markdown("### Past Sessions")
        threads = _get("/threads") or []
        if threads:
            for t in threads[:12]:
                tid   = t["thread_id"]
                turns = t.get("turn_count", 0)
                label = f"{'▶ ' if tid == st.session_state.thread_id else ''}{tid[:20]}… ({turns}t)"
                if st.button(label, key=f"thread_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    st.session_state.messages  = []
                    st.rerun()
        else:
            st.caption("No sessions found.")


# ── Pipeline badge HTML ───────────────────────────────────────────────────────
_NODE_LABELS = {
    "query_analyzer":       "analyze",
    "router_node":          "route",
    "vector_search":        "vector",
    "web_search_tool":      "web",
    "sql_graph_tool":       "sql",
    "context_aggregator":   "aggregate",
    "relevance_grader":     "grade",
    "query_rewriter":       "rewrite",
    "answer_generator":     "generate",
    "hallucination_checker":"verify",
}

def _pipeline_html(nodes: List[str], active: bool = False) -> str:
    if not nodes:
        return ""
    parts = []
    for i, n in enumerate(nodes):
        label = _NODE_LABELS.get(n, n)
        cls   = "node-active" if (active and i == len(nodes) - 1) else "node-done"
        parts.append(f'<span class="{cls}">{label}</span>')
    return '<div class="pipeline">' + "".join(parts) + "</div>"


# ── Main chat area ────────────────────────────────────────────────────────────
def _render_messages() -> None:
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Pipeline trace (stored after streaming)
                nodes = meta.get("nodes", [])
                if nodes:
                    st.markdown(_pipeline_html(nodes), unsafe_allow_html=True)

                st.markdown(content)

                # Footer: sources + latency
                footer_parts = []
                for src in meta.get("sources", []):
                    footer_parts.append(f'<span class="src-pill">{src}</span>')
                if meta.get("latency_ms"):
                    footer_parts.append(
                        f'<span class="latency">⏱ {meta["latency_ms"]} ms</span>'
                    )
                if meta.get("retry_count", 0):
                    footer_parts.append(
                        f'<span class="latency">↺ {meta["retry_count"]} retries</span>'
                    )
                if footer_parts:
                    st.markdown(" ".join(footer_parts), unsafe_allow_html=True)


def _handle_input(question: str) -> None:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question, "meta": {}})

    thread_id = st.session_state.thread_id
    nodes_seen: List[str] = []
    generation = ""

    with st.chat_message("assistant", avatar="🤖"):
        pipeline_placeholder  = st.empty()
        answer_placeholder    = st.empty()
        answer_placeholder.markdown("_Thinking…_")

        for nodes, gen, done in _stream_answer(question, thread_id):
            nodes_seen = nodes
            if nodes_seen:
                pipeline_placeholder.markdown(
                    _pipeline_html(nodes_seen, active=not done),
                    unsafe_allow_html=True,
                )
            if gen:
                generation = gen
                answer_placeholder.markdown(generation)
            if done:
                break

        if not generation:
            generation = "_(no answer generated)_"
            answer_placeholder.markdown(generation)

    # Store the completed AI message
    st.session_state.messages.append({
        "role":    "ai",
        "content": generation,
        "meta":    {"nodes": nodes_seen},
    })


# ── App layout ────────────────────────────────────────────────────────────────
_render_sidebar()

# Header
st.markdown("""
<div class="rag-header">
  <div class="rag-logo">⚡</div>
  <div>
    <div class="rag-title">Agentic RAG</div>
    <div class="rag-sub">Vector · Web · SQL · Self-corrective</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("#### Suggested questions")
    suggestions = [
        "What documents are in the knowledge base?",
        "Summarise the latest quarterly report",
        "Compare the sales figures across regions",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                _handle_input(suggestion)
                st.rerun()

# Chat history
_render_messages()

# Chat input
if question := st.chat_input("Ask a question…"):
    _handle_input(question)
    st.rerun()