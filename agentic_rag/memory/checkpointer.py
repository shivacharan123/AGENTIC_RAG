"""
memory/checkpointer.py — Short-term session memory (LangGraph persistence)

This module is responsible for building the LangGraph "checkpointer" — the
object that persists `GraphState` snapshots across nodes within a single run,
and across turns within a single `thread_id` (i.e. a chat session).

It plugs directly into graph/workflow.py:

    from agentic_rag.memory.checkpointer import build_checkpointer

    checkpointer = build_checkpointer(config)
    app = workflow.compile(checkpointer=checkpointer)

    # Then every invocation is scoped to a session via thread_id:
    app.invoke(
        {"user_query": "..."},
        config={"configurable": {"thread_id": session_id}},
    )

Three backends are supported, selected via the `memory.checkpointer` block
in config.yaml:

    memory:
      checkpointer:
        backend: "memory"      # "memory" | "sqlite" | "postgres"
        sqlite_path: "data/checkpoints.sqlite"
        postgres_uri: "postgresql://user:pass@host:5432/db"

- "memory"   -> in-process, lost on restart. Good for local dev/tests.
- "sqlite"   -> single-file, durable across restarts. Good for a single
                process / small deployments.
- "postgres" -> shared, durable, safe for multiple worker processes.

All three implementations satisfy LangGraph's BaseCheckpointSaver interface,
so swapping backends never requires touching nodes.py / edges.py / workflow.py.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CheckpointerConfig:
    """Mirrors the `memory.checkpointer` block in config.yaml."""
    backend: str = "memory"  # "memory" | "sqlite" | "postgres"
    sqlite_path: str = "data/checkpoints.sqlite"
    postgres_uri: str = ""
    # If True, missing optional deps (langgraph-checkpoint-sqlite /
    # -postgres) fall back to MemorySaver with a warning instead of raising.
    fallback_to_memory_on_error: bool = True


def _config_from_dict(config_dict: dict[str, Any]) -> CheckpointerConfig:
    return CheckpointerConfig(
        backend=str(config_dict.get("backend", "memory")).lower(),
        sqlite_path=config_dict.get("sqlite_path", "data/checkpoints.sqlite"),
        postgres_uri=config_dict.get("postgres_uri", "") or os.environ.get("CHECKPOINTER_POSTGRES_URI", ""),
        fallback_to_memory_on_error=bool(config_dict.get("fallback_to_memory_on_error", True)),
    )


# ---------------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------------

def _build_memory_saver() -> BaseCheckpointSaver:
    logger.info("Checkpointer: using in-process MemorySaver (non-durable).")
    return MemorySaver()


def _build_sqlite_saver(cfg: CheckpointerConfig) -> BaseCheckpointSaver:
    """
    Builds a durable, file-backed SqliteSaver.

    NOTE: SqliteSaver requires a persistent sqlite3.Connection — the
    `check_same_thread=False` flag is required because LangGraph nodes may
    run on a different thread than the one that opened the connection
    (e.g. inside Streamlit, FastAPI, or LangGraph's own thread pool used
    for parallel fan-out nodes like vector_search / web_search_tool /
    sql_graph_tool).
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ImportError(
            "SQLite checkpointer requires 'langgraph-checkpoint-sqlite'. "
            "Install it with: pip install langgraph-checkpoint-sqlite"
        ) from exc

    db_path = cfg.sqlite_path
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent — creates checkpoint tables on first run

    logger.info(f"Checkpointer: using durable SqliteSaver at '{db_path}'.")
    return saver


def _build_postgres_saver(cfg: CheckpointerConfig) -> BaseCheckpointSaver:
    """
    Builds a durable, shared PostgresSaver — safe for multiple worker
    processes hitting the same checkpoint store concurrently.
    """
    if not cfg.postgres_uri:
        raise ValueError(
            "Checkpointer backend is 'postgres' but no postgres_uri was "
            "provided (set memory.checkpointer.postgres_uri in config.yaml "
            "or the CHECKPOINTER_POSTGRES_URI env var)."
        )

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ImportError(
            "Postgres checkpointer requires 'langgraph-checkpoint-postgres'. "
            "Install it with: pip install langgraph-checkpoint-postgres psycopg[binary]"
        ) from exc

    # PostgresSaver.from_conn_string returns a context manager in recent
    # langgraph versions; we enter it once and keep the saver alive for the
    # lifetime of the process (mirrors the module-level "build once, reuse"
    # pattern already used for vector_store_client / llm in nodes.py).
    cm = PostgresSaver.from_conn_string(cfg.postgres_uri)
    saver = cm.__enter__()
    saver.setup()  # idempotent — creates checkpoint tables on first run

    logger.info("Checkpointer: using durable, shared PostgresSaver.")
    return saver


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

@dataclass
class RagCheckpointer:
    """
    Compatibility wrapper expected by agentic_rag.api.main/main.py.

    It exposes:
      - .checkpointer (LangGraph BaseCheckpointSaver)
      - get_chat_history(thread_id)
      - list_threads()
      - delete_thread(thread_id)
    """
    checkpointer: BaseCheckpointSaver

    def get_chat_history(self, thread_id: str) -> list[dict[str, Any]]:
        """
        Best-effort read of persisted chat_history for a given thread_id.

        Note: Exact method support depends on the underlying checkpointer backend.
        """
        if not thread_id:
            return []

        try:
            # Many LangGraph checkpointers support `get`/`get_tuple` via app.get_state,
            # but here we only have the saver. Use optional capabilities when present.
            if hasattr(self.checkpointer, "get"):
                state = self.checkpointer.get(("configurable", thread_id))  # type: ignore[attr-defined]
                if isinstance(state, dict):
                    return state.get("chat_history", []) or []
        except Exception:
            pass

        return []

    def list_threads(self) -> list[dict[str, Any]]:
        """
        Best-effort listing of thread metadata. Returns [] if backend doesn't support it.
        """
        try:
            if hasattr(self.checkpointer, "list"):
                return self.checkpointer.list()  # type: ignore[attr-defined]
        except Exception:
            pass
        return []

    def delete_thread(self, thread_id: str) -> bool:
        """
        Best-effort deletion of thread checkpoints. Returns False if backend doesn't support it.
        """
        try:
            if hasattr(self.checkpointer, "delete"):
                self.checkpointer.delete(thread_id)  # type: ignore[attr-defined]
                return True
        except Exception:
            pass
        return False


def build_checkpointer(config_dict: Optional[dict[str, Any]] = None) -> BaseCheckpointSaver:
    """
    Build a LangGraph checkpointer from the `memory.checkpointer` block of
    config.yaml.

    Falls back to MemorySaver if the requested backend's optional
    dependency is missing and fallback_to_memory_on_error=True (the
    default) — this keeps local dev/tests working without requiring every
    optional driver to be installed.
    """
    cfg = _config_from_dict(config_dict or {})

    builders = {
        "memory":   lambda: _build_memory_saver(),
        "sqlite":   lambda: _build_sqlite_saver(cfg),
        "postgres": lambda: _build_postgres_saver(cfg),
    }

    build_fn = builders.get(cfg.backend)
    if build_fn is None:
        logger.warning(
            f"Checkpointer: unknown backend '{cfg.backend}', defaulting to 'memory'."
        )
        return _build_memory_saver()

    try:
        return build_fn()
    except Exception as exc:
        if cfg.backend != "memory" and cfg.fallback_to_memory_on_error:
            logger.error(
                f"Checkpointer: failed to build '{cfg.backend}' backend ({exc}). "
                "Falling back to in-process MemorySaver."
            )
            return _build_memory_saver()
        raise


def build_checkpointer_from_config(config_dict: Optional[dict[str, Any]] = None) -> RagCheckpointer:
    """
    Compatibility factory expected by agentic_rag.api.main/main.py.
    """
    cp = build_checkpointer(config_dict)
    return RagCheckpointer(checkpointer=cp)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def make_thread_config(session_id: str, **extra_configurable: Any) -> dict[str, Any]:
    """
    Convenience helper for building the `config` dict passed to
    `app.invoke(...)` / `app.stream(...)`, so callers don't have to
    remember LangGraph's nested {"configurable": {"thread_id": ...}} shape.

    Usage:
        app.invoke(inputs, config=make_thread_config(session_id))
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")

    return {"configurable": {"thread_id": session_id, **extra_configurable}}


def get_session_history(
    app: Any,
    session_id: str,
) -> list[dict[str, Any]]:
    """
    Reads back the persisted chat_history for a given session_id from the
    checkpointer, without re-running the graph. Useful for rehydrating a
    chat UI on page load.

    Returns an empty list if no checkpoint exists yet for this session.
    """
    thread_config = make_thread_config(session_id)
    try:
        snapshot = app.get_state(thread_config)
    except Exception as exc:
        logger.warning(f"get_session_history: failed to read state for '{session_id}': {exc}")
        return []

    if not snapshot or not snapshot.values:
        return []

    return snapshot.values.get("chat_history", [])


@contextmanager
def session_scope(app: Any, session_id: str) -> Iterator[dict[str, Any]]:
    """
    Small ergonomic wrapper for one-off scripts / notebooks:

        with session_scope(app, "user-123") as thread_config:
            app.invoke({"user_query": "Hello"}, config=thread_config)
            app.invoke({"user_query": "Follow up"}, config=thread_config)

    Doesn't open any real resource itself — thread isolation is handled
    entirely by LangGraph's checkpointer keyed on thread_id — but groups
    related calls under one readable block and keeps the thread_config
    construction in one place.
    """
    yield make_thread_config(session_id)