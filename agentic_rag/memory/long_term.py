"""
memory/long_term.py — Long-Term Memory (Cross-Session Compression & Fact Store)

Responsibilities:
    • COMPRESSION: Every N turns (config: summary_frequency), compress the
      growing chat_history list into a single LLM-generated summary string.
      The summary REPLACES the raw turns so the context window never overflows.
    • FACT STORE: Persist explicit facts (user preferences, domain entities,
      key decisions) across completely separate sessions in a SQLite table.
      Exposed via store_fact() / retrieve_facts() called from nodes.py.
    • CONTEXT INJECTION: Provide build_memory_context() which nodes.py calls
      at the start of each turn to inject both the session summary and any
      relevant stored facts into GraphState["memory_context"].
    • SESSION SUMMARY PERSISTENCE: Save each compressed summary to SQLite so
      it survives process restarts — the summary is the long-term memory of
      what the user and system discussed, not just the current turn's state.

How it fits in the memory workflow:
    nodes.py  answer_generator (end of every turn):
        long_term.maybe_compress(thread_id, state["chat_history"])
            └─ if len(chat_history) % summary_frequency == 0:
                   summary = _summarise(chat_history)
                   _save_summary(thread_id, summary)
                   return summary  →  state["chat_history"] = [{"role":"system","content":summary}]
                   (single summary message replaces all raw turns)

    nodes.py  query_analyzer (start of every turn):
        context = long_term.build_memory_context(thread_id, current_query)
        state["memory_context"] = context
            ├─ loads stored summary for the thread
            └─ retrieves relevant facts matching the current query

Integration points:
    nodes.py        → maybe_compress(), build_memory_context(), store_fact()
    builder.py      → build_long_term_from_config()  (instantiation only)
    checkpointer.py → no direct link; long_term saves its own SQLite tables
                      (separate db file to keep concerns isolated)

Config block (config.yaml):
    memory:
      long_term:
        db_path: "data/memory/long_term.db"
        summary_frequency: 5          # compress every N turns
        summary_model: "gpt-4o-mini"
        summary_temperature: 0.3
        summary_max_tokens: 512
        max_facts_per_thread: 100     # cap on stored facts per session
        fact_retrieval_top_k: 5       # how many facts to inject per turn
        fact_relevance_threshold: 0.0 # keyword match threshold (0 = return all top-k)
"""

from __future__ import annotations
import os
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from pydantic import SecretStr
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LongTermMemoryConfig:
    db_path: str                    = "data/memory/long_term.db"
    summary_frequency: int          = 5
    summary_model: str              = "llama-3.3-70b-versatile"   # ← was meta-llama/...
    summary_temperature: float      = 0.3
    summary_max_tokens: int         = 512
    max_facts_per_thread: int       = 100
    fact_retrieval_top_k: int       = 5
    fact_relevance_threshold: float = 0.0
    llm_kwargs: Dict[str, Any]      = field(default_factory=dict)
# ---------------------------------------------------------------------------
# Typed fact schema
# ---------------------------------------------------------------------------

class StoredFact(BaseModel):
    """A single persisted fact in the long-term fact store."""
    fact_id: str           = Field(description="SHA-256 hash of thread_id + content.")
    thread_id: str         = Field(description="Session this fact came from.")
    content: str           = Field(description="The fact text.")
    category: str          = Field(
        default="general",
        description=(
            "Semantic category for retrieval: 'preference', 'entity', "
            "'decision', 'constraint', 'general'."
        ),
    )
    created_at: str        = Field(description="ISO datetime string.")
    source_turn: int       = Field(default=0, description="Turn index when fact was stored.")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARISATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise conversation summariser for an Agentic RAG system.\n\n"
            "You will receive a list of conversation turns between a USER and an ASSISTANT. "
            "Produce a CONCISE but COMPLETE summary that captures:\n"
            "  1. What the user was trying to find or accomplish.\n"
            "  2. Key facts, figures, or entities discussed.\n"
            "  3. Which retrieval sources contributed (vector / web / SQL).\n"
            "  4. Any unresolved questions or follow-ups the user indicated.\n"
            "  5. Decisions, preferences, or constraints the user expressed.\n\n"
            "Rules:\n"
            "  • Maximum {max_tokens} words.\n"
            "  • Write in third person: 'The user asked...', 'The system retrieved...'.\n"
            "  • Do NOT include the raw Q&A verbatim — summarise, do not copy.\n"
            "  • Preserve all specific numbers, dates, product names, and entity names.\n"
            "  • Format: one paragraph of narrative + a bullet list of key facts extracted."
        ),
    ),
    (
        "human",
        (
            "Conversation turns to summarise:\n\n{conversation_turns}\n\n"
            "Existing summary (if any — incorporate and extend, do not repeat):\n"
            "{existing_summary}"
        ),
    ),
])

FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an information extractor for a long-term memory system.\n\n"
            "Extract ONLY durable, reusable facts from the conversation turn below. "
            "A durable fact is something that will be useful in a FUTURE session — "
            "not just context for the current question.\n\n"
            "Extract facts in these categories:\n"
            "  • 'preference' — user preferences, styles, or constraints.\n"
            "  • 'entity'     — named products, people, projects, companies.\n"
            "  • 'decision'   — choices or commitments the user made.\n"
            "  • 'constraint' — rules, limits, or requirements the user stated.\n\n"
            "Return ONLY a JSON array of objects:\n"
            '[({{"content": "...", "category": "..."}}), ...]\n\n'  # <-- FIXED: Doubled curly braces here
            "Return [] if no durable facts are present. "
            "Return ONLY the JSON array — no preamble, no markdown."
        ),
    ),
    (
        "human",
        (
            "User message:      {user_message}\n"
            "Assistant answer:  {assistant_answer}"
        ),
    ),
])


# ---------------------------------------------------------------------------
# SQLite schema bootstrap
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_summaries (
    thread_id       TEXT    NOT NULL,
    summary         TEXT    NOT NULL,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    PRIMARY KEY (thread_id)
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id         TEXT    NOT NULL PRIMARY KEY,
    thread_id       TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT 'general',
    created_at      TEXT    NOT NULL,
    source_turn     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_facts_thread   ON facts (thread_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts (category);
"""


# ---------------------------------------------------------------------------
# Long-term memory manager
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Long-term memory manager for the Agentic RAG system.

    Two responsibilities, accessed by nodes.py:

    1. COMPRESSION (called from answer_generator node):
        maybe_compress(thread_id, chat_history, turn_index)
            → if turn_index % summary_frequency == 0 and turn_index > 0:
                compresses raw turns into a summary
                saves summary to SQLite
                returns compressed chat_history (one system message)
            → otherwise: returns chat_history unchanged

    2. CONTEXT INJECTION (called from query_analyzer node):
        build_memory_context(thread_id, current_query)
            → loads the stored summary for this thread (if any)
            → retrieves top-k relevant facts
            → returns a formatted string for injection into GraphState["memory_context"]

    FACT STORE (called from answer_generator node for explicit facts):
        store_fact(thread_id, content, category, source_turn)
        retrieve_facts(thread_id, query, top_k) → List[StoredFact]
        auto_extract_and_store_facts(thread_id, user_msg, assistant_answer, turn)
    """

    def __init__(
        self,
        config: LongTermMemoryConfig,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.config = config
        self._db_path = Path(config.db_path)
        self._ensure_db_dir()
        self._bootstrap_db()

        self._llm = llm or self._build_llm()
        self._summary_chain = SUMMARISATION_PROMPT | self._llm
        self._fact_chain    = FACT_EXTRACTION_PROMPT | self._llm

        logger.info(
            "LongTermMemory initialised (db=%s, freq=%d, model=%s).",
            self._db_path, config.summary_frequency, config.summary_model,
        )

    # ------------------------------------------------------------------
    # PRIMARY API — called from nodes.py
    # ------------------------------------------------------------------

    def maybe_compress(
        self,
        thread_id: str,
        chat_history: List[Dict[str, str]],
        turn_index: int,
    ) -> List[Dict[str, str]]:
        """Conditionally compress chat_history into a summary."""
        should_compress = (
            turn_index > 0
            and self.config.summary_frequency > 0
            and turn_index % self.config.summary_frequency == 0
            and len(chat_history) > 1
        )

        if not should_compress:
            return chat_history

        logger.info(
            "LongTermMemory: compressing %d turns at turn_index=%d for thread %r.",
            len(chat_history), turn_index, thread_id,
        )

        existing_summary = self._load_summary(thread_id)
        new_summary = self._summarise(chat_history, existing_summary or "")

        if new_summary:
            self._save_summary(thread_id, new_summary, turn_count=turn_index)
            logger.info(
                "LongTermMemory: summary saved for thread %r (%d chars).",
                thread_id, len(new_summary),
            )
            return [{"role": "system", "content": f"[Session Summary]\n{new_summary}"}]

        logger.warning(
            "LongTermMemory: summarisation failed for thread %r; keeping raw history.",
            thread_id,
        )
        return chat_history

    def build_memory_context(
        self,
        thread_id: str,
        current_query: str,
    ) -> str:
        """Build the memory context string injected into GraphState at turn start."""
        parts: List[str] = []

        summary = self._load_summary(thread_id)
        if summary:
            parts.append(f"[Prior Session Context]\n{summary}")

        facts = self.retrieve_facts(
            thread_id=thread_id,
            query=current_query,
            top_k=self.config.fact_retrieval_top_k,
        )
        if facts:
            fact_lines = "\n".join(
                f"  [{f.category}] {f.content}" for f in facts
            )
            parts.append(f"[Remembered Facts]\n{fact_lines}")

        if not parts:
            return ""

        context = "\n\n".join(parts)
        logger.debug(
            "LongTermMemory.build_memory_context: %d chars for thread %r.",
            len(context), thread_id,
        )
        return context

    # ------------------------------------------------------------------
    # FACT STORE API
    # ------------------------------------------------------------------

    def store_fact(
        self,
        thread_id: str,
        content: str,
        category: str = "general",
        source_turn: int = 0,
    ) -> StoredFact:
        """Persist a single explicit fact to the long-term fact store."""
        fact_id    = _fact_id(thread_id, content)
        created_at = datetime.utcnow().isoformat()

        conn = sqlite3.connect(str(self._db_path))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE thread_id = ?", (thread_id,)
            ).fetchone()[0]

            if count >= self.config.max_facts_per_thread:
                logger.warning(
                    "LongTermMemory: fact cap reached (%d) for thread %r. "
                    "Oldest fact will be evicted.",
                    self.config.max_facts_per_thread, thread_id,
                )
                conn.execute(
                    """
                    DELETE FROM facts WHERE fact_id = (
                        SELECT fact_id FROM facts WHERE thread_id = ?
                        ORDER BY created_at ASC LIMIT 1
                    )
                    """,
                    (thread_id,),
                )

            conn.execute(
                """
                INSERT OR IGNORE INTO facts
                    (fact_id, thread_id, content, category, created_at, source_turn)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_id, thread_id, content, category, created_at, source_turn),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug(
            "store_fact: [%s] %r for thread %r.", category, content[:60], thread_id
        )
        return StoredFact(
            fact_id=fact_id,
            thread_id=thread_id,
            content=content,
            category=category,
            created_at=created_at,
            source_turn=source_turn,
        )

    def retrieve_facts(
        self,
        thread_id: str,
        query: str = "",
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[StoredFact]:
        """Retrieve relevant facts for a given thread and query."""
        top_k = top_k or self.config.fact_retrieval_top_k

        conn = sqlite3.connect(str(self._db_path))
        try:
            sql = "SELECT fact_id, thread_id, content, category, created_at, source_turn FROM facts WHERE thread_id = ?"
            params: List[Any] = [thread_id]
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY created_at DESC"

            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        facts = [
            StoredFact(
                fact_id=row[0],
                thread_id=row[1],
                content=row[2],
                category=row[3],
                created_at=row[4],
                source_turn=row[5],
            )
            for row in rows
        ]

        if not query:
            return facts[:top_k]

        scored: List[Tuple[float, StoredFact]] = [
            (_keyword_score(query, f.content), f) for f in facts
        ]
        scored.sort(key=lambda x: -x[0])

        filtered = [
            f for score, f in scored
            if score >= self.config.fact_relevance_threshold
        ]
        return filtered[:top_k]

    def auto_extract_and_store_facts(
        self,
        thread_id: str,
        user_message: str,
        assistant_answer: str,
        source_turn: int = 0,
    ) -> List[StoredFact]:
        """Automatically extract and store durable facts from a conversation turn."""
        if not user_message and not assistant_answer:
            return []

        try:
            response = self._fact_chain.invoke({
                "user_message":     user_message,
                "assistant_answer": assistant_answer,
            })
            raw = response.content if hasattr(response, "content") else str(response)
            extracted = self._parse_fact_json(raw)
        except Exception as exc:
            logger.warning("auto_extract_and_store_facts: LLM call failed: %s", exc)
            return []

        stored: List[StoredFact] = []
        for item in extracted:
            content  = (item.get("content") or "").strip()
            category = item.get("category", "general")
            if content:
                fact = self.store_fact(thread_id, content, category, source_turn)
                stored.append(fact)

        logger.info(
            "auto_extract_and_store_facts: %d fact(s) stored for thread %r.",
            len(stored), thread_id,
        )
        return stored

    def get_thread_summary(self, thread_id: str) -> Optional[str]:
        return self._load_summary(thread_id)

    def delete_thread_memory(self, thread_id: str) -> Dict[str, int]:
        conn = sqlite3.connect(str(self._db_path))
        try:
            s_del = conn.execute(
                "DELETE FROM session_summaries WHERE thread_id = ?", (thread_id,)
            ).rowcount
            f_del = conn.execute(
                "DELETE FROM facts WHERE thread_id = ?", (thread_id,)
            ).rowcount
            conn.commit()
        finally:
            conn.close()

        logger.info(
            "delete_thread_memory(%r): %d summary row(s), %d fact(s) deleted.",
            thread_id, s_del, f_del,
        )
        return {"summaries_deleted": s_del, "facts_deleted": f_del}

    # ------------------------------------------------------------------
    # Private Methods & Helpers
    # ------------------------------------------------------------------

    def _summarise(
        self,
        chat_history: List[Dict[str, str]],
        existing_summary: str,
    ) -> str:
        turns_text = _format_turns_for_summary(chat_history)

        try:
            response = self._summary_chain.invoke({
                "conversation_turns": turns_text,
                "existing_summary":   existing_summary or "None.",
                "max_tokens":         self.config.summary_max_tokens,
            })
            summary = response.content if hasattr(response, "content") else str(response)
            return summary.strip()
        except Exception as exc:
            logger.error("LongTermMemory._summarise: LLM call failed: %s", exc)
            return ""

    def _ensure_db_dir(self) -> None:
        db_dir = self._db_path.parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info("LongTermMemory: created db directory %s.", db_dir)

    def _bootstrap_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        conn.close()
        logger.debug("LongTermMemory: database schema ensured at %s.", self._db_path)

    def _load_summary(self, thread_id: str) -> Optional[str]:
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT summary FROM session_summaries WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _save_summary(
        self,
        thread_id: str,
        summary: str,
        turn_count: int = 0,
    ) -> None:
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT INTO session_summaries (thread_id, summary, turn_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                summary    = excluded.summary,
                turn_count = excluded.turn_count,
                updated_at = excluded.updated_at
            """,
            (thread_id, summary, turn_count, now, now),
        )
        conn.commit()
        conn.close()

    def _build_llm(self) -> BaseChatModel:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found — LongTermMemory summarisation disabled.")
            return None
        return ChatGroq(
            model=self.config.summary_model,
            temperature=self.config.summary_temperature,
            max_tokens=self.config.summary_max_tokens,
            api_key=SecretStr(api_key),
        )


    @staticmethod
    def _parse_fact_json(raw: str) -> List[Dict[str, str]]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            parsed = json.loads(raw.strip())
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            pass
        logger.debug("_parse_fact_json: could not parse: %r", raw[:120])
        return []

    def __repr__(self) -> str:
        return (
            f"LongTermMemory(db={self._db_path!r}, "
            f"freq={self.config.summary_frequency}, "
            f"model={self.config.summary_model!r})"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _fact_id(thread_id: str, content: str) -> str:
    return hashlib.sha256(f"{thread_id}::{content}".encode()).hexdigest()[:32]


def _keyword_score(query: str, content: str) -> float:
    query_words   = set(query.lower().split())
    content_lower = content.lower()
    if not query_words:
        return 0.0
    matched = sum(1 for w in query_words if w in content_lower)
    return matched / len(query_words)


def _format_turns_for_summary(chat_history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for i, msg in enumerate(chat_history):
        role    = msg.get("role", "unknown").upper()
        content = (msg.get("content", "") or "").strip()
        if content.startswith("[Session Summary]"):
            lines.append(f"[PRIOR SUMMARY]\n{content.replace('[Session Summary]','').strip()}")
        else:
            lines.append(f"Turn {i // 2 + 1} — {role}:\n{content}")
    return "\n\n".join(lines)


def build_long_term_from_config(
    config_dict: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
) -> LongTermMemory:
    cfg = LongTermMemoryConfig(
        db_path=config_dict.get("db_path", "data/memory/long_term.db"),
        summary_frequency=int(config_dict.get("summary_frequency", 5)),
        summary_model=config_dict.get("summary_model", "llama-3.3-70b-versatile"),  # ← updated default
        summary_temperature=float(config_dict.get("summary_temperature", 0.3)),
        summary_max_tokens=int(config_dict.get("summary_max_tokens", 512)),
        max_facts_per_thread=int(config_dict.get("max_facts_per_thread", 100)),
        fact_retrieval_top_k=int(config_dict.get("fact_retrieval_top_k", 5)),
        fact_relevance_threshold=float(config_dict.get("fact_relevance_threshold", 0.0)),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
    )
    return LongTermMemory(cfg, llm=llm)


if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LongTermMemoryConfig(
        db_path=f"{tmp}/long_term.db",
        summary_frequency=2,
        summary_model="llama-3.3-70b-versatile",   # ← was meta-llama/...
    )
        ltm = LongTermMemory(cfg)
        tid = "test-thread-001"

        ltm.store_fact(tid, "User prefers concise bullet-point answers.", "preference", 0)
        ltm.store_fact(tid, "Project is called Agentic RAG.", "entity", 0)

        facts = ltm.retrieve_facts(tid, query="What does the user prefer?")
        print(f"\nRetrieved {len(facts)} fact(s):")
        for f in facts:
            print(f"  [{f.category}] {f.content}")

        history = [
            {"role": "user",      "content": "What were Product X sales in Q1 2025?"},
            {"role": "assistant", "content": "Product X sold 1.2M units in Q1 2025."},
            {"role": "user",      "content": "How does that compare to Q1 2024?"},
            {"role": "assistant", "content": "Q1 2024 had 0.9M units, so growth was 33%."},
        ]
        compressed = ltm.maybe_compress(tid, history, turn_index=2)
        print(f"\nCompressed history has {len(compressed)} message(s).")
        if compressed:
            print(f"Summary preview: {compressed[0]['content'][:200]}")

        ctx = ltm.build_memory_context(tid, "What are Product X's Q2 2025 projections?")
        print(f"\nMemory context ({len(ctx)} chars):\n{ctx[:300]}")

        print("\nSmoke-test passed.")