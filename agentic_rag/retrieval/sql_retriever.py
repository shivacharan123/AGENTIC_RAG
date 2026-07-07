

from __future__ import annotations
import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field, SecretStr
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_community.utilities import SQLDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SQLRetrieverConfig:
    """Mirrors the `retrieval.sql` block in config.yaml."""
    # Use repo-root absolute path so Uvicorn/CWD differences don't break SQLite.
    db_uri: str = ""
    llm_model: str          = "qwen/qwen3-coder:free"
    llm_provider: str       = "huggingface"   # "openrouter" or "huggingface"
    llm_temperature: float  = 0.0
    max_rows: int           = 50
    max_retries: int        = 1
    retry_delay: float      = 0.5
    include_tables: List[str] = field(default_factory=list)
    sample_rows_in_prompt: int = 3
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    reuse_passed_llm: bool = False   # explicit opt-in to share a caller-provided LLM instance


# ---------------------------------------------------------------------------
# Pydantic structured output schemas
# ---------------------------------------------------------------------------

def _strip_markdown_sql(v: str) -> str:
    """Helper utility to ensure markdown blocks don't trigger validation failures."""
    cleaned = v.strip()
    if cleaned.lower().startswith("```sql"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


class SQLQueryOutput(BaseModel):
    is_relevant: bool = Field(
        description="True only if the query can be answered using the schema. Set False if out of scope."
    )
    sql_query: str = Field(
        description="A single, valid, read-only SELECT statement. Empty string if is_relevant is False."
    )
    reasoning: str = Field(
        default="", description="Brief explanation of which tables/columns were used."
    )

    @field_validator("sql_query")
    @classmethod
    def clean_and_validate_sql(cls, v: str) -> str:
        cleaned = _strip_markdown_sql(v)
        if cleaned and not cleaned.upper().startswith("SELECT"):
            raise ValueError(f"Only SELECT statements are allowed. Got: {cleaned[:80]!r}")
        return cleaned

    @field_validator("sql_query")
    @classmethod
    def no_destructive_keywords(cls, v: str) -> str:
        danger = {"DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE"}
        upper = v.upper()
        found = [kw for kw in danger if re.search(rf"\b{kw}\b", upper)]
        if found:
            raise ValueError(f"Forbidden SQL keyword(s) detected: {found}. Query rejected.")
        return v


class SQLCorrectionOutput(BaseModel):
    sql_query: str = Field(
        description="A corrected, valid SELECT statement fixing the previous error."
    )
    explanation: str = Field(description="What was wrong and how it was fixed.")

    @field_validator("sql_query")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        cleaned = _strip_markdown_sql(v)
        if cleaned and not cleaned.upper().startswith("SELECT"):
            raise ValueError(f"Only SELECT statements are allowed. Got: {cleaned[:80]!r}")
        return cleaned


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_NL_TO_SQL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert, safety-conscious SQL generator.\n\n"
            "DATABASE SCHEMA:\n{schema}\n\n"
            "RULES:\n"
            "1. Only write read-only SELECT statements.\n"
            "2. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.\n"
            "3. Always LIMIT results to {max_rows} rows.\n"
            "4. If the question cannot be answered with the given schema, "
            "   set is_relevant=False and sql_query=''.\n"
            "5. Prefer explicit column names over SELECT *.\n"
            "6. Use table aliases for readability on JOINs."
        ),
    ),
    ("human", "User question: {query}"),
])

_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert SQL debugger.\n\n"
            "DATABASE SCHEMA:\n{schema}\n\n"
            "The following SQL query failed with this error:\n"
            "SQL: {bad_sql}\n"
            "ERROR: {error}\n\n"
            "Write a corrected SELECT statement that fixes the error. "
            "Follow the same safety rules: read-only, LIMIT {max_rows} rows."
        ),
    ),
    ("human", "Original question: {query}"),
])


# ---------------------------------------------------------------------------
# Row -> Document conversion
# ---------------------------------------------------------------------------

def _rows_to_documents(
    rows: List[Dict[str, Any]], query: str, sql: str, table_names: List[str],
) -> List[Document]:
    if not rows:
        return []

    docs: List[Document] = []

    if len(rows) > 1:
        summary_lines = [f"SQL query returned {len(rows)} row(s) for: '{query}'"]
        for i, row in enumerate(rows[:5], 1):
            summary_lines.append(f"  Row {i}: " + " | ".join(f"{k}: {v}" for k, v in row.items()))
        if len(rows) > 5:
            summary_lines.append(f"  ... and {len(rows) - 5} more row(s).")

        docs.append(Document(
            page_content="\n".join(summary_lines),
            metadata={"source": "sql", "sql_query": sql, "doc_type": "summary", "score": 0.0},
        ))

    for i, row in enumerate(rows):
        content = " | ".join(f"{k}: {v}" for k, v in row.items())
        docs.append(Document(
            page_content=content,
            metadata={"source": "sql", "sql_query": sql, "row_index": i, "doc_type": "row", "score": 0.0},
        ))

    return docs


# ---------------------------------------------------------------------------
# Public SQLRetriever class
# ---------------------------------------------------------------------------

class SQLRetriever:
    _db: Optional['SQLDatabase']
    _llm: BaseChatModel

    def __init__(self, config: SQLRetrieverConfig, llm: Optional[BaseChatModel] = None) -> None:
        # If config didn't specify db_uri, derive a stable absolute path:
        # agentic_rag/data/database.db (relative to this file).
        self.config = config

        if llm is not None and config.reuse_passed_llm:
            # Caller has explicitly opted to share their LLM instance.
            self._llm = llm
        else:
            # Build a dedicated client for this retriever's configured model/provider.
            api_key = self._resolve_api_key(config.llm_provider)
            if not api_key:
                env_var = "HF_TOKEN" if config.llm_provider == "huggingface" else "OPENROUTER_API_KEY"
                raise ValueError(
                    f"{env_var} is missing - required by SQLRetriever for provider "
                    f"'{config.llm_provider}'."
                )
            self._llm = self._build_llm(api_key)

        self._db = self._connect_db()

        self._sql_chain  = _NL_TO_SQL_PROMPT  | self._llm.with_structured_output(SQLQueryOutput)
        self._corr_chain = _CORRECTION_PROMPT | self._llm.with_structured_output(SQLCorrectionOutput)

        logger.info(
            f"SQLRetriever initialised (db={config.db_uri.split('@')[-1]}, "
            f"provider={config.llm_provider}, model={config.llm_model}, "
            f"reused_llm={config.reuse_passed_llm and llm is not None})"
        )

    @staticmethod
    def _resolve_api_key(provider: str) -> Optional[str]:
        if provider == "huggingface":
            # HF_TOKEN is the standard env var name used by huggingface_hub /
            # most HF tooling. HUGGINGFACEHUB_API_TOKEN is accepted as a
            # fallback since some older LangChain integrations still expect it.
            return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        return os.environ.get("OPENROUTER_API_KEY")

    def _build_llm(self, api_key: str) -> BaseChatModel:
        if self.config.llm_provider == "huggingface":
            # Hugging Face's OpenAI-compatible router for Inference Providers.
            # Works with ChatOpenAI exactly like OpenRouter does, just a
            # different base_url + token.
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                base_url="https://router.huggingface.co/v1",
                api_key=SecretStr(api_key),
                **self.config.llm_kwargs,
            )

        # Default: OpenRouter
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(api_key),
            **self.config.llm_kwargs,
        )

    def _connect_db(self) -> Optional['SQLDatabase']:
        try:
            from langchain_community.utilities import SQLDatabase

            # If db_uri wasn't provided, derive a stable absolute path
            # (agentic_rag/data/database.db).
            db_uri = self.config.db_uri
            if not db_uri:
                repo_data_db = (
                    Path(__file__).resolve().parents[1] / "data" / "database.db"
                )
                # sqlite URI for absolute Windows paths:
                #   sqlite:///C:/path/to/db.sqlite
                repo_db_posix = str(repo_data_db).replace("\\", "/")
                db_uri = f"sqlite:///{repo_db_posix}"
            else:
                db_uri = db_uri.replace("\\", "/")

            # SQLAlchemy's documented convention for SQLite is:
            #   sqlite:///relative/path.db        -> 3 slashes, relative
            #   sqlite:////absolute/posix/path.db -> 4 slashes, absolute POSIX
            #   sqlite:///C:/absolute/windows.db   -> 3 slashes, absolute Windows
            #     (the drive letter itself disambiguates it as absolute,
            #      so Windows does NOT need a 4th slash)
            safe_uri = db_uri
            if not safe_uri.startswith("sqlite://"):
                safe_uri = f"sqlite:///{safe_uri}"

            kwargs: dict[str, Any] = {"sample_rows_in_table_info": self.config.sample_rows_in_prompt}
            if self.config.include_tables:
                kwargs["include_tables"] = self.config.include_tables

            db = SQLDatabase.from_uri(safe_uri, **kwargs)
            logger.info(f"SQLRetriever: connected to DB. Tables: {db.get_usable_table_names()}")
            return db
        except Exception as exc:
            logger.error(f"SQLRetriever: failed to connect to DB at {self.config.db_uri}: {exc}")
            return None

    def _get_schema(self) -> str:
        db = self._db
        return db.get_table_info() if db else ""

    def _get_table_names(self) -> List[str]:
        db = self._db
        return db.get_usable_table_names() if db else []

    # -----------------------------------------------------------------
    # Rate-limit-aware retry helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Detects 429 / rate-limit errors across OpenAI-compatible client exceptions."""
        msg = str(exc).lower()
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return status_code == 429 or "429" in msg or "rate-limit" in msg or "rate limit" in msg

    @staticmethod
    def _extract_retry_after(exc: Exception, default: float) -> float:
        """Pulls 'retry_after_seconds' from the provider's error payload if present,
        otherwise falls back to the given default."""
        msg = str(exc)
        match = re.search(r"retry_after_seconds['\"]?\s*[:=]\s*([\d.]+)", msg)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return default

    def nl_to_sql(self, query: str) -> Tuple[str, SQLQueryOutput]:
        schema = self._get_schema()
        if not schema:
            raise RuntimeError("SQLRetriever.nl_to_sql: no schema available.")

        last_exc: Optional[Exception] = None
        delay = self.config.retry_delay  # grows via doubling on consecutive rate-limit hits

        for attempt in range(1, self.config.max_retries + 2):
            try:
                result: SQLQueryOutput = self._sql_chain.invoke({
                    "schema": schema, "query": query, "max_rows": self.config.max_rows,
                })
                return result.sql_query, result
            except Exception as exc:
                last_exc = exc
                if attempt <= self.config.max_retries:
                    if self._is_rate_limit_error(exc):
                        wait_time = self._extract_retry_after(exc, default=delay)
                        logger.warning(
                            f"nl_to_sql: rate-limited (attempt {attempt}/{self.config.max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry."
                        )
                        time.sleep(wait_time)
                        delay *= 2  # double backoff for any further rate-limit hit
                    else:
                        logger.warning(
                            f"nl_to_sql: attempt {attempt}/{self.config.max_retries + 1} failed "
                            f"({exc}). Retrying in {self.config.retry_delay}s."
                        )
                        time.sleep(self.config.retry_delay)

        raise RuntimeError(f"nl_to_sql: all attempts failed. Last error: {last_exc}") from last_exc

    def fetch_structured_data(self, sql: str, query: str = "") -> List[Document]:
        db = self._db
        if not sql or not sql.strip() or not db:
            return []

        upper = sql.upper()
        if "LIMIT" not in upper:
            sql = f"{sql.rstrip(';')} LIMIT {self.config.max_rows}"

        try:
            raw: str = db.run(sql, fetch="all")
            if not raw or raw.strip() in ("", "[]", "None"):
                return []

            rows = self._parse_db_output(raw)
            table_names = self._get_table_names()
            return _rows_to_documents(rows, query, sql, table_names)
        except Exception as exc:
            logger.warning(f"fetch_structured_data execution error: {exc}")
            return []

    @staticmethod
    def _parse_db_output(raw: str) -> List[Dict[str, Any]]:
        import ast
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                rows: List[Dict[str, Any]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        rows.append(item)
                    elif isinstance(item, (tuple, list)):
                        rows.append({f"col_{i}": v for i, v in enumerate(item)})
                    else:
                        rows.append({"value": item})
                return rows
        except (ValueError, SyntaxError):
            pass
        return [{"result": raw}]

    def _self_correct_sql(self, bad_sql: str, error: str, query: str) -> Optional[str]:
        schema = self._get_schema()
        try:
            correction: SQLCorrectionOutput = self._corr_chain.invoke({
                "schema": schema, "bad_sql": bad_sql, "error": error,
                "query": query, "max_rows": self.config.max_rows,
            })
            return correction.sql_query or None
        except Exception as exc:
            logger.warning(f"Self-correction failed: {exc}")
            return None

    def get_structured_data(self, query: str) -> List[Document]:
        db = self._db
        if not query or not query.strip() or not db:
            return []

        try:
            sql, output = self.nl_to_sql(query)
        except RuntimeError:
            return []

        if not output.is_relevant or not sql:
            return []

        docs = self.fetch_structured_data(sql, query=query)

        if not docs and self.config.max_retries > 0:
            corrected_sql = self._self_correct_sql(bad_sql=sql, error="No results returned.", query=query)
            if corrected_sql and corrected_sql != sql:
                docs = self.fetch_structured_data(corrected_sql, query=query)

        return docs

    async def async_get_structured_data(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_structured_data(query))

    def close(self) -> None:
        """Release the underlying SQLAlchemy engine/connection.
        Call this on app shutdown, or in tests, to avoid Windows file-lock
        PermissionErrors when cleaning up SQLite test databases."""
        if self._db is not None:
            try:
                self._db._engine.dispose()
            except Exception as exc:
                logger.warning(f"SQLRetriever.close(): failed to dispose engine: {exc}")


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def build_sql_retriever_from_config(
    config_dict: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
) -> SQLRetriever:
    cfg = SQLRetrieverConfig(
        db_uri=config_dict.get("db_uri", "sqlite:///data/default.db"),
        llm_model=config_dict.get("llm_model", "qwen/qwen3-coder:free"),
        llm_provider=config_dict.get("llm_provider", "openrouter"),
        llm_temperature=float(config_dict.get("llm_temperature", 0.0)),
        max_rows=int(config_dict.get("max_rows", 50)),
        max_retries=int(config_dict.get("max_retries", 1)),
        retry_delay=float(config_dict.get("retry_delay", 0.5)),
        include_tables=config_dict.get("include_tables", []),
        sample_rows_in_prompt=int(config_dict.get("sample_rows_in_prompt", 3)),
        llm_kwargs=config_dict.get("llm_kwargs", {}),
        reuse_passed_llm=bool(config_dict.get("reuse_passed_llm", False)),
    )
    return SQLRetriever(cfg, llm=llm)