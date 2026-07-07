"""
test_sql_retriever.py — Standalone test harness for sql_retriever.py

What this does, step by step:
    1. Creates a throwaway SQLite test database with a couple of tables + sample rows.
    2. Builds an SQLRetrieverConfig pointing at that test DB.
    3. Instantiates SQLRetriever directly (bypasses nodes.py / load_config() entirely,
       so you can isolate failures to this file only).
    4. Runs through each public method individually with print output, so you can see
       exactly which stage breaks if something does:
         - DB connection      (_connect_db)
         - Schema retrieval   (_get_schema / _get_table_names)
         - NL -> SQL          (nl_to_sql)            <- requires OPENROUTER_API_KEY
         - SQL execution      (fetch_structured_data)
         - Full pipeline      (get_structured_data)
         - Async wrapper      (async_get_structured_data)
         - Safety validators  (SQLQueryOutput rejecting DROP/DELETE/etc.)

Run:
    export OPENROUTER_API_KEY=sk-...          (Windows: set OPENROUTER_API_KEY=sk-...)
    python test_sql_retriever.py

If you only want to test DB/schema/safety logic WITHOUT burning API calls,
set SKIP_LLM_TESTS = True below.
"""

import asyncio
import os
import sqlite3
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# 0. Point this at wherever sql_retriever.py actually lives on disk.
#    Place this test file in the SAME folder as sql_retriever.py, or
#    edit sys.path.insert(...) below to point at that folder.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from sql_retriever import (
        SQLRetriever,
        SQLRetrieverConfig,
        SQLQueryOutput,
        build_sql_retriever_from_config,
    )
except ImportError as e:
    print(f"Could not import sql_retriever.py: {e}")
    print("Make sure this test file sits in the same folder as sql_retriever.py,")
    print("or update sys.path.insert(...) above to point at it.")
    sys.exit(1)

SKIP_LLM_TESTS = False   # flip to True to skip anything that calls the LLM


# ---------------------------------------------------------------------------
# 1. Build a throwaway test database
# ---------------------------------------------------------------------------
TEST_DB_PATH = Path(__file__).resolve().parent / "test_database.db"


def build_test_db(path: Path) -> None:
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL
        )
    """)
    cur.executemany(
        "INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)",
        [
            ("Alice Johnson", "Engineering", 95000),
            ("Bob Smith",     "Sales",       62000),
            ("Carol White",   "Engineering", 105000),
            ("Dan Lee",       "Marketing",   71000),
            ("Eve Patel",     "Engineering", 88000),
        ],
    )

    cur.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget INTEGER NOT NULL
        )
    """)
    cur.executemany(
        "INSERT INTO departments (name, budget) VALUES (?, ?)",
        [
            ("Engineering", 500000),
            ("Sales",       250000),
            ("Marketing",   180000),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Test DB created at: {path}")


# ---------------------------------------------------------------------------
# 2. Individual test stages
# ---------------------------------------------------------------------------

def test_db_connection(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: DB connection ---")
    if retriever._db is None:
        print("FAIL: retriever._db is None - connection failed. Check db_uri formatting.")
        return False
    print("PASS: DB connected.")
    return True


def test_schema_introspection(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: schema introspection ---")
    schema = retriever._get_schema()
    tables = retriever._get_table_names()
    if not schema:
        print("FAIL: _get_schema() returned empty string.")
        return False
    print(f"PASS: Tables found: {tables}")
    print(f"   Schema preview:\n{schema[:300]}...")
    return "employees" in tables and "departments" in tables


def test_safety_validators() -> bool:
    print("\n--- TEST: Pydantic safety validators (no LLM/DB needed) ---")
    ok = True

    try:
        out = SQLQueryOutput(
            is_relevant=True,
            sql_query="```sql\nSELECT name, salary FROM employees\n```",
            reasoning="test",
        )
        assert out.sql_query.upper().startswith("SELECT"), "markdown fences not stripped"
        print(f"PASS: Markdown-fenced SELECT cleaned to: {out.sql_query!r}")
    except Exception as e:
        print(f"FAIL: Valid SELECT with markdown fences was rejected: {e}")
        ok = False

    try:
        SQLQueryOutput(is_relevant=True, sql_query="DELETE FROM employees", reasoning="bad")
        print("FAIL: DELETE statement was NOT rejected - safety validator is broken!")
        ok = False
    except Exception:
        print("PASS: DELETE statement correctly rejected.")

    try:
        SQLQueryOutput(is_relevant=True, sql_query="UPDATE employees SET salary=0", reasoning="bad")
        print("FAIL: UPDATE statement was NOT rejected!")
        ok = False
    except Exception:
        print("PASS: UPDATE statement correctly rejected.")

    try:
        out = SQLQueryOutput(
            is_relevant=True,
            sql_query="SELECT last_update, dropout_rate FROM employees",
            reasoning="column name false-positive check",
        )
        print("PASS: Column names like 'last_update'/'dropout_rate' did not trigger false rejection.")
    except Exception as e:
        print(f"FAIL: False positive - legit column names were rejected: {e}")
        ok = False

    return ok


def test_fetch_structured_data_directly(retriever: SQLRetriever) -> bool:
    """Bypasses the LLM entirely - feeds raw SQL straight to fetch_structured_data."""
    print("\n--- TEST: fetch_structured_data() with hand-written SQL (no LLM) ---")
    docs = retriever.fetch_structured_data(
        "SELECT name, department, salary FROM employees WHERE department = 'Engineering'",
        query="engineers and their salaries",
    )
    if not docs:
        print("FAIL: No documents returned - execution likely failed silently.")
        return False
    print(f"PASS: Got {len(docs)} document(s). Sample:\n{docs[0].page_content[:200]}")
    return True


def test_malformed_sql_returns_empty(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: fetch_structured_data() with malformed SQL (should return []) ---")
    docs = retriever.fetch_structured_data("SELECT nonexistent_column FROM employees")
    if docs == []:
        print("PASS: Malformed SQL correctly returned [] instead of raising.")
        return True
    print(f"FAIL: Expected [], got: {docs}")
    return False


def test_nl_to_sql(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: nl_to_sql() - requires LLM call ---")
    if SKIP_LLM_TESTS:
        print("SKIPPED (SKIP_LLM_TESTS=True)")
        return True
    try:
        sql, output = retriever.nl_to_sql("Which employees work in Engineering?")
        print(f"   is_relevant = {output.is_relevant}")
        print(f"   sql_query   = {sql!r}")
        print(f"   reasoning   = {output.reasoning!r}")
        if not output.is_relevant or not sql:
            print("FAIL: Model marked the query as not relevant or returned empty SQL.")
            return False
        if not sql.upper().startswith("SELECT"):
            print("FAIL: Generated SQL is not a SELECT statement.")
            return False
        print("PASS: nl_to_sql produced a usable SELECT statement.")
        return True
    except Exception as e:
        print(f"FAIL: nl_to_sql raised: {e}")
        return False


def test_full_pipeline(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: get_structured_data() - full pipeline (LLM + DB) ---")
    if SKIP_LLM_TESTS:
        print("SKIPPED (SKIP_LLM_TESTS=True)")
        return True
    docs = retriever.get_structured_data("List all employees with salary above 80000")
    if not docs:
        print("FAIL: Full pipeline returned no documents.")
        return False
    print(f"PASS: Full pipeline returned {len(docs)} document(s).")
    for d in docs[:3]:
        print(f"   - {d.page_content[:120]}")
    return True


def test_irrelevant_query(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: get_structured_data() with an out-of-scope question ---")
    if SKIP_LLM_TESTS:
        print("SKIPPED (SKIP_LLM_TESTS=True)")
        return True
    docs = retriever.get_structured_data("What is the capital of France?")
    if docs == []:
        print("PASS: Out-of-scope query correctly returned [] (is_relevant=False path).")
        return True
    print(f"WARN: Expected [], got {len(docs)} doc(s) - model may have hallucinated relevance.")
    return False


def test_async_wrapper(retriever: SQLRetriever) -> bool:
    print("\n--- TEST: async_get_structured_data() ---")
    if SKIP_LLM_TESTS:
        print("SKIPPED (SKIP_LLM_TESTS=True)")
        return True

    async def _run():
        return await retriever.async_get_structured_data("How many departments are there?")

    docs = asyncio.run(_run())
    if not docs:
        print("FAIL: Async wrapper returned no documents.")
        return False
    print(f"PASS: Async wrapper returned {len(docs)} document(s).")
    return True


def test_factory_function() -> bool:
    print("\n--- TEST: build_sql_retriever_from_config() factory ---")
    if not (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("SKIPPED (no HF_TOKEN set)")
        return True
    config_dict = {
        "db_uri": str(TEST_DB_PATH),
        "llm_provider": "huggingface",
        "llm_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "max_rows": 10,
    }
    factory_retriever = None
    try:
        factory_retriever = build_sql_retriever_from_config(config_dict)
        ok = factory_retriever._db is not None
        print("PASS: Factory built a working SQLRetriever." if ok else "FAIL: Factory built retriever but DB is None.")
        return ok
    except Exception as e:
        print(f"FAIL: Factory raised: {e}")
        return False
    finally:
        # This instance is separate from the main `retriever` used elsewhere in
        # the suite and opens its own engine on the same file - dispose it here
        # or it'll hold a Windows file lock and block cleanup at the end.
        if factory_retriever is not None:
            factory_retriever.close()


# ---------------------------------------------------------------------------
# 3. Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SQL RETRIEVER TEST SUITE")
    print("=" * 70)

    if not (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")) and not SKIP_LLM_TESTS:
        print("\nWARNING: HF_TOKEN is not set. LLM-dependent tests will fail.")
        print("Either export it, or set SKIP_LLM_TESTS = True at the top of this file.\n")

    build_test_db(TEST_DB_PATH)

    config = SQLRetrieverConfig(
        db_uri=str(TEST_DB_PATH),
        llm_provider="huggingface",
        llm_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        max_rows=10,
        max_retries=1,
    )

    try:
        retriever = SQLRetriever(config)
    except Exception as e:
        print(f"\nFATAL: Could not construct SQLRetriever: {e}")
        print("(Likely missing OPENROUTER_API_KEY - __init__ requires it even if")
        print(" SKIP_LLM_TESTS is True. Export it and retry, or temporarily comment")
        print(" out the api_key check in sql_retriever.py to test DB-only paths.)")
        sys.exit(1)

    results = {}
    results["db_connection"]        = test_db_connection(retriever)
    results["schema_introspection"] = test_schema_introspection(retriever)
    results["safety_validators"]    = test_safety_validators()
    results["fetch_direct_sql"]     = test_fetch_structured_data_directly(retriever)
    results["malformed_sql"]        = test_malformed_sql_returns_empty(retriever)
    results["nl_to_sql"]            = test_nl_to_sql(retriever)
    results["full_pipeline"]        = test_full_pipeline(retriever)
    results["irrelevant_query"]     = test_irrelevant_query(retriever)
    results["async_wrapper"]        = test_async_wrapper(retriever)
    results["factory_function"]     = test_factory_function()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    n_passed = sum(results.values())
    n_total = len(results)
    print(f"\n{n_passed}/{n_total} tests passed.")

    # Release the main retriever's SQLAlchemy engine before deleting the file.
    # On Windows, SQLite keeps an OS-level file lock open until the engine
    # is disposed, so unlink() fails with PermissionError (WinError 32)
    # if we skip this step. (test_factory_function disposes its own instance.)
    retriever.close()

    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
            print(f"\nCleaned up {TEST_DB_PATH}")
        except PermissionError:
            print(f"\n(non-fatal) Could not delete {TEST_DB_PATH} - "
                  f"still locked by another process. Delete it manually if needed.")


if __name__ == "__main__":
    main()