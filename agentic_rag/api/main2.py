"""
main2.py — CLI entrypoint for the Agentic RAG project.

Provides:
  1) Optional ingestion pipeline (loader → chunker → embedder → vector index)
  2) FastAPI application with lifespan management
  3) Health-check and metadata endpoints
  4) Graceful startup / shutdown via uvicorn

Usage examples
--------------
API only (defaults from config.yaml):
    python main2.py

Ingest then serve:
    python main2.py --ingest

Ingest from a custom directory:
    python main2.py --ingest --directory "D:/docs" --index-name my_index

Custom host / port:
    python main2.py --host 127.0.0.1 --port 9000

Hot-reload (development):
    python main2.py --reload --debug

Environment variables
---------------------
HUGGINGFACE_TOKEN   — required by the Embedder
OPENROUTER_API_KEY  — required by the chat/query models (or OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.api.main import app as _rag_app          # existing RAG router/app
from agentic_rag.ingestion.embedder import Embedder
from agentic_rag.prompts.config import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("agentic_rag.main2")


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def run_ingestion(directory: Optional[str], index_name: str, cfg: dict[str, Any]) -> None:
    """Run the full ingestion pipeline and log timing + chunk count."""
    raw_docs_dir: Optional[str] = cfg.get("data", {}).get("raw_docs_dir")
    effective_dir = directory or raw_docs_dir

    if not effective_dir:
        raise ValueError(
            "No ingestion directory supplied. "
            "Pass --directory or set data.raw_docs_dir in config.yaml."
        )

    logger.info(
        "Ingestion started | directory=%s | index_name=%s",
        effective_dir,
        index_name,
    )
    t0 = time.perf_counter()

    embedder = Embedder()

    if directory:
        # Caller specified a directory explicitly → load from that path only
        from agentic_rag.ingestion.loader import load_from_directory  # lazy import

        docs = load_from_directory(effective_dir)
        if not docs:
            raise RuntimeError(f"No documents found in directory: {effective_dir}")

        vector_store = embedder.embed_and_store(docs, index_name=index_name)
    else:
        # Use the full pipeline defined by the embedder (reads config internally)
        vector_store = embedder.run_full_pipeline(index_name=index_name)

    # Try to surface how many chunks were indexed (FAISS-specific; graceful fallback)
    chunks_indexed: Any = getattr(getattr(vector_store, "index", None), "ntotal", "unknown")

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    logger.info(
        "Ingestion complete | chunks_indexed=%s | elapsed_ms=%.0f",
        chunks_indexed,
        elapsed_ms,
    )


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def build_app(cfg: dict[str, Any]) -> FastAPI:
    """
    Build and return the FastAPI application.

    Mounts the existing RAG sub-application and adds:
      - CORS middleware
      - Global exception handler
      - /health endpoint
      - /info  endpoint
    """
    api_cfg: dict[str, Any] = cfg.get("api", {})

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # ── Startup ──────────────────────────────────────────────────────────
        logger.info(
            "Server starting | title=%s | version=%s",
            application.title,
            application.version,
        )
        yield
        # ── Shutdown ─────────────────────────────────────────────────────────
        logger.info("Server shutting down — cleaning up resources.")

    app = FastAPI(
        title=api_cfg.get("title", "Agentic RAG API"),
        description=(
            "Retrieval-Augmented Generation service with optional agentic orchestration. "
            "Ingest documents, query via natural language, and inspect pipeline health."
        ),
        version=api_cfg.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: list[str] = api_cfg.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Health & info endpoints ───────────────────────────────────────────────
    @app.get("/health", tags=["Meta"], summary="Liveness probe")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", tags=["Meta"], summary="API metadata")
    async def info() -> dict[str, Any]:
        return {
            "title": app.title,
            "version": app.version,
            "docs": "/docs",
        }

    # ── Mount the existing RAG application ───────────────────────────────────
    app.mount("/rag", _rag_app)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main2",
        description="Agentic RAG — optional ingestion + FastAPI server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ingest_group = parser.add_argument_group("Ingestion")
    ingest_group.add_argument(
        "--ingest",
        action="store_true",
        help="Run the ingestion pipeline before starting the API server.",
    )
    ingest_group.add_argument(
        "--directory",
        metavar="PATH",
        default=None,
        help="Path to the directory containing raw documents (overrides config.yaml).",
    )
    ingest_group.add_argument(
        "--index-name",
        metavar="NAME",
        default="agentic_rag",
        help="Name of the vector index to write during ingestion.",
    )

    server_group = parser.add_argument_group("Server")
    server_group.add_argument(
        "--host",
        default=None,
        help="Bind host (overrides config.yaml api.host; default 0.0.0.0).",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (overrides config.yaml api.port; default 8000).",
    )
    server_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (>1 disables --reload).",
    )
    server_group.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn hot-reload (development only; forces --workers 1).",
    )
    server_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and uvicorn debug mode.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    load_dotenv()
    args = _parse_args(argv)
    _configure_logging(args.debug)

    # Load config once; share across ingestion + app factory
    cfg: dict[str, Any] = load_config() or {}
    api_cfg: dict[str, Any] = cfg.get("api", {})

    host: str = args.host or api_cfg.get("host", "0.0.0.0")
    port: int = args.port or int(api_cfg.get("port", 8000))

    # --reload requires a single worker
    reload: bool = args.reload or args.debug
    workers: int = 1 if reload else max(1, args.workers)

    # ── Optional ingestion ───────────────────────────────────────────────────
    if args.ingest:
        try:
            run_ingestion(directory=args.directory, index_name=args.index_name, cfg=cfg)
        except (ValueError, RuntimeError) as exc:
            logger.error("Ingestion failed: %s", exc)
            sys.exit(1)

    # ── Build FastAPI app ────────────────────────────────────────────────────
    app = build_app(cfg)

    logger.info(
        "Starting server | host=%s | port=%d | workers=%d | reload=%s",
        host, port, workers, reload,
    )

    uvicorn_kwargs: dict[str, Any] = dict(
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="debug" if args.debug else "info",
        access_log=True,
    )

    if reload:
        # uvicorn.run with reload requires a string import path, not an object
        uvicorn.run("agentic_rag.api.main2:app", **uvicorn_kwargs)
    else:
        uvicorn.run(app, **uvicorn_kwargs)


# Expose the app at module level so `uvicorn agentic_rag.api.main2:app` works
# and --reload mode can import it by string.
_cfg: dict[str, Any] = {}
try:
    _cfg = load_config() or {}
except Exception:
    pass

app = build_app(_cfg)

# Backwards-compatible alias expected by some uvicorn/entrypoint configs.
fastapi_app = app


if __name__ == "__main__":
    main(sys.argv[1:])

