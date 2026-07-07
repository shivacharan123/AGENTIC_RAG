"""
Document loader for Agentic RAG ingestion pipeline.
Supports: TXT, MD, PDF. Handles metadata extraction and basic cleaning.

Data flow:
    config.yaml (data.raw_docs_dir)
        ↓
    load_from_config()          ← NEW: reads path from config automatically
        ↓
    load_from_directory()       ← can still be called manually with explicit path
        ↓
    load_single_document()
        ↓
    List[Document]  →  passed to chunker.py
"""

import os
import re
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from agentic_rag.prompts.config import load_config  # needed to read data.raw_docs_dir


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic noise reduction: remove excessive whitespace."""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_single_document(file_path: str) -> list[Document]:
    """
    Load one file using the appropriate LangChain loader.
    Returns a list of Documents (one per page for PDFs, one for text files).
    """
    path_obj = Path(file_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    suffix = path_obj.suffix.lower()
    base_metadata = {
        "source":     str(path_obj),
        "title":      path_obj.stem,
        "date_added": datetime.now().isoformat(),
        "file_type":  suffix.lstrip("."),
    }

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(str(path_obj)).load()
            for i, doc in enumerate(docs):
                doc.metadata.update({**base_metadata, "page": i + 1})
                doc.page_content = clean_text(doc.page_content)

        elif suffix in {".txt", ".text"}:
            docs = TextLoader(str(path_obj), encoding="utf-8").load()
            for doc in docs:
                doc.metadata.update(base_metadata)
                doc.page_content = clean_text(doc.page_content)

        elif suffix in {".md", ".markdown"}:
            docs = UnstructuredMarkdownLoader(str(path_obj)).load()
            for doc in docs:
                doc.metadata.update(base_metadata)
                doc.page_content = clean_text(doc.page_content)

        else:
            # Fallback: treat as plain text
            docs = TextLoader(str(path_obj), encoding="utf-8").load()
            for doc in docs:
                doc.metadata.update(base_metadata)
                doc.page_content = clean_text(doc.page_content)

        return docs

    except Exception as e:
        print(f"[loader] Error loading {path_obj}: {e}")
        return [Document(
            page_content=f"ERROR loading document: {e}",
            metadata={**base_metadata, "error": str(e)},
        )]


# ---------------------------------------------------------------------------
# Multi-file loaders
# ---------------------------------------------------------------------------

def load_documents(file_paths: list[str]) -> list[Document]:
    """Load an explicit list of file paths."""
    all_docs: list[Document] = []
    for path in file_paths:
        all_docs.extend(load_single_document(path))
    print(f"[loader] Loaded {len(all_docs)} document(s) from {len(file_paths)} file(s).")
    return all_docs


def load_from_directory(
    directory: str,
    extensions: list[str] | None = None,
) -> list[Document]:
    """
    Recursively load all supported files from *directory*.
    Called by load_from_config() and can also be used directly.
    """
    if extensions is None:
        extensions = [".pdf", ".txt", ".md", ".markdown"]

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(
            f"[loader] Data directory not found: {dir_path}\n"
            f"Check 'data.raw_docs_dir' in config.yaml."
        )

    all_docs: list[Document] = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                all_docs.extend(load_single_document(full_path))

    print(f"[loader] Loaded {len(all_docs)} document(s) from: {dir_path}")
    return all_docs


def load_from_config() -> list[Document]:
    """
    PRIMARY ENTRY POINT.

    Reads data.raw_docs_dir from config.yaml and loads all documents found
    there. This is what embedder.py calls so no path is ever hardcoded by
    the caller.

    Flow:
        config.yaml  →  data.raw_docs_dir
                ↓
        load_from_directory(raw_docs_dir)
                ↓
        List[Document]
    """
    config = load_config()
    raw_docs_dir = config.get("data", {}).get("raw_docs_dir")

    if not raw_docs_dir:
        raise ValueError(
            "[loader] 'data.raw_docs_dir' is missing from config.yaml. "
            "Add it under the 'data:' section."
        )

    print(f"[loader] Reading data directory from config: {raw_docs_dir}")
    return load_from_directory(raw_docs_dir)