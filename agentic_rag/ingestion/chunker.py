"""
Document chunking for Agentic RAG.

Data flow:
    List[Document]  (from loader.py)
        ↓
    chunk_documents()
        ↓       ↑ chunk_size / chunk_overlap / separators
    config.yaml     (ingestion section)
        ↓
    List[Document]  (chunks, with chunk metadata)  →  passed to embedder.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..prompts.config import load_config


def chunk_documents(
    documents: list[Document],
    chunk_size:    int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into overlapping chunks, preserving all source metadata.

    Parameters
    ----------
    documents:      Output of loader.load_from_config() / load_from_directory()
    chunk_size:     Override config value (optional)
    chunk_overlap:  Override config value (optional)

    Config keys used (ingestion section):
        ingestion.chunk_size     → default 1000
        ingestion.chunk_overlap  → default 200
        ingestion.separators     → default ["\n\n", "\n", ".", "!", "?"]
    """
    config = load_config()
    ingestion_cfg = config.get("ingestion", {})

    # Resolve final values: explicit arg > config > hard default
    final_chunk_size    = chunk_size    or ingestion_cfg.get("chunk_size",    1000)
    final_chunk_overlap = chunk_overlap or ingestion_cfg.get("chunk_overlap",  200)
    final_separators    = ingestion_cfg.get("separators", ["\n\n", "\n", ".", "!", "?"])

    print(
        f"[chunker] Splitting {len(documents)} document(s) "
        f"(chunk_size={final_chunk_size}, overlap={final_chunk_overlap})"
    )

    # Always instantiate with final values - never mutate after construction
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=final_separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Attach chunk-level metadata (source metadata is already preserved by splitter)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id":   i,
            "chunk_size": len(chunk.page_content),
        })

    print(f"[chunker] Produced {len(chunks)} chunk(s).")
    return chunks


def semantic_chunk_documents(
    documents: list[Document],
    embedding_function=None,
) -> list[Document]:
    """
    Placeholder for future semantic chunking.
    Falls back to recursive splitting until implemented.
    """
    return chunk_documents(documents)