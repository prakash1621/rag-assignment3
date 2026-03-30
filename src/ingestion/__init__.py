"""
Document ingestion module — loader, chunker, indexer.

Re-exports from app/ modules for backward compatibility while providing
the src/ingestion/ structure expected by the assignment.
"""

from app.ingestion import scan_knowledge_base, extract_text_from_file, chunk_documents
from app.embedding import (
    get_embeddings,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    save_file_metadata,
    get_file_metadata,
)

__all__ = [
    "scan_knowledge_base",
    "extract_text_from_file",
    "chunk_documents",
    "get_embeddings",
    "create_vector_store",
    "save_vector_store",
    "load_vector_store",
    "save_file_metadata",
    "get_file_metadata",
]
