"""
Vector store indexer — creates, saves, and loads FAISS vector stores.
"""

from app.embedding import (
    get_embeddings,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    save_file_metadata,
    get_file_metadata,
)

__all__ = [
    "get_embeddings",
    "create_vector_store",
    "save_vector_store",
    "load_vector_store",
    "save_file_metadata",
    "get_file_metadata",
]
