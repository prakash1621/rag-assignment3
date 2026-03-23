"""Document ingestion module — loader, chunker, indexer."""

from .loader import scan_knowledge_base, extract_text_from_file
from .chunker import chunk_documents, ParentChildChunker, SemanticChunker
from .indexer import (
    get_embeddings, create_vector_store, save_vector_store,
    load_vector_store, save_file_metadata, get_file_metadata,
)
