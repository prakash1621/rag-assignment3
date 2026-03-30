"""
Document chunker — splits documents into chunks for embedding.

Strategies available:
- Parent-Child: Large parents (3000 chars) + small children (500 chars)
- Semantic: Embedding-based boundary detection
- Basic: Fixed-size character splitting (legacy)
"""

from app.ingestion import chunk_documents
from src.chunking import ParentChildChunker, SemanticChunker

__all__ = ["chunk_documents", "ParentChildChunker", "SemanticChunker"]
