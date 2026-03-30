"""
Agentic RAG Module - Adaptive, Corrective, and Self-Reflective RAG

Combines three techniques:
- Adaptive RAG: Routes queries to vectorstore or direct LLM based on query type
- Corrective RAG: Grades retrieved documents and filters irrelevant ones
- Self-RAG: Checks for hallucinations and answer quality, retries if needed
"""

from .graph import build_agentic_rag_graph
from .state import AgenticRAGState

__all__ = ["build_agentic_rag_graph", "AgenticRAGState"]
