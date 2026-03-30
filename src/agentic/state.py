"""
Agentic RAG State Definition

Uses Annotated fields with reducers so LangGraph properly merges
partial node outputs into the full accumulated state.
"""

from typing import List, Dict, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
import operator


def replace_value(existing, new):
    """Reducer that replaces the old value with the new one."""
    return new


def merge_list(existing, new):
    """Reducer that extends the existing list with new items."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    return existing + new


class AgenticRAGState(TypedDict):
    """State passed between nodes in the agentic RAG graph."""
    # Input — set once, never changes
    question: str

    # Routing
    route: Annotated[str, replace_value]

    # Retrieval
    documents: Annotated[list, replace_value]

    # Grading
    relevant_docs: Annotated[list, replace_value]
    all_docs_irrelevant: Annotated[bool, replace_value]

    # Generation
    answer: Annotated[str, replace_value]

    # Reflection
    hallucination_free: Annotated[bool, replace_value]
    answer_useful: Annotated[bool, replace_value]
    retry_count: Annotated[int, replace_value]
    max_retries: int

    # Corrective RAG: track if we already rewrote the query
    query_rewritten: Annotated[bool, replace_value]

    # Injected dependencies (not modified by nodes)
    _vectorstore: Optional[object]
    _retriever_func: Optional[object]
    _reranker_func: Optional[object]
    _generator_func: Optional[object]

    # Chat history for conversational context
    chat_history: Optional[list]

    # Trace — append-only
    trace: Annotated[list, merge_list]
