"""
Agentic RAG Pipeline — Custom Python implementation (no LangGraph).

Combines Adaptive RAG, Corrective RAG, and Self-RAG in a pure Python
state machine with explicit control flow and loop control.

Flow:
  Router ─┬─ direct_llm ──→ Direct Generate ──→ END
          ├─ web_search ──→ Web Search → Generate → Reflect → END
          └─ vectorstore ──→ Rewriter → Retrieve → Grade Docs
                ┬─ has relevant → Rerank → Generate → Hallucination Check → Answer Grade → END
                └─ all irrelevant ─┬─ first time → Corrective Rewrite → Retrieve (retry)
                                   └─ already retried → Web Search fallback → Generate → END
"""

import logging
from typing import Dict, Callable

from src.nodes import (
    route_query,
    rewrite_query,
    retrieve_docs,
    grade_documents,
    rerank_docs,
    generate_answer,
    generate_direct,
    generate_fallback,
    check_hallucination,
    grade_answer,
    rewrite_query_corrective,
    web_search,
)

logger = logging.getLogger(__name__)


def _merge_state(state: Dict, update: Dict) -> Dict:
    """Merge a node's partial output into the full state (trace is append-only)."""
    for key, value in update.items():
        if key == "trace":
            state["trace"] = state.get("trace", []) + (value or [])
        else:
            state[key] = value
    return state


def _run_node(state: Dict, node_func: Callable, name: str) -> Dict:
    """Run a single node and merge its output into state."""
    logger.info(f"[Pipeline] Running node: {name}")
    update = node_func(state)
    return _merge_state(state, update)


# ─── Main Pipeline Runner ───────────────────────────────────────────────────

def run_agentic_rag(
    question: str,
    vectorstore=None,
    retriever_func=None,
    reranker_func=None,
    generator_func=None,
    max_retries: int = 3,
    cached_chunks=None,
    chat_history=None,
) -> Dict:
    """
    Run the full agentic RAG pipeline.

    Implements:
      - Adaptive RAG: 3-way routing (vectorstore / direct_llm / web_search)
      - Corrective RAG: doc grading → query rewrite → retry → web fallback
      - Self-RAG: hallucination check + answer quality grading with retry loop

    Args:
        question:        User query.
        vectorstore:     FAISS vector store.
        retriever_func:  Function(vectorstore, question) → docs.
        reranker_func:   Function(question, docs) → reranked docs.
        generator_func:  Function(question, docs) → answer string.
        max_retries:     Max self-correction loops (default 3).
        cached_chunks:   Pre-cached retrieved docs (skip retrieval if provided).
        chat_history:    Conversation history for query rewriting.

    Returns:
        Dict with 'answer', 'trace', 'route', 'retry_count', 'from_cache',
        and '_retrieved_docs'.
    """

    # ── Initialise state ─────────────────────────────────────────────────
    state: Dict = {
        "question": question,
        "route": "",
        "documents": [],
        "relevant_docs": [],
        "all_docs_irrelevant": False,
        "answer": "",
        "hallucination_free": True,
        "answer_useful": True,
        "retry_count": 0,
        "max_retries": max_retries,
        "query_rewritten": False,
        "trace": [],
        "chat_history": chat_history,
        "_vectorstore": vectorstore,
        "_retriever_func": retriever_func,
        "_reranker_func": reranker_func,
        "_generator_func": generator_func,
    }

    # If cached chunks provided, skip retrieval + grading
    if cached_chunks:
        state["route"] = "vectorstore"
        state["documents"] = cached_chunks
        state["relevant_docs"] = cached_chunks
        state["all_docs_irrelevant"] = False
        state["trace"].append({
            "node": "cache", "decision": "retrieval cache hit",
            "reason": "Using cached chunks — skipping retrieval and doc grading",
        })
        state = _run_node(state, rerank_docs, "rerank")
        state = _run_generate_and_reflect(state)
        return _build_result(state)

    # ── Step 1: Adaptive RAG — Route the query ───────────────────────────
    state = _run_node(state, route_query, "router")
    route = state.get("route", "vectorstore")

    # ── Branch A: Direct LLM (no retrieval) ──────────────────────────────
    if route == "direct_llm":
        state = _run_node(state, generate_direct, "direct_llm")
        return _build_result(state)

    # ── Branch B: Web Search ─────────────────────────────────────────────
    if route == "web_search":
        state = _run_node(state, web_search, "web_search")
        state = _run_generate_and_reflect(state)
        return _build_result(state)

    # ── Branch C: Vectorstore path ───────────────────────────────────────
    # Step 2: Rewrite query using chat history
    state = _run_node(state, rewrite_query, "rewriter")

    # Step 3: Retrieve documents
    state = _run_node(state, retrieve_docs, "retrieve")

    # Step 4: Corrective RAG — Grade documents
    state = _run_node(state, grade_documents, "grade_docs")

    if state.get("all_docs_irrelevant"):
        # First failure → corrective rewrite + retry
        if not state.get("query_rewritten"):
            state = _run_node(state, rewrite_query_corrective, "corrective_rewrite")
            state = _run_node(state, retrieve_docs, "retrieve")
            state = _run_node(state, grade_documents, "grade_docs")

        # Still irrelevant → web search fallback
        if state.get("all_docs_irrelevant"):
            state = _run_node(state, web_search, "web_search_fallback")
            if not state.get("relevant_docs"):
                state = _run_node(state, generate_fallback, "fallback")
                return _build_result(state)

    # Step 5: Rerank relevant documents
    state = _run_node(state, rerank_docs, "rerank")

    # Step 6-8: Generate → Hallucination check → Answer quality (with retries)
    state = _run_generate_and_reflect(state)

    return _build_result(state)


# ─── Self-RAG reflection loop ───────────────────────────────────────────────

def _run_generate_and_reflect(state: Dict) -> Dict:
    """Generate answer, then Self-RAG reflection loop (max 3 retries)."""
    max_retries = state.get("max_retries", 3)

    while True:
        state = _run_node(state, generate_answer, "generate")
        state = _run_node(state, check_hallucination, "hallucination_check")

        if not state.get("hallucination_free"):
            if state.get("retry_count", 0) < max_retries:
                logger.info(f"[Self-RAG] Hallucination — retrying ({state['retry_count']}/{max_retries})")
                continue
            else:
                logger.warning("[Self-RAG] Max retries on hallucination — returning best answer")
                break

        state = _run_node(state, grade_answer, "grade_answer")

        if state.get("answer_useful"):
            break

        if state.get("retry_count", 0) >= max_retries:
            logger.warning("[Self-RAG] Max retries on answer quality — returning best answer")
            break

        logger.info(f"[Self-RAG] Answer not useful — retrying ({state['retry_count']}/{max_retries})")

    return state


# ─── Result builder ──────────────────────────────────────────────────────────

def _build_result(state: Dict) -> Dict:
    """Extract the final result from pipeline state."""
    return {
        "answer": state.get("answer", "No answer generated."),
        "trace": state.get("trace", []),
        "route": state.get("route", "unknown"),
        "retry_count": state.get("retry_count", 0),
        "from_cache": False,
        "_retrieved_docs": state.get("documents", []),
    }


# ─── Graph info (for demo notebook) ─────────────────────────────────────────

def get_pipeline_info() -> Dict:
    """Return pipeline node and edge info."""
    return {
        "nodes": [
            "router", "rewriter", "retrieve", "grade_docs", "rerank",
            "generate", "direct_llm", "fallback", "hallucination_check",
            "grade_answer", "corrective_rewrite", "web_search",
        ],
        "edges": {
            "router": ["rewriter (vectorstore)", "direct_llm", "web_search"],
            "rewriter": ["retrieve"],
            "retrieve": ["grade_docs"],
            "grade_docs": ["rerank (relevant)", "corrective_rewrite (irrelevant)", "web_search_fallback (retry failed)"],
            "corrective_rewrite": ["retrieve (retry)"],
            "rerank": ["generate"],
            "generate": ["hallucination_check"],
            "hallucination_check": ["grade_answer (grounded)", "generate (retry)"],
            "grade_answer": ["END (useful)", "generate (retry)"],
            "direct_llm": ["END"],
            "fallback": ["END"],
        },
    }
