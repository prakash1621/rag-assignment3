"""
Agentic RAG Graph - LangGraph state machine combining Adaptive, Corrective, and Self-RAG.

Full flow:
  Router ─┬─ direct_llm ──→ Direct Generate ──→ END
          ├─ web_search ──→ Web Search → Generate → Reflect → END
          └─ vectorstore ──→ Rewriter → Retrieve → Grade Docs ─┬─ has relevant → Rerank → Generate → Hallucination Check → Answer Grade → END
                                                                 └─ all irrelevant ─┬─ first time → Corrective Rewrite → Retrieve (retry)
                                                                                     └─ already retried → Web Search fallback → Generate → END
"""

import logging
from typing import Dict, Optional, Callable
from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .nodes import (
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


# ─── Conditional edge functions ──────────────────────────────────────────────

def route_after_router(state: Dict) -> str:
    route = state.get("route", "vectorstore")
    if route == "direct_llm":
        return "direct_llm"
    if route == "web_search":
        return "web_search"
    return "rewriter"  # vectorstore path goes through rewriter first


def route_after_grading(state: Dict) -> str:
    if not state.get("all_docs_irrelevant"):
        return "rerank"
    # All docs irrelevant — have we already retried?
    if not state.get("query_rewritten"):
        return "corrective_rewrite"  # first failure → rewrite and retry
    return "web_search_fallback"  # already retried → web search


def route_after_hallucination_check(state: Dict) -> str:
    if state.get("hallucination_free"):
        return "grade_answer"
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generate"
    return "grade_answer"


def route_after_answer_grade(state: Dict) -> str:
    if state.get("answer_useful"):
        return END
    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generate"
    return END


def route_after_web_search_fallback(state: Dict) -> str:
    """After web search fallback, generate if we got results, else fallback."""
    if state.get("relevant_docs"):
        return "generate"
    return "fallback"


# ─── Graph Builder ───────────────────────────────────────────────────────────

def build_agentic_rag_graph():
    """Build the full agentic RAG graph with all 3 layers."""
    workflow = StateGraph(AgenticRAGState)

    # Add all nodes
    workflow.add_node("router", route_query)
    workflow.add_node("rewriter", rewrite_query)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("grade_docs", grade_documents)
    workflow.add_node("rerank", rerank_docs)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("direct_llm", generate_direct)
    workflow.add_node("fallback", generate_fallback)
    workflow.add_node("hallucination_check", check_hallucination)
    workflow.add_node("grade_answer", grade_answer)
    workflow.add_node("corrective_rewrite", rewrite_query_corrective)
    workflow.add_node("web_search", web_search)           # direct route
    workflow.add_node("web_search_fallback", web_search)  # fallback after failed retrieval

    # Entry point
    workflow.set_entry_point("router")

    # Router → 3 paths
    workflow.add_conditional_edges("router", route_after_router, {
        "rewriter": "rewriter",
        "direct_llm": "direct_llm",
        "web_search": "web_search",
    })

    # Vectorstore path
    workflow.add_edge("rewriter", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")

    workflow.add_conditional_edges("grade_docs", route_after_grading, {
        "rerank": "rerank",
        "corrective_rewrite": "corrective_rewrite",
        "web_search_fallback": "web_search_fallback",
    })

    # Corrective rewrite → retry retrieval
    workflow.add_edge("corrective_rewrite", "retrieve")

    # Web search fallback → generate or final fallback
    workflow.add_conditional_edges("web_search_fallback", route_after_web_search_fallback, {
        "generate": "generate",
        "fallback": "fallback",
    })

    # Main generation path
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "hallucination_check")

    workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check, {
        "grade_answer": "grade_answer",
        "generate": "generate",
    })

    workflow.add_conditional_edges("grade_answer", route_after_answer_grade, {
        END: END,
        "generate": "generate",
    })

    # Web search direct route → generate → reflect
    workflow.add_edge("web_search", "generate")

    # Terminal nodes
    workflow.add_edge("direct_llm", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# ─── Runner ──────────────────────────────────────────────────────────────────

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

    Args:
        cached_chunks: Pre-cached retrieved docs (from Tier 3).
        chat_history: Conversation history for query rewriting.

    Returns:
        Dict with 'answer', 'trace', '_retrieved_docs', and metadata.
    """
    graph = build_agentic_rag_graph()

    base_state = {
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

    if cached_chunks:
        initial_state = {
            **base_state,
            "question": question,
            "route": "vectorstore",
            "documents": cached_chunks,
            "relevant_docs": cached_chunks,
            "all_docs_irrelevant": False,
            "trace": [{"node": "cache", "decision": "retrieval cache hit", "reason": "Using cached chunks — skipping FAISS retrieval and doc grading"}],
        }
    else:
        initial_state = {
            **base_state,
            "question": question,
        }

    result = graph.invoke(initial_state, {"recursion_limit": 50})

    return {
        "answer": result.get("answer", "No answer generated."),
        "trace": result.get("trace", []),
        "route": result.get("route", "unknown"),
        "retry_count": result.get("retry_count", 0),
        "from_cache": False,
        "_retrieved_docs": result.get("documents", []),
    }
