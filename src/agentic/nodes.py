"""
Agentic RAG Nodes - LLM-powered decision functions for the graph.

Each node takes the graph state, performs one decision, and returns updated state.
IMPORTANT: trace field uses a merge_list reducer, so nodes must return ONLY new trace entries.
"""

import json
import logging
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.generation.generator import get_llm
from app.config import CATEGORY_KEYWORDS

logger = logging.getLogger(__name__)

# Build a descriptive list of KB categories for the router prompt
_CATEGORY_DESCRIPTIONS = {
    "Teradata": "Teradata data warehouse and SQL discrepancies",
    "pusa-sell-kb": "PUSA passenger seat sell system (Redshift, UAT)",
    "dot": "DOT project — fare calculations, currency conversion, DOT compliance",
    "avaya": "Avaya call center module — SMS opt-in, agent traces, call history",
    "swav": "SWAV vacation system — QMO automation",
    "galaxy": "Galaxy currency conversion system",
    "bppsl": "BPPSL booking/fare proration at segment and leg level",
}
_category_list = "; ".join(
    f'"{name}" ({desc})' for name, desc in _CATEGORY_DESCRIPTIONS.items()
    if name in CATEGORY_KEYWORDS
)


# ─── Adaptive RAG: Query Router ─────────────────────────────────────────────

ROUTER_PROMPT = PromptTemplate(
    input_variables=["question", "categories"],
    template="""You are a query router for a knowledge-base Q&A system.

The knowledge base contains ONLY internal organizational documents about these projects/categories:
{categories}

IMPORTANT: Words like "dot", "avaya", "bppsl", "galaxy", "swav", "pusa" are PROJECT NAMES in this organization, not generic English words. If the user mentions any of these, route to vectorstore.

Given the user question, decide the best route:
- "vectorstore": The question mentions ANY of the project/category names above, or asks about internal organizational systems, processes, tools, data, tables, or architecture.
- "web_search": The question requires current/external information NOT related to any of the above projects (e.g. latest news, public APIs, external tools, industry trends).
- "direct_llm": The question is a greeting, chitchat, general knowledge about public figures, or simple factual question that needs no retrieval.

Respond with ONLY a JSON object: {{"route": "vectorstore"}} or {{"route": "web_search"}} or {{"route": "direct_llm"}}

Question: {question}"""
)


def route_query(state: Dict) -> Dict:
    """Adaptive RAG: Route query to vectorstore or direct LLM."""
    question = state["question"]

    llm = get_llm()
    chain = ROUTER_PROMPT | llm
    result = chain.invoke({"question": question, "categories": _category_list})

    try:
        parsed = json.loads(result.content.strip())
        route = parsed.get("route", "vectorstore")
    except (json.JSONDecodeError, AttributeError):
        route = "vectorstore"

    if route not in ("vectorstore", "direct_llm", "web_search"):
        route = "vectorstore"

    logger.info(f"[Router] Query routed to: {route}")

    return {
        "route": route,
        "trace": [{"node": "router", "decision": route, "reason": f"Query routed to {route}"}],
    }


# ─── Conversational Query Rewriter ──────────────────────────────────────────

REWRITE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Rewrite the follow-up question as a standalone question using context from the chat history. Output ONLY the rewritten question, nothing else.

If the question is already standalone, output it unchanged.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)


def rewrite_query(state: Dict) -> Dict:
    """Rewrite follow-up questions using chat history for context."""
    question = state["question"]
    chat_history = state.get("chat_history") or []

    # Skip rewriting if no chat history
    if not chat_history:
        return {
            "trace": [{"node": "rewriter", "decision": "skipped", "reason": "No chat history — using original question"}],
        }

    # Build a concise history string (last 4 messages max)
    recent = chat_history[-4:]
    history_str = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])

    llm = get_llm()
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"chat_history": history_str, "question": question})

    rewritten = result.content.strip().split("\n")[0].strip()

    # Only use rewritten if it's meaningfully different
    if rewritten and rewritten.lower() != question.lower():
        logger.info(f"[Rewriter] '{question}' → '{rewritten}'")
        return {
            "question": rewritten,
            "trace": [{"node": "rewriter", "decision": "rewritten", "reason": f"'{question}' → '{rewritten}'"}],
        }

    return {
        "trace": [{"node": "rewriter", "decision": "unchanged", "reason": "Question is already standalone"}],
    }# ─── Retrieval Node ──────────────────────────────────────────────────────────

def retrieve_docs(state: Dict) -> Dict:
    """Retrieve documents from vectorstore."""
    question = state["question"]
    vectorstore = state.get("_vectorstore")
    retriever_func = state.get("_retriever_func")

    if not vectorstore or not retriever_func:
        return {
            "documents": [],
            "trace": [{"node": "retrieve", "decision": "error", "reason": "Vectorstore or retriever not available"}],
        }

    docs = retriever_func(vectorstore, question)
    logger.info(f"[Retrieve] Got {len(docs)} documents")

    return {
        "documents": docs,
        "trace": [{"node": "retrieve", "decision": f"{len(docs)} docs", "reason": f"Retrieved {len(docs)} documents from vector store"}],
    }


# ─── Corrective RAG: Document Grader ────────────────────────────────────────

GRADER_PROMPT = PromptTemplate(
    input_variables=["document", "question"],
    template="""You are checking if a document should be EXCLUDED from answering a question.

Only respond {{"relevant": false}} if the document is completely unrelated to the question and contains NO useful information for ANY part of the question.

If the document discusses ANY topic, keyword, project, or concept mentioned in the question, respond {{"relevant": true}}.

When in doubt, respond {{"relevant": true}}.

Respond with ONLY a JSON object.

Document:
{document}

Question: {question}"""
)


def grade_documents(state: Dict) -> Dict:
    """Corrective RAG: Grade each retrieved document for relevance."""
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            "relevant_docs": [],
            "all_docs_irrelevant": True,
            "trace": [{"node": "grader", "decision": "no docs", "reason": "No documents retrieved"}],
        }

    llm = get_llm()
    chain = GRADER_PROMPT | llm

    relevant_docs = []
    for doc in documents:
        content = doc.page_content[:1500]
        result = chain.invoke({"document": content, "question": question})

        try:
            parsed = json.loads(result.content.strip())
            is_relevant = parsed.get("relevant", False)
        except (json.JSONDecodeError, AttributeError):
            is_relevant = True

        if is_relevant:
            relevant_docs.append(doc)

    all_irrelevant = len(relevant_docs) == 0
    logger.info(f"[Grader] {len(relevant_docs)}/{len(documents)} docs relevant")

    return {
        "relevant_docs": relevant_docs,
        "all_docs_irrelevant": all_irrelevant,
        "trace": [{"node": "grader", "decision": f"{len(relevant_docs)}/{len(documents)} relevant", "reason": f"Kept {len(relevant_docs)} of {len(documents)} documents after relevance grading"}],
    }


# ─── Rerank Node ─────────────────────────────────────────────────────────────

def rerank_docs(state: Dict) -> Dict:
    """Rerank relevant documents using the existing reranker."""
    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])
    reranker_func = state.get("_reranker_func")

    if reranker_func and relevant_docs:
        reranked = reranker_func(question, relevant_docs)
        return {
            "relevant_docs": reranked,
            "trace": [{"node": "rerank", "decision": f"top {len(reranked)}", "reason": f"Reranked to top {len(reranked)} documents"}],
        }

    return {
        "trace": [{"node": "rerank", "decision": "skipped", "reason": "No reranker or no docs"}],
    }


# ─── Generation Nodes ────────────────────────────────────────────────────────

def generate_answer(state: Dict) -> Dict:
    """Generate answer from relevant documents."""
    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])
    generator_func = state.get("_generator_func")
    route = state.get("route", "")

    # Check if docs came from web search (category == "web_search")
    is_web = route == "web_search" or (
        relevant_docs and any(
            d.metadata.get("category") == "web_search" for d in relevant_docs
        )
    )

    if is_web and relevant_docs:
        # Use a web-friendly prompt instead of the strict KB-only prompt
        context = "\n\n".join([
            f"[{d.metadata.get('filename', 'web')}]\n{d.page_content}"
            for d in relevant_docs
        ])
        llm = get_llm()
        web_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant. Answer the user's question based on the web search results provided below.
Synthesize the information clearly and concisely. Cite the source URL when possible.

Web search results:
{context}

Question: {question}"""
        )
        chain = web_prompt | llm
        result = chain.invoke({"context": context, "question": question})
        answer = result.content

        # Add source URLs
        sources = [d.metadata.get("source", "") for d in relevant_docs if d.metadata.get("source")]
        if sources:
            answer += "\n\n---\n**🌐 Sources:** " + ", ".join(sources[:3])
    elif generator_func and relevant_docs:
        answer = generator_func(question, relevant_docs)
    elif not relevant_docs:
        answer = "I couldn't find relevant information in the knowledge base to answer your question. Could you rephrase or ask about a specific topic covered in the documents?"
    else:
        answer = "Generator not available."

    logger.info("[Generate] Answer generated")

    return {
        "answer": answer,
        "trace": [{"node": "generate", "decision": "answer generated", "reason": "Generated answer from relevant documents"}],
    }


def generate_direct(state: Dict) -> Dict:
    """Generate answer directly from LLM without retrieval."""
    question = state["question"]

    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a friendly KT Onboarding Assistant. The user's message doesn't require document lookup.
Respond naturally and helpfully. If they greet you, greet them back and let them know you can help with questions about organizational documents.

User: {question}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question})

    logger.info("[Direct LLM] Responded without retrieval")

    return {
        "answer": result.content,
        "trace": [{"node": "direct_llm", "decision": "direct response", "reason": "Answered without retrieval (greeting/general query)"}],
    }


def generate_fallback(state: Dict) -> Dict:
    """Fallback generation when all retrieved docs are irrelevant."""
    question = state["question"]

    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a KT Onboarding Assistant. The user asked a question but no relevant documents were found in the knowledge base.

Politely inform the user that the information they're looking for isn't available in the current documentation. Suggest they:
1. Rephrase their question with more specific terms
2. Check if the relevant documents have been uploaded to the knowledge base
3. Contact their team lead for information not yet documented

User question: {question}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question})

    logger.info("[Fallback] All docs irrelevant, generated fallback")

    return {
        "answer": result.content,
        "trace": [{"node": "fallback", "decision": "no relevant docs", "reason": "All retrieved documents were irrelevant — generated fallback response"}],
    }


# ─── Self-RAG: Hallucination Checker ────────────────────────────────────────

HALLUCINATION_PROMPT = PromptTemplate(
    input_variables=["documents", "answer"],
    template="""You are a hallucination grader. Given a set of source documents and an LLM-generated answer, determine if the answer is grounded in the documents.

An answer is grounded if every factual claim it makes can be traced back to the provided documents. Minor phrasing differences are OK.

Respond with ONLY a JSON object: {{"grounded": true}} or {{"grounded": false}}

Source documents:
{documents}

Generated answer:
{answer}"""
)


def check_hallucination(state: Dict) -> Dict:
    """Self-RAG: Check if the generated answer is grounded in retrieved documents."""
    answer = state.get("answer", "")
    relevant_docs = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)

    if not relevant_docs:
        return {
            "hallucination_free": True,
            "trace": [{"node": "hallucination_check", "decision": "skipped", "reason": "No docs to check against"}],
        }

    docs_text = "\n\n---\n\n".join([d.page_content[:1000] for d in relevant_docs[:3]])

    llm = get_llm()
    chain = HALLUCINATION_PROMPT | llm
    result = chain.invoke({"documents": docs_text, "answer": answer})

    try:
        parsed = json.loads(result.content.strip())
        grounded = parsed.get("grounded", True)
    except (json.JSONDecodeError, AttributeError):
        grounded = True

    logger.info(f"[Hallucination Check] Grounded: {grounded}")

    update = {
        "hallucination_free": grounded,
        "trace": [{
            "node": "hallucination_check",
            "decision": "grounded" if grounded else "hallucination detected",
            "reason": f"Answer {'is' if grounded else 'is NOT'} grounded in source documents (attempt {retry_count + 1})",
        }],
    }
    # Increment retry_count on hallucination so the retry loop has a hard stop
    if not grounded:
        update["retry_count"] = retry_count + 1

    return update


# ─── Self-RAG: Answer Quality Grader ────────────────────────────────────────

ANSWER_GRADER_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an answer quality grader. Given a user question and a generated answer, determine if the answer actually addresses and is useful for the question.

Respond with ONLY a JSON object: {{"useful": true}} or {{"useful": false}}

Question: {question}

Answer: {answer}"""
)


def grade_answer(state: Dict) -> Dict:
    """Self-RAG: Grade if the answer actually addresses the question."""
    question = state["question"]
    answer = state.get("answer", "")
    retry_count = state.get("retry_count", 0)

    llm = get_llm()
    chain = ANSWER_GRADER_PROMPT | llm
    result = chain.invoke({"question": question, "answer": answer})

    try:
        parsed = json.loads(result.content.strip())
        useful = parsed.get("useful", True)
    except (json.JSONDecodeError, AttributeError):
        useful = True

    logger.info(f"[Answer Grader] Useful: {useful}")

    return {
        "answer_useful": useful,
        "retry_count": retry_count + 1,
        "trace": [{
            "node": "answer_grader",
            "decision": "useful" if useful else "not useful",
            "reason": f"Answer {'addresses' if useful else 'does NOT address'} the question (attempt {retry_count + 1})",
        }],
    }


# ─── Corrective RAG: Query Rewriter (on grading failure) ────────────────────

CORRECTIVE_REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a query rewriter. The original question failed to retrieve relevant documents from the knowledge base.

Rewrite the question to be more specific, use alternative keywords, or broaden the scope to improve retrieval.

Output ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""
)


def rewrite_query_corrective(state: Dict) -> Dict:
    """Corrective RAG: Rewrite query when all retrieved docs are irrelevant."""
    question = state["question"]

    llm = get_llm()
    chain = CORRECTIVE_REWRITE_PROMPT | llm
    result = chain.invoke({"question": question})

    rewritten = result.content.strip().split("\n")[0].strip()
    logger.info(f"[Corrective Rewriter] '{question}' → '{rewritten}'")

    return {
        "question": rewritten,
        "all_docs_irrelevant": False,  # reset for retry
        "query_rewritten": True,  # mark so we don't retry again
        "trace": [{"node": "corrective_rewrite", "decision": "query rewritten", "reason": f"Docs irrelevant — rewrote: '{question}' → '{rewritten}'"}],
    }


# ─── Web Search Node (Tavily) ───────────────────────────────────────────────

def web_search(state: Dict) -> Dict:
    """Fallback: Search the web using Tavily when vectorstore fails."""
    import os
    question = state["question"]

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        logger.warning("[Web Search] No TAVILY_API_KEY set — skipping web search")
        return {
            "relevant_docs": [],
            "all_docs_irrelevant": True,
            "trace": [{"node": "web_search", "decision": "skipped", "reason": "TAVILY_API_KEY not configured"}],
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query=question, max_results=3)

        docs = []
        for r in results.get("results", []):
            doc = Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", ""), "category": "web_search", "filename": r.get("title", "")}
            )
            docs.append(doc)

        logger.info(f"[Web Search] Got {len(docs)} results from Tavily")

        return {
            "relevant_docs": docs,
            "all_docs_irrelevant": len(docs) == 0,
            "trace": [{"node": "web_search", "decision": f"{len(docs)} results", "reason": f"Web search returned {len(docs)} results"}],
        }
    except Exception as e:
        logger.error(f"[Web Search] Error: {e}")
        return {
            "relevant_docs": [],
            "all_docs_irrelevant": True,
            "trace": [{"node": "web_search", "decision": "error", "reason": f"Web search failed: {str(e)[:100]}"}],
        }
