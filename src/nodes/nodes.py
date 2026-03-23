"""
Agentic RAG Nodes — router, retriever, grader, rewriter, generator, fallback.

Each node takes the pipeline state dict, performs one decision, and returns
a partial state update. The trace field is append-only.
"""

import os
import json
import logging
import numpy as np
import boto3
from typing import Dict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
LLM_MODEL = os.environ.get("AWS_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
EMBEDDING_MODEL = os.environ.get("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
RETRIEVAL_K = 10
RERANK_TOP_K = 3

CATEGORY_KEYWORDS = {
    "dot": ["dot", "fare", "currency", "calculation"],
    "avaya": ["avaya", "qa", "validation", "regression", "testing"],
    "bppsl": ["bppsl", "reference"],
}

_CATEGORY_DESCRIPTIONS = {
    "dot": "DOT project — fare calculations, currency conversion, DOT compliance",
    "avaya": "Avaya call center module — SMS opt-in, agent traces, call history",
    "bppsl": "BPPSL booking/fare proration at segment and leg level",
}
_category_list = "; ".join(
    f'"{name}" ({desc})' for name, desc in _CATEGORY_DESCRIPTIONS.items()
)


def get_llm():
    """Initialize AWS Bedrock LLM client."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return ChatBedrock(client=bedrock, model_id=LLM_MODEL, temperature=0)


def get_embeddings():
    """Get Bedrock embeddings client."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return BedrockEmbeddings(client=bedrock, model_id=EMBEDDING_MODEL)


# ─── Adaptive RAG: Query Router ─────────────────────────────────────────────

ROUTER_PROMPT = PromptTemplate(
    input_variables=["question", "categories"],
    template="""You are a query router for a knowledge-base Q&A system.

The knowledge base contains ONLY internal organizational documents about these projects/categories:
{categories}

IMPORTANT: Words like "dot", "avaya", "bppsl" are PROJECT NAMES in this organization, not generic English words. If the user mentions any of these, route to vectorstore.

Given the user question, decide the best route:
- "vectorstore": The question mentions ANY of the project/category names above, or asks about internal organizational systems, processes, tools, data, tables, or architecture.
- "web_search": The question requires current/external information NOT related to any of the above projects (e.g. latest news, public APIs, external tools, industry trends).
- "direct_llm": The question is a greeting, chitchat, general knowledge, or simple factual question that needs no retrieval.

Respond with ONLY a JSON object: {{"route": "vectorstore"}} or {{"route": "web_search"}} or {{"route": "direct_llm"}}

Question: {question}"""
)


def route_query(state: Dict) -> Dict:
    """Adaptive RAG: Route query to vectorstore, direct LLM, or web search."""
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

    if not chat_history:
        return {
            "trace": [{"node": "rewriter", "decision": "skipped", "reason": "No chat history"}],
        }

    recent = chat_history[-4:]
    history_str = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])

    llm = get_llm()
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"chat_history": history_str, "question": question})
    rewritten = result.content.strip().split("\n")[0].strip()

    if rewritten and rewritten.lower() != question.lower():
        logger.info(f"[Rewriter] '{question}' → '{rewritten}'")
        return {
            "question": rewritten,
            "trace": [{"node": "rewriter", "decision": "rewritten", "reason": f"'{question}' → '{rewritten}'"}],
        }

    return {
        "trace": [{"node": "rewriter", "decision": "unchanged", "reason": "Question is already standalone"}],
    }


# ─── Retriever ───────────────────────────────────────────────────────────────

def _detect_categories(question):
    """Detect relevant KB categories from query keywords."""
    question_lower = question.lower()
    return [cat for cat, kws in CATEGORY_KEYWORDS.items()
            if any(kw in question_lower for kw in kws)]


def retrieve_docs(state: Dict) -> Dict:
    """Retrieve documents from vectorstore with category-aware balancing."""
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
        "trace": [{"node": "retrieve", "decision": f"{len(docs)} docs", "reason": f"Retrieved {len(docs)} documents"}],
    }


def retrieve_documents(vectorstore, question):
    """Standalone retriever: fetch top-k chunks with category balancing."""
    detected = _detect_categories(question)

    if len(detected) == 1:
        docs = vectorstore.similarity_search(question, k=RETRIEVAL_K, filter={"category": detected[0]})
        if not docs:
            docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    elif len(detected) > 1:
        docs, seen = [], set()
        per_cat_k = max(RETRIEVAL_K // len(detected), 3)
        for cat in detected:
            for doc in vectorstore.similarity_search(question, k=per_cat_k, filter={"category": cat}):
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
        if len(docs) < RETRIEVAL_K:
            for doc in vectorstore.similarity_search(question, k=RETRIEVAL_K):
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
                if len(docs) >= RETRIEVAL_K:
                    break
    else:
        docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)

    return docs


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
    """Corrective RAG: Grade each retrieved document for relevance (yes/no)."""
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
        "trace": [{"node": "grader", "decision": f"{len(relevant_docs)}/{len(documents)} relevant",
                    "reason": f"Kept {len(relevant_docs)} of {len(documents)} documents"}],
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
        "all_docs_irrelevant": False,
        "query_rewritten": True,
        "trace": [{"node": "corrective_rewrite", "decision": "query rewritten",
                    "reason": f"Docs irrelevant — rewrote: '{question}' → '{rewritten}'"}],
    }


# ─── Reranker ────────────────────────────────────────────────────────────────

def rerank_docs(state: Dict) -> Dict:
    """Rerank relevant documents using the injected reranker."""
    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])
    reranker_func = state.get("_reranker_func")

    if reranker_func and relevant_docs:
        reranked = reranker_func(question, relevant_docs)
        return {
            "relevant_docs": reranked,
            "trace": [{"node": "rerank", "decision": f"top {len(reranked)}", "reason": f"Reranked to top {len(reranked)} documents"}],
        }
    return {"trace": [{"node": "rerank", "decision": "skipped", "reason": "No reranker or no docs"}]}


def rerank_documents(question, docs):
    """Standalone reranker: embedding similarity with category balancing."""
    embeddings = get_embeddings()
    q_emb = embeddings.embed_query(question)

    scored = []
    for doc in docs:
        d_emb = embeddings.embed_query(doc.page_content)
        score = np.dot(q_emb, d_emb)
        scored.append((score, doc))
    scored.sort(reverse=True, key=lambda x: x[0])

    categories = set(d.metadata.get('category', '') for _, d in scored)
    if len(categories) > 1:
        selected, seen_cats = [], set()
        for score, doc in scored:
            cat = doc.metadata.get('category', '')
            if cat not in seen_cats:
                selected.append(doc)
                seen_cats.add(cat)
            if len(selected) >= RERANK_TOP_K:
                break
        if len(selected) < RERANK_TOP_K:
            for score, doc in scored:
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= RERANK_TOP_K:
                    break
        return selected[:RERANK_TOP_K]
    else:
        return [doc for _, doc in scored[:RERANK_TOP_K]]


# ─── Generator ───────────────────────────────────────────────────────────────

GENERATE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a KT Onboarding Assistant Agent.

Use ONLY information from the provided documents. Do NOT use general knowledge.
If the information is not in the documents, say so clearly.
Cite the document source when possible.

Context:
{context}

Question:
{question}"""
)


def generate_answer(state: Dict) -> Dict:
    """Generate answer from relevant documents."""
    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])
    generator_func = state.get("_generator_func")
    route = state.get("route", "")

    is_web = route == "web_search" or (
        relevant_docs and any(d.metadata.get("category") == "web_search" for d in relevant_docs)
    )

    if is_web and relevant_docs:
        context = "\n\n".join([
            f"[{d.metadata.get('filename', 'web')}]\n{d.page_content}" for d in relevant_docs
        ])
        llm = get_llm()
        web_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the user's question based on the web search results below.
Synthesize clearly and cite source URLs when possible.

Web search results:
{context}

Question: {question}"""
        )
        chain = web_prompt | llm
        result = chain.invoke({"context": context, "question": question})
        answer = result.content
        sources = [d.metadata.get("source", "") for d in relevant_docs if d.metadata.get("source")]
        if sources:
            answer += "\n\n---\n**🌐 Sources:** " + ", ".join(sources[:3])
    elif generator_func and relevant_docs:
        answer = generator_func(question, relevant_docs)
    elif not relevant_docs:
        answer = "I couldn't find relevant information to answer your question."
    else:
        answer = "Generator not available."

    logger.info("[Generate] Answer generated")
    return {
        "answer": answer,
        "trace": [{"node": "generate", "decision": "answer generated", "reason": "Generated answer from context"}],
    }


def generate_answer_standalone(question, docs):
    """Standalone generator: synthesise context into a grounded response with citation."""
    context = "\n\n".join([
        f"[{d.metadata.get('category', 'unknown')}/{d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
        for d in docs
    ])
    llm = get_llm()
    chain = GENERATE_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    primary_cat = docs[0].metadata.get('category', 'unknown') if docs else 'unknown'
    primary_file = docs[0].metadata.get('filename', 'unknown') if docs else 'unknown'
    return response.content + f"\n\n---\n**📚 Source:** {primary_cat}/{primary_file}"


def generate_direct(state: Dict) -> Dict:
    """Generate answer directly from LLM without retrieval."""
    question = state["question"]
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a friendly KT Onboarding Assistant. The user's message doesn't require document lookup.
Respond naturally and helpfully.

User: {question}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question})

    logger.info("[Direct LLM] Responded without retrieval")
    return {
        "answer": result.content,
        "trace": [{"node": "direct_llm", "decision": "direct response", "reason": "Answered without retrieval"}],
    }


def generate_fallback(state: Dict) -> Dict:
    """Fallback generation when all retrieved docs are irrelevant and web search fails."""
    question = state["question"]
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a KT Onboarding Assistant. No relevant documents were found for this question.

Politely inform the user and suggest they rephrase or check if the relevant documents have been uploaded.

User question: {question}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question})

    logger.info("[Fallback] Generated fallback response")
    return {
        "answer": result.content,
        "trace": [{"node": "fallback", "decision": "no relevant docs", "reason": "All docs irrelevant — fallback response"}],
    }


# ─── Self-RAG: Hallucination Grader ─────────────────────────────────────────

HALLUCINATION_PROMPT = PromptTemplate(
    input_variables=["documents", "answer"],
    template="""You are a hallucination grader. Given source documents and an LLM-generated answer, determine if the answer is grounded in the documents.

An answer is grounded if every factual claim can be traced back to the documents.

Respond with ONLY a JSON object: {{"grounded": true}} or {{"grounded": false}}

Source documents:
{documents}

Generated answer:
{answer}"""
)


def check_hallucination(state: Dict) -> Dict:
    """Self-RAG: Check if the generated answer is grounded in source documents."""
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
        "trace": [{"node": "hallucination_check",
                    "decision": "grounded" if grounded else "hallucination detected",
                    "reason": f"Answer {'is' if grounded else 'is NOT'} grounded (attempt {retry_count + 1})"}],
    }
    if not grounded:
        update["retry_count"] = retry_count + 1
    return update


# ─── Self-RAG: Answer Quality Grader ────────────────────────────────────────

ANSWER_GRADER_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an answer quality grader. Given a question and answer, determine if the answer actually addresses the question.

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
        "trace": [{"node": "answer_grader",
                    "decision": "useful" if useful else "not useful",
                    "reason": f"Answer {'addresses' if useful else 'does NOT address'} the question (attempt {retry_count + 1})"}],
    }


# ─── Web Search (Tavily) ────────────────────────────────────────────────────

def web_search(state: Dict) -> Dict:
    """Fallback: Search the web using Tavily when vectorstore fails."""
    question = state["question"]
    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_key:
        logger.warning("[Web Search] No TAVILY_API_KEY — skipping")
        return {
            "relevant_docs": [],
            "all_docs_irrelevant": True,
            "trace": [{"node": "web_search", "decision": "skipped", "reason": "TAVILY_API_KEY not configured"}],
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query=question, max_results=3)

        docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", ""), "category": "web_search", "filename": r.get("title", "")}
            )
            for r in results.get("results", [])
        ]

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
