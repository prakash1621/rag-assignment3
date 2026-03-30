"""
Main entry point - Agentic RAG Pipeline
Combines Adaptive RAG, Corrective RAG, and Self-RAG via LangGraph
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import shutil
from pathlib import Path

from app.config import VECTOR_STORE_PATH
from app.ingestion import scan_knowledge_base, extract_text_from_file
from app.embedding import create_vector_store, save_vector_store, load_vector_store

from src.retrieval import retrieve_documents, rerank_documents
from src.generation import generate_answer
from src.pipeline import RAGPipeline
from src.utils import setup_logger

logger = setup_logger("main")

# Page config
st.set_page_config(
    page_title="Agentic RAG Pipeline",
    page_icon="🧠",
    layout="wide"
)

# Header
st.markdown("""
# 🧠 Agentic RAG Pipeline
## Adaptive + Corrective + Self-Reflective RAG

- 🔀 **Adaptive RAG** — Routes queries to vectorstore, direct LLM, or web search
- ✅ **Corrective RAG** — Grades retrieved docs, rewrites query, falls back if irrelevant
- 🔄 **Self-RAG** — Checks for hallucinations, retries poor answers
""")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline()
        logger.info("Pipeline initialized")
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.session_state.pipeline = None

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    ["parent_child", "semantic"],
    help="Parent-Child: Large parents + small children. Semantic: Embedding-based boundaries."
)

show_trace = st.sidebar.checkbox("Show Agent Trace", value=True)


# Build knowledge base
def build_knowledge_base():
    with st.spinner(f"Building knowledge base with {chunking_strategy} chunking..."):
        categories = scan_knowledge_base()
        if not categories:
            st.error("No documents found")
            return

        all_chunks = []
        all_metadatas = []

        for category, files in categories.items():
            for file_path in files:
                text, _ = extract_text_from_file(file_path)

                if text.strip():
                    metadata = {
                        "source": file_path,
                        "category": category,
                        "filename": os.path.basename(file_path)
                    }

                    chunks, metadatas = st.session_state.pipeline.chunk_document(
                        text, metadata, strategy=chunking_strategy
                    )

                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)

        if all_chunks:
            vectorstore = create_vector_store(all_chunks, all_metadatas)
            save_vector_store(vectorstore)
            st.session_state.vectorstore = vectorstore
            st.session_state.pipeline.clear_caches()

            st.success(f"✅ Knowledge base built with {len(all_chunks)} chunks "
                      f"from {len(categories)} categories using {chunking_strategy} strategy")
        else:
            st.warning("No content found to process")


# Load existing vector store
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vector_store()
    if st.session_state.vectorstore:
        st.info("✅ Loaded existing knowledge base")

# Sidebar buttons
if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    build_knowledge_base()

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.traces = []
    if st.session_state.pipeline:
        st.session_state.pipeline.clear_caches()
    st.sidebar.success("Knowledge base cleared")
    st.rerun()


# ─── Trace Display Helper ───────────────────────────────────────────────────

NODE_ICONS = {
    "cache": "💾",
    "router": "🔀",
    "rewriter": "✏️",
    "retrieve": "📥",
    "grader": "✅",
    "rerank": "📊",
    "generate": "⚡",
    "direct_llm": "💬",
    "fallback": "⚠️",
    "hallucination_check": "🔍",
    "answer_grader": "🎯",
    "corrective_rewrite": "🔄",
    "web_search": "🌐",
}


def render_trace(trace):
    if not trace:
        return

    st.markdown("---")
    st.markdown("**🧠 Agent Decision Trace:**")

    cols = st.columns(min(len(trace), 5))
    for i, step in enumerate(trace):
        col_idx = i % 5
        with cols[col_idx]:
            icon = NODE_ICONS.get(step["node"], "⚙️")
            st.markdown(f"{icon} **{step['node']}**")
            st.caption(f"{step['decision']}")

    with st.expander("Full trace details"):
        for i, step in enumerate(trace):
            icon = NODE_ICONS.get(step["node"], "⚙️")
            st.markdown(f"{i+1}. {icon} **{step['node']}** → _{step['decision']}_")
            st.caption(step.get("reason", ""))


# ─── Process Question ────────────────────────────────────────────────────────

def process_question(question):
    if st.session_state.pipeline is None:
        st.warning("Pipeline not ready")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    result = st.session_state.pipeline.agentic_query(
        question=question,
        vectorstore=st.session_state.vectorstore,
        retriever_func=retrieve_documents,
        reranker_func=rerank_documents,
        generator_func=generate_answer,
        chat_history=st.session_state.messages,
    )
    answer = result['answer']
    trace = result.get('trace', [])

    if result.get('from_cache'):
        cache_tier = result.get('cache_tier', 'unknown')
        answer = f"💾 *[Cached — semantic match]*\n\n{answer}"

    st.session_state.traces.append(trace)
    st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── Chat Display ────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and show_trace:
            trace_idx = i // 2
            if trace_idx < len(st.session_state.traces):
                trace = st.session_state.traces[trace_idx]
                if trace:
                    render_trace(trace)

# User input
if st.session_state.pipeline is not None:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        process_question(question)
        st.rerun()
else:
    st.error("Pipeline failed to initialize. Check your AWS credentials and config.")

# Footer
st.markdown("---")
st.caption("🧠 Agentic RAG Pipeline | Adaptive + Corrective + Self-RAG | LangGraph-powered")
