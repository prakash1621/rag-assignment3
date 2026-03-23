"""
Streamlit UI for the Agentic RAG Pipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.ingestion import (
    scan_knowledge_base, extract_text_from_file,
    create_vector_store, save_vector_store, load_vector_store,
)
from src.ingestion.chunker import ParentChildChunker, SemanticChunker
from src.nodes import retrieve_documents, rerank_documents, generate_answer_standalone
from src.graph import run_agentic_rag

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Agentic RAG Pipeline", page_icon="🧠", layout="wide")

st.markdown("""
# 🧠 Agentic RAG Pipeline
**Adaptive + Corrective + Self-RAG**

- 🔀 Adaptive RAG — Routes queries to vectorstore, direct LLM, or web search
- ✅ Corrective RAG — Grades docs, rewrites query, falls back if irrelevant
- 🔄 Self-RAG — Checks for hallucinations, retries poor answers
""")

# ─── Session state ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vector_store()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy", ["parent_child", "semantic"],
    help="Parent-Child: large parents + small children. Semantic: embedding-based boundaries."
)
show_trace = st.sidebar.checkbox("Show Agent Trace", value=True)


# ─── Build Knowledge Base ────────────────────────────────────────────────────

def build_knowledge_base():
    with st.spinner(f"Building knowledge base with {chunking_strategy} chunking..."):
        categories = scan_knowledge_base()
        if not categories:
            st.error("No documents found in knowledge-base/")
            return

        all_chunks, all_metas = [], []

        if chunking_strategy == "parent_child":
            chunker = ParentChildChunker()
        else:
            # Semantic chunker needs an embedder function
            import boto3, json, numpy as np
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            model_id = os.environ.get("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
            client = boto3.client("bedrock-runtime", region_name=region)

            def embedder(text):
                body = json.dumps({"inputText": text})
                resp = client.invoke_model(modelId=model_id, body=body,
                                           contentType="application/json", accept="application/json")
                return np.array(json.loads(resp["body"].read())["embedding"])

            chunker = SemanticChunker(embedder=embedder)

        for category, files in categories.items():
            for file_path in files:
                text, _ = extract_text_from_file(file_path)
                if text.strip():
                    metadata = {
                        "source": file_path,
                        "category": category,
                        "filename": os.path.basename(file_path),
                    }
                    chunks, metas = chunker.chunk(text, metadata)
                    all_chunks.extend(chunks)
                    all_metas.extend(metas)

        if all_chunks:
            vectorstore = create_vector_store(all_chunks, all_metas)
            save_vector_store(vectorstore)
            st.session_state.vectorstore = vectorstore
            st.success(f"Indexed {len(all_chunks)} chunks from {sum(len(f) for f in categories.values())} files")
        else:
            st.error("No text extracted from documents")


col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔨 Rebuild KB"):
        build_knowledge_base()
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Status
if st.session_state.vectorstore:
    st.sidebar.success("✅ Vector store loaded")
else:
    st.sidebar.warning("⚠️ No vector store — click Rebuild KB")

# ─── Chat Interface ──────────────────────────────────────────────────────────

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "trace" in msg and show_trace:
            with st.expander("📍 Agent Trace"):
                for step in msg["trace"]:
                    icon = {"router": "🔀", "rewriter": "✏️", "retrieve": "📥",
                            "grader": "✅", "rerank": "📊", "generate": "⚡",
                            "direct_llm": "💬", "fallback": "⚠️",
                            "hallucination_check": "🔍", "answer_grader": "🎯",
                            "corrective_rewrite": "🔄", "web_search": "🌐"
                            }.get(step["node"], "⚙️")
                    st.markdown(f"{icon} **{step['node']}** → {step['decision']}")
                    st.caption(step.get("reason", ""))

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build chat history for rewriter
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            result = run_agentic_rag(
                question=prompt,
                vectorstore=st.session_state.vectorstore,
                retriever_func=retrieve_documents,
                reranker_func=rerank_documents,
                generator_func=generate_answer_standalone,
                max_retries=3,
                chat_history=chat_history if chat_history else None,
            )

        st.markdown(result["answer"])

        if show_trace and result.get("trace"):
            with st.expander("📍 Agent Trace"):
                for step in result["trace"]:
                    icon = {"router": "🔀", "rewriter": "✏️", "retrieve": "📥",
                            "grader": "✅", "rerank": "📊", "generate": "⚡",
                            "direct_llm": "💬", "fallback": "⚠️",
                            "hallucination_check": "🔍", "answer_grader": "🎯",
                            "corrective_rewrite": "🔄", "web_search": "🌐"
                            }.get(step["node"], "⚙️")
                    st.markdown(f"{icon} **{step['node']}** → {step['decision']}")
                    st.caption(step.get("reason", ""))

            st.caption(f"🛤️ Route: {result.get('route')} | 🔁 Retries: {result.get('retry_count', 0)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": result.get("trace", []),
    })
