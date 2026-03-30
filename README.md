# 🧠 Agentic RAG Pipeline — Adaptive + Corrective + Self-RAG

**Assignment 3 — RAG Architect 2026**

A production-grade agentic RAG system that goes beyond basic retrieve-and-generate. The pipeline intelligently routes queries, grades retrieved documents, rewrites failed queries, falls back to web search, and self-checks answers for hallucinations and quality — all orchestrated as a LangGraph state machine.

**Stack:** LangGraph · AWS Bedrock (Claude 3 Haiku + Titan Embeddings) · FAISS · Tavily · Streamlit

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  🔴 Semantic Cache (cosine ≥ 0.95)                               │
│     Similar past query? → Return cached response (skip pipeline) │
└──────────────────────────────────────────────────────────────────┘
    │ Cache miss
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  🔵 ADAPTIVE RAG: Query Router (LLM-powered)                    │
│     Classifies → vectorstore | direct_llm | web_search          │
└──────────────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ 🟣 Direct │    │ 🔵 Vectorstore│    │ 🌐 Web Search │
│   LLM     │    │   Path       │    │   (Tavily)    │
│  (no RAG) │    │              │    │               │
└──────────┘    └──────────────┘    └──────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
   END          ┌──────────────┐        Generate
                │ ✏️ Rewriter   │            │
                │ (chat context)│            ▼
                └──────────────┘      Self-RAG checks
                        │                    │
                        ▼                   END
                ┌──────────────┐
                │ 📥 Retrieve   │
                │  (FAISS, k=10)│
                └──────────────┘
                        │
                        ▼
                ┌──────────────────────────────────────┐
                │ 🟢 CORRECTIVE RAG: Grade Documents    │
                │     LLM grades each doc (relevant?)   │
                └──────────────────────────────────────┘
                   │                          │
                   ▼                          ▼
              Has relevant docs         All irrelevant
                   │                          │
                   ▼                     ┌────┴────┐
              📊 Rerank                  │ 1st try? │
                   │                     ├─ Yes ────→ 🔄 Corrective Rewrite → Retrieve (retry)
                   ▼                     └─ No ─────→ 🌐 Web Search fallback
              ⚡ Generate
                   │
                   ▼
                ┌──────────────────────────────────────┐
                │ 🟡 SELF-RAG: Hallucination Check      │
                │     Is answer grounded in sources?     │
                │     ↓ No → retry (max 3)               │
                └──────────────────────────────────────┘
                        │
                        ▼
                ┌──────────────────────────────────────┐
                │ 🟡 SELF-RAG: Answer Quality Grader    │
                │     Does answer address the question?  │
                │     ↓ No → retry (max 3)               │
                └──────────────────────────────────────┘
                        │
                        ▼
                      END
```

---

## Framework Choice: Why LangGraph

We evaluated three options:

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **LangGraph** | Native state management, conditional edges, built-in retry/recursion limits, typed state with reducers | Slightly more boilerplate than raw Python | ✅ **Chosen** |
| **LlamaIndex** | Good for simple RAG, built-in retrievers | Less flexible for custom agentic loops, harder to add conditional branching | ❌ |
| **Custom Python** | Full control, no dependencies | Must build state management, loop control, and graph execution from scratch | ❌ |

**Why LangGraph won:**
1. The pipeline has 12 nodes with conditional branching — LangGraph's `StateGraph` + `add_conditional_edges` maps directly to this.
2. The `Annotated` typed state with reducers (`replace_value`, `merge_list`) prevents state corruption when nodes return partial updates.
3. Built-in `recursion_limit` provides a safety net against infinite loops without manual tracking.
4. The graph compiles to a runnable that can be invoked with a single `graph.invoke()` call.

---

## Chunking Strategy: Why Parent-Child + Semantic

Two chunking strategies are available, selectable from the sidebar:

**Parent-Child Chunking (default):**
- Parent chunks: 3000 chars with 500 overlap — provides broad context to the LLM
- Child chunks: 500 chars with 100 overlap — used for precise embedding similarity search
- Retrieval searches children, but the parent chunk is passed to the LLM for generation
- Best for: structured documents with clear sections (our KB has project docs with headers)

**Semantic Chunking:**
- Uses embedding cosine similarity to detect natural topic boundaries
- Buffer size of 1 sentence, percentile-based breakpoint at 95th percentile
- Produces variable-length chunks that respect semantic coherence
- Best for: unstructured text where fixed-size splitting would break mid-concept

**Why both?** Different document types benefit from different strategies. Project overview docs with clear sections work well with parent-child. Dense technical docs with flowing prose work better with semantic chunking. The user can switch strategies and rebuild the KB to compare.

---

## Design Trade-offs

### 1. Quality vs Latency
The agentic pipeline makes 5-7 LLM calls per query (router, grader ×N, generator, hallucination check, answer grader) vs 1 call for standard RAG. This increases latency from ~3s to ~10-15s but catches bad retrievals and hallucinations.

**Mitigation:** Semantic caching. Repeated/similar queries hit cache in <100ms, amortizing the cost over time.

### 2. Grader Prompt: Exclusion vs Inclusion Framing
Claude Haiku defaults to "no" on ambiguous yes/no questions. When we asked "is this document relevant?", the grader marked most docs as irrelevant (false negatives). Flipping to "should this document be EXCLUDED?" fixed this — now the default "no" means "don't exclude" = relevant.

### 3. Multi-Category Retrieval
A single FAISS similarity search for "compare Avaya and DOT" returns docs from whichever category has stronger embedding overlap, starving the other. We split into per-category retrieval (k/N per category) with category-balanced reranking to ensure coverage.

### 4. Conversational Context
Follow-up questions like "explain the first one" fail without context. The conversational query rewriter uses the last 4 chat messages to resolve references before retrieval. This adds one LLM call but prevents complete retrieval failures on follow-ups.

### 5. Web Search as Safety Net
When the corrective rewrite + retry still fails, Tavily web search provides a last-resort answer. This prevents the system from returning "I don't know" on queries that are slightly outside the KB but answerable from the web.

### 6. Loop Control
Max 3 retries on hallucination/quality failures. The `retry_count` field in state increments on each failure, and conditional edges check it before looping. Combined with LangGraph's `recursion_limit: 50`, the system always terminates.

---

## Project Structure

```
Sathvik/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── config.yaml                     # Pipeline configuration
├── main.py                         # Streamlit UI entry point
│
├── src/                            # Source code
│   ├── graph.py                    # ★ Full pipeline assembly (assignment entry point)
│   ├── ingestion/                  # ★ Loader, chunker, indexer
│   │   ├── loader.py               #   Document scanner + text extractor
│   │   ├── chunker.py              #   Chunking strategies (parent-child, semantic)
│   │   └── indexer.py              #   FAISS vector store create/save/load
│   ├── nodes/                      # ★ All agentic decision nodes
│   │   └── __init__.py             #   Router, grader, rewriter, generator, fallback
│   ├── agentic/                    # Core agentic RAG implementation
│   │   ├── graph.py                #   LangGraph state machine (12 nodes)
│   │   ├── nodes.py                #   Node implementations
│   │   └── state.py                #   Typed state with Annotated reducers
│   ├── chunking/                   # Document chunking strategies
│   │   ├── parent_child.py         #   Parent-child chunking
│   │   └── semantic_chunker.py     #   Semantic chunking
│   ├── caching/                    # Semantic cache (bonus feature)
│   │   ├── semantic_cache.py       #   Embedding similarity cache (≥0.95)
│   │   └── cache_manager.py        #   Cache interface
│   ├── retrieval/                  # Document retrieval
│   │   ├── retriever.py            #   FAISS retrieval with category detection
│   │   └── reranker.py             #   Embedding-based reranking
│   ├── generation/                 # LLM generation
│   │   └── generator.py            #   AWS Bedrock (Claude 3 Haiku)
│   ├── utils/                      # Utilities
│   │   ├── config_loader.py        #   YAML config loader
│   │   ├── embeddings.py           #   Titan embedding wrapper
│   │   └── logger.py               #   Logging setup
│   └── pipeline.py                 # Pipeline orchestration (agentic)
│
├── app/                            # Shared utilities (config, ingestion, embeddings)
│   ├── config.py                   #   Constants and category keywords
│   ├── ingestion.py                #   Document loader (PDF, DOCX, MD)
│   └── embedding.py                #   Vector store create/save/load
│
├── notebooks/
│   └── demo.ipynb                  # End-to-end demo (all 3 branches)
│
├── evaluation/
│   └── results.md                  # Test cases + metrics (7 test cases)
│
├── knowledge-base/                 # Source documents by category
│   ├── avaya/                      # Avaya call center module
│   ├── bppsl/                      # Booking/fare proration
│   └── dot/                        # DOT fare/currency
│
├── vector_store/                   # FAISS index (generated)
└── cache/                          # Semantic cache DB (generated)
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- AWS account with Bedrock access (Claude 3 Haiku + Titan Embeddings)
- AWS credentials configured (`~/.aws/credentials` or environment variables)

### Installation

```bash
# Clone and install
git clone <your-repo-url>
cd Sathvik
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials and optionally TAVILY_API_KEY
```

### Run

```bash
streamlit run main.py
```

1. Open `http://localhost:8501`
2. Click "Rebuild Knowledge Base" in the sidebar
3. Start asking questions — the agentic pipeline handles routing, grading, and self-reflection automatically

### Demo Notebook

```bash
cd notebooks
jupyter notebook demo.ipynb
```

---

## Bonus Features Implemented

| Feature | Points | Implementation |
|---------|--------|---------------|
| Advanced chunking (parent-child + semantic) | +10 | `src/chunking/` — two strategies, switchable from sidebar |
| Semantic cache (cosine ≥ 0.95) | +10 | `src/caching/semantic_cache.py` — skips pipeline on similar past queries |
| 3-tier router | +10 | `src/agentic/nodes.py` — LLM classifies as direct_llm / vectorstore / web_search |

---

## Configuration

All settings in `config.yaml`:

```yaml
# Key settings
chunking:
  parent_child: { parent_size: 3000, child_size: 500 }
  semantic: { breakpoint_threshold_amount: 95 }

caching:
  semantic: { similarity_threshold: 0.95, ttl_seconds: 3600 }

agentic:
  max_retries: 2
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | AWS Bedrock — Claude 3 Haiku |
| Embeddings | AWS Bedrock — Amazon Titan Embed Text v1 |
| Vector Store | FAISS (faiss-cpu) |
| Agentic Framework | LangGraph |
| Web Search | Tavily |
| Caching | Semantic cache (in-memory, cosine ≥ 0.95) |
| UI | Streamlit |
| Orchestration | LangChain (minimal — prompts, Bedrock client) |

---

## References

- [Self-RAG — Asai et al., 2023](https://arxiv.org/abs/2310.11511)
- [Corrective RAG — Yan et al., 2024](https://arxiv.org/abs/2401.15884)
- [Adaptive RAG — Jeong et al., 2024](https://arxiv.org/abs/2403.14403)
- [LangGraph Agentic RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
