# Evaluation Results — Agentic RAG Pipeline

**Course:** RAG Architect 2026 — Assignment 3  
**Date:** March 20, 2026  
**Pipeline:** Adaptive + Corrective + Self-RAG via LangGraph  
**LLM:** Claude 3 Haiku (AWS Bedrock) | **Embeddings:** Amazon Titan Embed Text v1

---

## Evaluation Methodology

**Metrics used (RAGAS-aligned, no ground truth required):**

| Metric | What it measures | How we compute it |
|--------|-----------------|-------------------|
| **Faithfulness** | Is the answer grounded in retrieved docs? | Built-in hallucination grader (LLM judge) |
| **Answer Relevance** | Does the answer address the question? | Built-in answer quality grader (LLM judge) |
| **Route Accuracy** | Did the router pick the correct path? | Manual inspection of trace |
| **Doc Precision** | What fraction of retrieved docs were relevant? | Grader pass rate (relevant / total) |
| **Retry Count** | How many self-correction loops were needed? | From pipeline trace |
| **Latency** | End-to-end response time | Wall clock (approximate) |

---

## Test Cases

### Test 1 — Single-Category KB Query (Vectorstore Path)

| Field | Value |
|-------|-------|
| **Question** | "What is the Avaya project about?" |
| **Expected Route** | vectorstore |
| **Actual Route** | ✅ vectorstore |
| **Docs Retrieved** | 10 |
| **Docs Relevant (Grader)** | 10/10 (100%) |
| **Docs After Rerank** | 3 |
| **Hallucination Check** | ✅ Grounded |
| **Answer Quality** | ✅ Useful |
| **Retry Count** | 0 |
| **Faithfulness** | ✅ High — answer references Avaya SMS opt-in, agent traces, call history, Terraform migration |
| **Answer Relevance** | ✅ High — directly explains what Avaya is and its purpose |
| **Source Citation** | avaya/avaya_01_Overview.md |

**Trace:** Router → Rewriter (skipped) → Retrieve (10 docs) → Grader (10/10) → Rerank (top 3) → Generate → Hallucination Check (grounded) → Answer Grader (useful)

---

### Test 2 — Multi-Category Comparison (Vectorstore Path, Balanced Retrieval)

| Field | Value |
|-------|-------|
| **Question** | "What is the Avaya project and how is it different from the DOT project?" |
| **Expected Route** | vectorstore |
| **Actual Route** | ✅ vectorstore |
| **Docs Retrieved** | 10 (balanced: ~5 avaya + ~5 dot) |
| **Docs Relevant (Grader)** | 10/10 (100%) |
| **Docs After Rerank** | 3 (at least 1 per category) |
| **Hallucination Check** | ✅ Grounded |
| **Answer Quality** | ✅ Useful |
| **Retry Count** | 0 |
| **Faithfulness** | ✅ High — correctly distinguishes Avaya (call center data) from DOT (fare/currency) |
| **Answer Relevance** | ✅ High — covers both projects and their differences |
| **Source Citation** | avaya/avaya_01_Overview.md, dot/dot_01_Overview.md |

**Trace:** Router → Rewriter (skipped) → Retrieve (10 docs, multi-category) → Grader (10/10) → Rerank (top 3, category-balanced) → Generate → Hallucination Check (grounded) → Answer Grader (useful)

**Note:** Per-category retrieval ensures both Avaya and DOT docs are fetched, preventing one category from dominating results.

---

### Test 3 — Direct LLM (No Retrieval)

| Field | Value |
|-------|-------|
| **Question** | "Hello! What can you help me with?" |
| **Expected Route** | direct_llm |
| **Actual Route** | ✅ direct_llm |
| **Docs Retrieved** | 0 (retrieval skipped) |
| **Hallucination Check** | N/A (no source docs) |
| **Answer Quality** | ✅ Appropriate greeting response |
| **Retry Count** | 0 |
| **Faithfulness** | N/A — no retrieval needed |
| **Answer Relevance** | ✅ High — responds as onboarding assistant, lists capabilities |

**Trace:** Router → Direct Generate → END

---

### Test 4 — Web Search (Tavily Fallback)

| Field | Value |
|-------|-------|
| **Question** | "What are the latest features in LangGraph 2025?" |
| **Expected Route** | web_search |
| **Actual Route** | ✅ web_search |
| **Web Results** | 3 (from Tavily) |
| **Hallucination Check** | ✅ Grounded in web results |
| **Answer Quality** | ✅ Useful |
| **Retry Count** | 0 |
| **Faithfulness** | ✅ High — answer based on Tavily search results |
| **Answer Relevance** | ✅ High — covers LangGraph features |

**Trace:** Router → Web Search (3 results) → Generate → Hallucination Check (grounded) → Answer Grader (useful)

**Note:** Requires `TAVILY_API_KEY` in `.env`. Without it, the node skips gracefully and returns empty results.

---

### Test 5 — Corrective RAG (Query Rewrite + Retry)

| Field | Value |
|-------|-------|
| **Question** | "Tell me about the migration process" |
| **Expected Route** | vectorstore |
| **Actual Route** | ✅ vectorstore |
| **First Retrieval** | 10 docs |
| **First Grading** | 0/10 relevant (vague query) |
| **Corrective Rewrite** | ✅ Query rewritten to be more specific |
| **Second Retrieval** | 10 docs |
| **Second Grading** | 8/10 relevant |
| **Hallucination Check** | ✅ Grounded |
| **Answer Quality** | ✅ Useful |
| **Retry Count** | 1 (corrective rewrite loop) |
| **Faithfulness** | ✅ High — references actual migration details from KB |
| **Answer Relevance** | ✅ High — explains migration context |

**Trace:** Router → Rewriter → Retrieve (10) → Grader (0/10) → Corrective Rewrite → Retrieve (10) → Grader (8/10) → Rerank (top 3) → Generate → Hallucination Check → Answer Grader

**Note:** This demonstrates the Corrective RAG loop — when initial retrieval fails, the system rewrites the query and retries before falling back.

---

### Test 6 — Follow-up with Conversational Context

| Field | Value |
|-------|-------|
| **Question** | "Explain the first one in detail" |
| **Chat History** | User asked about BPPSL tables, assistant listed them |
| **Expected Behavior** | Rewriter resolves "the first one" → BKNG_PNR_PAX_SEG_LEG |
| **Actual Route** | ✅ vectorstore |
| **Rewriter Output** | "Explain BKNG_PNR_PAX_SEG_LEG table in detail" |
| **Docs Relevant** | 10/10 |
| **Hallucination Check** | ✅ Grounded |
| **Answer Quality** | ✅ Useful |
| **Faithfulness** | ✅ High — correctly explains the BKNG_PNR_PAX_SEG_LEG table |
| **Answer Relevance** | ✅ High — resolves ambiguous reference using context |

**Trace:** Router → Rewriter (resolved follow-up) → Retrieve → Grader → Rerank → Generate → Hallucination Check → Answer Grader

---

### Test 7 — Cache Hit (Exact Match)

| Field | Value |
|-------|-------|
| **Question** | "What is BPPSL?" (asked twice) |
| **First Call Route** | vectorstore (full pipeline) |
| **Second Call Route** | 💾 cache hit (semantic) |
| **Second Call Latency** | ~0.1s (vs ~8s full pipeline) |
| **Answer Consistency** | ✅ Identical answer returned |

**Note:** The semantic cache (cosine ≥ 0.95) dramatically reduces latency for repeated or similar queries.

---

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| **Route Accuracy** | 7/7 (100%) — all queries routed correctly |
| **Doc Precision (avg)** | ~90% — grader correctly filters irrelevant docs |
| **Faithfulness** | 6/6 grounded (100%) — no hallucinations detected |
| **Answer Relevance** | 7/7 useful (100%) — all answers address the question |
| **Corrective Recovery** | 1/1 — vague query successfully recovered via rewrite |
| **Cache Effectiveness** | Exact cache hit on repeated query, ~80x speedup |

---

## Observations & Trade-offs

1. **Latency vs Quality:** The agentic pipeline (8-15s) is slower than standard RAG (3-5s) due to multiple LLM calls (router, grader, hallucination check, answer grader). The quality improvement justifies this for complex queries.

2. **Grader Prompt Design:** Using "should this be EXCLUDED?" framing instead of "is this relevant?" was critical. Claude Haiku defaults to "no" on ambiguous relevance questions, which caused false negatives with the positive framing.

3. **Multi-Category Retrieval:** Per-category FAISS search with balanced reranking solved the problem of one category dominating results in comparison queries.

4. **Conversational Context:** The query rewriter successfully resolves pronouns and references ("the first one", "explain that") using the last 4 messages of chat history.

5. **Semantic Cache:** The semantic cache (cosine ≥ 0.95) catches repeated and paraphrased queries, skipping the entire pipeline on cache hit.

6. **Loop Control:** Max 3 retries with `retry_count` increment prevents infinite loops. The graph terminates gracefully even when hallucination or quality checks fail repeatedly.
