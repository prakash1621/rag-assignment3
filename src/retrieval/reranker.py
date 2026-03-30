"""
Document reranking logic using embedding similarity
"""

import numpy as np
from app.embedding import get_embeddings
from app.config import RERANK_TOP_K


def rerank_documents(question, docs):
    """
    Rerank documents based on embedding similarity.
    
    For multi-category doc sets, ensures at least one doc per category
    makes it into the final selection for balanced coverage.
    
    Args:
        question: User query string
        docs: List of retrieved documents
        
    Returns:
        List of reranked documents (top K)
    """
    embeddings = get_embeddings()
    question_embedding = embeddings.embed_query(question)
    
    scored_docs = []
    for doc in docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        score = np.dot(question_embedding, doc_embedding)
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # Check how many distinct categories are in the docs
    categories = set(d.metadata.get('category', '') for _, d in scored_docs)
    
    if len(categories) > 1:
        # Multi-category: pick top doc from each category first, then fill remaining
        selected = []
        seen_categories = set()
        
        # First pass: best doc per category
        for score, doc in scored_docs:
            cat = doc.metadata.get('category', '')
            if cat not in seen_categories:
                selected.append(doc)
                seen_categories.add(cat)
            if len(selected) >= RERANK_TOP_K:
                break
        
        # Second pass: fill remaining slots by score
        if len(selected) < RERANK_TOP_K:
            for score, doc in scored_docs:
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= RERANK_TOP_K:
                    break
        
        return selected[:RERANK_TOP_K]
    else:
        # Single category: standard top-K by score
        return [doc for _, doc in scored_docs[:RERANK_TOP_K]]
