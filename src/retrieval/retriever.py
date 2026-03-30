"""
Document retrieval logic with category detection
"""

from app.config import RETRIEVAL_K, CATEGORY_KEYWORDS


def detect_categories(question):
    """
    Detect relevant categories from question keywords.
    
    Args:
        question: User query string
        
    Returns:
        List of detected category names
    """
    question_lower = question.lower()
    detected_categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_categories.append(category)
    
    return detected_categories


def retrieve_documents(vectorstore, question):
    """
    Retrieve relevant documents from vector store.
    
    For multi-category queries (e.g. "compare avaya and dot"), retrieves
    docs from each category separately to ensure balanced coverage.
    
    Args:
        vectorstore: FAISS vector store instance
        question: User query string
        
    Returns:
        List of retrieved documents
    """
    detected_categories = detect_categories(question)
    
    if len(detected_categories) == 1:
        # Single category — filter directly
        docs = vectorstore.similarity_search(
            question, 
            k=RETRIEVAL_K,
            filter={"category": detected_categories[0]}
        )
        if not docs:
            docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    elif len(detected_categories) > 1:
        # Multiple categories — retrieve from each to ensure balanced coverage
        docs = []
        per_category_k = max(RETRIEVAL_K // len(detected_categories), 3)
        seen_contents = set()
        
        for category in detected_categories:
            cat_docs = vectorstore.similarity_search(
                question,
                k=per_category_k,
                filter={"category": category}
            )
            for doc in cat_docs:
                # Deduplicate by content
                content_key = doc.page_content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    docs.append(doc)
        
        # If filtered retrieval returned too few, supplement with unfiltered
        if len(docs) < RETRIEVAL_K:
            extra = vectorstore.similarity_search(question, k=RETRIEVAL_K)
            for doc in extra:
                content_key = doc.page_content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    docs.append(doc)
                if len(docs) >= RETRIEVAL_K:
                    break
    else:
        # No category detected — unfiltered semantic search
        docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    
    return docs
