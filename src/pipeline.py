"""
Main RAG Pipeline — Agentic RAG with Semantic Cache
"""

import logging
from typing import List, Dict, Tuple
from src.utils import load_config, get_embedder, setup_logger
from src.chunking import ParentChildChunker, SemanticChunker
from src.caching import CacheManager

logger = setup_logger("rag_pipeline")


class RAGPipeline:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        logger.info("Configuration loaded")

        self.embedder = get_embedder(self.config)
        logger.info("Embedder initialized")

        # Chunking strategies
        pc_config = self.config['chunking']['parent_child']
        self.parent_child_chunker = ParentChildChunker(
            parent_size=pc_config['parent_size'],
            parent_overlap=pc_config['parent_overlap'],
            child_size=pc_config['child_size'],
            child_overlap=pc_config['child_overlap']
        )

        sem_config = self.config['chunking']['semantic']
        self.semantic_chunker = SemanticChunker(
            embedder=self.embedder,
            buffer_size=sem_config['buffer_size'],
            breakpoint_threshold_type=sem_config['breakpoint_threshold_type'],
            breakpoint_threshold_amount=sem_config['breakpoint_threshold_amount']
        )
        logger.info("Chunking strategies initialized")

        # Semantic cache (bonus feature)
        self.cache_manager = CacheManager(
            embedder=self.embedder,
            config=self.config['caching']
        )
        logger.info("Semantic cache initialized")

    def chunk_document(self, text: str, metadata: Dict,
                       strategy: str = "parent_child") -> Tuple[List[str], List[Dict]]:
        if strategy == "parent_child":
            return self.parent_child_chunker.chunk(text, metadata)
        elif strategy == "semantic":
            return self.semantic_chunker.chunk(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def agentic_query(self, question: str, vectorstore=None,
                      retriever_func=None, reranker_func=None,
                      generator_func=None, chat_history=None) -> Dict:
        """
        Process query through the agentic RAG graph (Adaptive + Corrective + Self-RAG).
        Checks semantic cache first, then falls through to the LangGraph pipeline.
        """
        from src.agentic.graph import run_agentic_rag

        logger.info(f"[Agentic] Processing query: {question[:100]}...")

        # Check semantic cache for similar past query
        cached_response = self.cache_manager.get_response(question)
        if cached_response:
            response, cache_tier = cached_response
            logger.info(f"[Agentic] Semantic cache HIT")
            return {
                'answer': response,
                'cache_tier': cache_tier,
                'from_cache': True,
                'trace': [{"node": "cache", "decision": f"hit ({cache_tier})", "reason": f"Served from {cache_tier} cache — skipped entire pipeline"}],
            }

        # Full agentic pipeline — cache miss
        result = run_agentic_rag(
            question=question,
            vectorstore=vectorstore,
            retriever_func=retriever_func,
            reranker_func=reranker_func,
            generator_func=generator_func,
            max_retries=2,
            chat_history=chat_history,
        )

        # Cache the response for future similar queries
        if result.get("answer"):
            self.cache_manager.cache_response(question, result["answer"])

        result['cache_tier'] = None
        return result

    def get_cache_stats(self) -> Dict:
        return self.cache_manager.get_all_stats()

    def clear_caches(self):
        self.cache_manager.clear_all()
