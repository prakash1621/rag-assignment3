"""
Cache Manager — Semantic Cache Only

Manages the semantic cache bonus feature.
Checks for similar past queries before running the full pipeline.
"""

from typing import Optional, Dict, Tuple
import logging
from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, embedder=None, config: Dict = None):
        self.config = config or {}
        semantic_config = self.config.get('semantic', {})

        if semantic_config.get('enabled', True) and embedder:
            self.semantic_cache = SemanticCache(
                embedder=embedder,
                similarity_threshold=semantic_config.get('similarity_threshold', 0.95),
                ttl_seconds=semantic_config.get('ttl_seconds', 3600),
                max_cache_size=semantic_config.get('max_cache_size', 1000),
            )
            logger.info("Semantic cache initialized")
        else:
            self.semantic_cache = None
            logger.info("Semantic cache disabled")

    def get_response(self, query: str) -> Optional[Tuple[str, str]]:
        """Check semantic cache for a similar past query."""
        try:
            if self.semantic_cache:
                result = self.semantic_cache.get(query)
                if result:
                    logger.info(f"✓ Semantic cache HIT (similarity: {result['similarity']:.3f})")
                    return result['response'], "semantic"
            return None
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None

    def cache_response(self, query: str, response: str) -> None:
        """Cache a query-response pair in semantic cache."""
        try:
            if self.semantic_cache:
                self.semantic_cache.set(query, response)
        except Exception as e:
            logger.error(f"Cache store error: {e}")

    def clear_all(self) -> None:
        if self.semantic_cache:
            self.semantic_cache.clear()
            logger.info("Semantic cache cleared")

    def get_all_stats(self) -> Dict:
        stats = {}
        if self.semantic_cache:
            stats['semantic'] = self.semantic_cache.get_stats()
        total_hits = sum(s['hits'] for s in stats.values())
        total_misses = sum(s['misses'] for s in stats.values())
        total = total_hits + total_misses
        stats['overall'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total,
            'hit_rate': total_hits / total if total > 0 else 0,
        }
        return stats
