"""
Semantic Cache — Bonus Feature (+10 pts)

Compares query embeddings using cosine similarity.
If similarity exceeds threshold (≥0.95), returns cached response.
Skips entire pipeline on cache hit.
"""

import time
import numpy as np
from typing import Optional, Dict
from collections import OrderedDict
from .base_cache import BaseSemanticCache, CacheOperationError


class SemanticCache(BaseSemanticCache):
    def __init__(self, embedder, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600, max_cache_size: int = 1000):
        super().__init__(embedder, similarity_threshold, ttl_seconds, max_cache_size)
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.embeddings: Dict[str, np.ndarray] = {}

    def _find_similar_query(self, query_embedding: np.ndarray):
        best_match = None
        best_similarity = 0.0
        current_time = time.time()
        expired_keys = []

        for cache_key, entry in self.cache.items():
            if current_time - entry['timestamp'] >= self.ttl_seconds:
                expired_keys.append(cache_key)
                continue
            cached_embedding = self.embeddings[cache_key]
            similarity = self.cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cache_key

        for key in expired_keys:
            del self.cache[key]
            del self.embeddings[key]

        if best_match and best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        return None

    def get(self, query: str) -> Optional[Dict]:
        try:
            query_embedding = self.embed_query(query)
            if query_embedding is None:
                self._record_miss()
                return None

            match = self._find_similar_query(query_embedding)
            if match:
                cache_key, similarity = match
                entry = self.cache[cache_key]
                self.cache.move_to_end(cache_key)
                self._record_hit()
                return {
                    'response': entry['response'],
                    'similarity': similarity,
                    'original_query': entry['query']
                }

            self._record_miss()
            return None
        except Exception as e:
            self._logger.error(f"Cache get failed: {e}")
            self._record_miss()
            return None

    def set(self, query: str, response: str) -> None:
        try:
            query_embedding = self.embed_query(query)
            if query_embedding is None:
                return

            cache_key = f"sem_{hash(query)}_{time.time()}"

            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.embeddings[oldest_key]

            self.cache[cache_key] = {
                'query': query,
                'response': response,
                'timestamp': time.time(),
            }
            self.embeddings[cache_key] = query_embedding
        except Exception as e:
            self._logger.error(f"Cache set failed: {e}")

    def clear(self) -> None:
        self.cache.clear()
        self.embeddings.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats['size'] = len(self.cache)
        return stats
