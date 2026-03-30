"""
Abstract Base Class for Semantic Cache

Defines interface for the semantic cache bonus feature.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import logging


class BaseSemanticCache(ABC):
    """Abstract base class for semantic similarity cache"""

    def __init__(self, embedder, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600, max_cache_size: int = 1000):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get(self, query: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set(self, query: str, response: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        try:
            embedding = self.embedder(query)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            return embedding
        except Exception as e:
            self._logger.error(f"Embedding failed for query '{query[:50]}...': {e}")
            return None

    def _record_hit(self):
        self.hits += 1

    def _record_miss(self):
        self.misses += 1

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'similarity_threshold': self.similarity_threshold,
        }


class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass


class CacheOperationError(CacheError):
    """Exception raised when cache operation fails"""
    pass
