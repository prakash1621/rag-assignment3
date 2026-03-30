"""
Semantic Chunking Strategy

Splits documents based on embedding similarity between sentences.
Creates chunks at natural semantic boundaries.
"""

import numpy as np
from typing import List, Dict, Tuple
from .base_chunker import BaseChunker, ChunkingOperationError


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on embedding similarity.
    
    Splits text at points where semantic similarity drops below threshold.
    """
    
    def __init__(self, embedder, buffer_size: int = 1,
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 95):
        """
        Args:
            embedder: Function that takes text and returns embedding vector
            buffer_size: Number of sentences to combine for embedding
            breakpoint_threshold_type: 'percentile', 'standard_deviation', or 'interquartile'
            breakpoint_threshold_amount: Threshold value for breakpoint detection
        """
        super().__init__()
        self.embedder = embedder
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_cosine_distances(self, embeddings: List[np.ndarray]) -> List[float]:
        """Calculate cosine distances between consecutive embeddings"""
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            distance = 1 - similarity
            distances.append(distance)
        return distances
    
    def _find_breakpoints(self, distances: List[float]) -> List[int]:
        """Find breakpoint indices based on threshold"""
        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = np.mean(distances) + self.breakpoint_threshold_amount * np.std(distances)
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            threshold = q3 + self.breakpoint_threshold_amount * iqr
        else:
            threshold = np.percentile(distances, 95)
        
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]
        return breakpoints
    
    def chunk(self, text: str, metadata: Dict = None) -> Tuple[List[str], List[Dict]]:
        """
        Chunk text using semantic similarity.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            (chunks, metadatas)
        """
        try:
            sentences = self._split_sentences(text)
            
            if len(sentences) <= self.buffer_size:
                return [text], [metadata or {}]
            
            # Create sentence groups
            sentence_groups = []
            for i in range(0, len(sentences), self.buffer_size):
                group = " ".join(sentences[i:i + self.buffer_size])
                sentence_groups.append(group)
            
            # Get embeddings
            embeddings = [self.embedder(group) for group in sentence_groups]
            
            # Calculate distances
            distances = self._calculate_cosine_distances(embeddings)
            
            # Find breakpoints
            breakpoints = self._find_breakpoints(distances)
            
            # Create chunks
            chunks = []
            metadatas = []
            start_idx = 0
            
            for breakpoint in breakpoints:
                end_idx = (breakpoint + 1) * self.buffer_size
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = " ".join(chunk_sentences)
                
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    chunk_metadata = (metadata or {}).copy()
                    chunk_metadata['chunk_type'] = 'semantic'
                    chunk_metadata['chunk_index'] = len(chunks) - 1
                    metadatas.append(chunk_metadata)
                
                start_idx = end_idx
            
            # Add remaining sentences
            if start_idx < len(sentences):
                chunk_text = " ".join(sentences[start_idx:])
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    chunk_metadata = (metadata or {}).copy()
                    chunk_metadata['chunk_type'] = 'semantic'
                    chunk_metadata['chunk_index'] = len(chunks) - 1
                    metadatas.append(chunk_metadata)
            
            self._logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks, metadatas
            
        except Exception as e:
            self._logger.error(f"Semantic chunking failed: {e}")
            raise ChunkingOperationError(f"Failed to chunk document: {e}")
