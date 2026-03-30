"""
Abstract Base Classes for Chunking Strategies

Defines interfaces and contracts for all chunking implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import logging


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict) -> Tuple[List[str], List[Dict]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata (source, category, etc.)
            
        Returns:
            Tuple of (chunk_texts, chunk_metadatas)
            
        Raises:
            ChunkingError: If chunking operation fails
        """
        pass
    
    def validate_input(self, text: str, metadata: Dict) -> None:
        """
        Validate input parameters.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Raises:
            ChunkingError: If validation fails
        """
        if not text or not isinstance(text, str):
            raise ChunkingError("Text must be a non-empty string")
        
        if not isinstance(metadata, dict):
            raise ChunkingError("Metadata must be a dictionary")
        
        if len(text.strip()) == 0:
            raise ChunkingError("Text cannot be empty or whitespace only")
    
    def get_info(self) -> Dict:
        """
        Get information about the chunking strategy.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'strategy': self.__class__.__name__,
            'description': self.__doc__ or "No description available"
        }


class ChunkingError(Exception):
    """Base exception for chunking-related errors"""
    pass


class ChunkingConfigurationError(ChunkingError):
    """Exception raised when chunking configuration is invalid"""
    pass


class ChunkingOperationError(ChunkingError):
    """Exception raised when chunking operation fails"""
    pass

