"""
Parent-Child Chunking Strategy

Splits documents into large parent chunks (for context) and smaller child chunks (for precision).
Retrieval happens at child level, but parent context is passed to LLM.
"""

from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib


class ParentChildChunker:
    def __init__(self, parent_size: int = 3000, parent_overlap: int = 500,
                 child_size: int = 500, child_overlap: int = 100):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, text: str, metadata: Dict) -> Tuple[List[str], List[Dict]]:
        """
        Create parent-child chunks from text.
        
        Returns:
            - child_chunks: List of child chunk texts (for embedding/retrieval)
            - child_metadatas: List of metadata dicts with parent_id linking to parent
        """
        parent_chunks = self.parent_splitter.split_text(text)
        
        all_child_chunks = []
        all_child_metadatas = []
        
        for parent_idx, parent_text in enumerate(parent_chunks):
            # Generate unique parent ID
            parent_id = hashlib.md5(
                f"{metadata.get('source', '')}_{parent_idx}_{parent_text[:100]}".encode()
            ).hexdigest()
            
            # Split parent into children
            child_chunks = self.child_splitter.split_text(parent_text)
            
            for child_idx, child_text in enumerate(child_chunks):
                all_child_chunks.append(child_text)
                
                # Store parent context in metadata
                child_metadata = metadata.copy()
                child_metadata.update({
                    'parent_id': parent_id,
                    'parent_text': parent_text,  # Full parent context
                    'child_index': child_idx,
                    'parent_index': parent_idx,
                    'chunk_type': 'child',
                    'chunking_strategy': 'parent_child'
                })
                all_child_metadatas.append(child_metadata)
        
        return all_child_chunks, all_child_metadatas
