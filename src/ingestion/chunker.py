"""
Document chunker — splits documents into chunks for embedding.

Strategies:
- Parent-Child: Large parents (3000 chars) + small children (500 chars)
- Semantic: Embedding-based boundary detection
- Basic: Fixed-size character splitting
"""

import hashlib
import re
import numpy as np
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


# ─── Basic Chunking (legacy) ────────────────────────────────────────────────

def chunk_documents(categories, extract_text_func, chunk_size=1500, chunk_overlap=300):
    """Basic fixed-size chunking for a set of categorized documents."""
    import os
    all_chunks, all_metadatas, all_links = [], [], []
    file_metadata = {}

    for category, files in categories.items():
        for file_path in files:
            text, links = extract_text_func(file_path)
            all_links.extend(links)
            if text.strip():
                splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        all_chunks.append(chunk[:8000])
                        all_metadatas.append({
                            "source": file_path,
                            "category": category,
                            "filename": os.path.basename(file_path),
                        })
                file_metadata[file_path] = os.path.getmtime(file_path)

    return all_chunks, all_metadatas, list(set(all_links)), file_metadata


# ─── Parent-Child Chunking ──────────────────────────────────────────────────

class ParentChildChunker:
    """Large parents for context, small children for precise retrieval."""

    def __init__(self, parent_size=3000, parent_overlap=500,
                 child_size=500, child_overlap=100):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str, metadata: Dict) -> Tuple[List[str], List[Dict]]:
        parent_chunks = self.parent_splitter.split_text(text)
        all_child_chunks, all_child_metadatas = [], []

        for parent_idx, parent_text in enumerate(parent_chunks):
            parent_id = hashlib.md5(
                f"{metadata.get('source', '')}_{parent_idx}_{parent_text[:100]}".encode()
            ).hexdigest()
            child_chunks = self.child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_chunks):
                all_child_chunks.append(child_text)
                child_meta = metadata.copy()
                child_meta.update({
                    'parent_id': parent_id,
                    'parent_text': parent_text,
                    'child_index': child_idx,
                    'parent_index': parent_idx,
                    'chunk_type': 'child',
                    'chunking_strategy': 'parent_child',
                })
                all_child_metadatas.append(child_meta)

        return all_child_chunks, all_child_metadatas


# ─── Semantic Chunking ──────────────────────────────────────────────────────

class SemanticChunker:
    """Splits text at natural semantic boundaries using embedding similarity."""

    def __init__(self, embedder, buffer_size=1,
                 breakpoint_threshold_type="percentile",
                 breakpoint_threshold_amount=95):
        self.embedder = embedder
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def chunk(self, text: str, metadata: Dict) -> Tuple[List[str], List[Dict]]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 1:
            return [text], [metadata]

        # Create buffered sentence groups
        groups = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            groups.append(" ".join(sentences[start:end]))

        # Embed groups
        embeddings = [self.embedder(g) for g in groups]

        # Calculate cosine distances between consecutive embeddings
        distances = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            distances.append(1 - sim)

        if not distances:
            return [text], [metadata]

        # Find breakpoints
        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = np.mean(distances) + self.breakpoint_threshold_amount * np.std(distances)
        else:
            q1, q3 = np.percentile(distances, [25, 75])
            threshold = q3 + self.breakpoint_threshold_amount * (q3 - q1)

        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        # Build chunks from breakpoints
        chunks, chunk_metadatas = [], []
        start_idx = 0
        for bp in breakpoints:
            chunk_text = " ".join(sentences[start_idx:bp + 1])
            if chunk_text.strip():
                meta = metadata.copy()
                meta.update({'chunk_type': 'semantic', 'chunking_strategy': 'semantic'})
                chunks.append(chunk_text)
                chunk_metadatas.append(meta)
            start_idx = bp + 1

        # Last chunk
        if start_idx < len(sentences):
            chunk_text = " ".join(sentences[start_idx:])
            if chunk_text.strip():
                meta = metadata.copy()
                meta.update({'chunk_type': 'semantic', 'chunking_strategy': 'semantic'})
                chunks.append(chunk_text)
                chunk_metadatas.append(meta)

        return chunks if chunks else [text], chunk_metadatas if chunk_metadatas else [metadata]
