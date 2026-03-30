"""
Document loader — scans knowledge base and extracts text from files.

Supports: PDF, DOCX, HTML, TXT, MD
"""

from app.ingestion import scan_knowledge_base, extract_text_from_file

__all__ = ["scan_knowledge_base", "extract_text_from_file"]
