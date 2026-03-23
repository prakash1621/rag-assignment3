"""
Document loader — scans knowledge base and extracts text from files.
Supports: PDF, DOCX, HTML, TXT, MD
"""

import os
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import re

KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "knowledge-base")


def extract_text_from_file(file_path):
    """Extract text and hyperlinks from a document file."""
    text = ""
    links = []

    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                if page.get('/Annots'):
                    for annot in page['/Annots']:
                        obj = annot.get_object()
                        if obj.get('/A') and obj['/A'].get('/URI'):
                            links.append(obj['/A']['/URI'])

        elif file_path.endswith(('.docx', '.doc')):
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    links.append(rel.target_ref)

        elif file_path.endswith(('.html', '.htm')):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                if link['href'].startswith('http'):
                    links.append(link['href'])
            for script in soup(["script", "style"]):
                script.decompose()
            text += soup.get_text(separator='\n', strip=True) + "\n"

        elif file_path.endswith('.txt') or file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                text += content + "\n"
                urls = re.findall(r'https?://[^\s]+', content)
                links.extend(urls)

    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {str(e)}")

    return text, links


def scan_knowledge_base(kb_path=None):
    """Scan knowledge base directory and return files grouped by category."""
    kb_path = kb_path or KB_PATH
    categories = {}
    if not os.path.exists(kb_path):
        return categories

    for category_path in Path(kb_path).iterdir():
        if category_path.is_dir():
            category_name = category_path.name
            files = []
            for file_path in category_path.rglob('*'):
                if file_path.name.startswith('~'):
                    continue
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.html', '.htm', '.txt', '.md']:
                    files.append(str(file_path))
            if files:
                categories[category_name] = files

    return categories
