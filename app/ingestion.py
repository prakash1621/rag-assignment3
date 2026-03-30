import os
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import CharacterTextSplitter
from app.config import KB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_file(file_path):
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

def scan_knowledge_base():
    categories = {}
    if not os.path.exists(KB_PATH):
        return categories
    
    for category_path in Path(KB_PATH).iterdir():
        if category_path.is_dir():
            category_name = category_path.name
            files = []
            for file_path in category_path.rglob('*'):
                if file_path.name.startswith('~$'):
                    continue
                if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.html', '.htm', '.txt', '.md']:
                    files.append(str(file_path))
            if files:
                categories[category_name] = files
    
    return categories

def chunk_documents(categories):
    all_chunks = []
    all_metadatas = []
    all_links = []
    file_metadata = {}
    
    for category, files in categories.items():
        for file_path in files:
            text, links = extract_text_from_file(file_path)
            all_links.extend(links)
            
            if text.strip():
                splitter = CharacterTextSplitter(separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                chunks = splitter.split_text(text)
                
                char_count = 0
                for chunk in chunks:
                    if chunk.strip():
                        start_line = sum(1 for c in text[:char_count].split('\n'))
                        char_count += len(chunk)
                        end_line = sum(1 for c in text[:char_count].split('\n'))
                        
                        all_chunks.append(chunk[:8000])
                        all_metadatas.append({
                            "source": file_path,
                            "category": category,
                            "filename": os.path.basename(file_path),
                            "start_line": start_line,
                            "end_line": end_line
                        })
                
                file_metadata[file_path] = os.path.getmtime(file_path)
    
    return all_chunks, all_metadatas, list(set(all_links)), file_metadata
