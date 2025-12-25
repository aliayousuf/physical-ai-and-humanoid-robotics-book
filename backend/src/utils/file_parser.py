from pypdf import PdfReader
import markdown
from typing import List, Dict, Any
import os
from pathlib import Path

def parse_pdf(file_path: str) -> str:
    """
    Parse PDF file and extract text content
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error parsing PDF file {file_path}: {str(e)}")

    return text

def parse_markdown(file_path: str) -> str:
    """
    Parse Markdown file and extract text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Convert markdown to text by removing markdown formatting
            # For now, we'll just return the raw content, but we could use markdown library
            # to convert to HTML and then extract plain text if needed
            html = markdown.markdown(content)
            # Simple text extraction from HTML (in a real implementation, use a proper HTML parser)
            # For now, just return the raw content
            return content
    except Exception as e:
        raise Exception(f"Error parsing Markdown file {file_path}: {str(e)}")

def parse_text(file_path: str) -> str:
    """
    Parse text file and extract content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except Exception as e:
        raise Exception(f"Error parsing text file {file_path}: {str(e)}")

def get_file_type(file_path: str) -> str:
    """
    Determine the file type based on extension
    """
    extension = Path(file_path).suffix.lower()
    if extension == '.pdf':
        return 'pdf'
    elif extension in ['.md', '.markdown']:
        return 'markdown'
    elif extension == '.txt':
        return 'text'
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def parse_file(file_path: str) -> str:
    """
    Parse a file based on its type and return text content
    """
    file_type = get_file_type(file_path)

    if file_type == 'pdf':
        return parse_pdf(file_path)
    elif file_type == 'markdown':
        return parse_markdown(file_path)
    elif file_type == 'text':
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def scan_docs_folder(docs_path: str, patterns: List[str] = None) -> List[str]:
    """
    Scan the docs folder for supported file types
    """
    if patterns is None:
        patterns = ["*.md", "*.pdf", "*.txt"]

    supported_extensions = {'.md', '.markdown', '.pdf', '.txt'}
    files = []

    for root, dirs, filenames in os.walk(docs_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            extension = Path(filename).suffix.lower()

            if extension in supported_extensions:
                files.append(file_path)

    return files