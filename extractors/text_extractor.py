"""
extractors/text_extractor.py
Extrae texto de diferentes formatos de archivo.
"""

import io
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    """
    Extrae texto del archivo según su extensión.
    
    Args:
        filename: Nombre del archivo
        content: Contenido del archivo en bytes
    
    Returns:
        Texto extraído
    """
    ext = Path(filename).suffix.lower()
    
    extractors = {
        '.txt': _extract_txt,
        '.md': _extract_txt,
        '.pdf': _extract_pdf,
        '.docx': _extract_docx,
    }
    
    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(f"Formato no soportado: {ext}")
    
    return extractor(content)


def extract_text_from_url(url: str) -> str:
    """
    Extrae texto de una URL.
    """
    import httpx
    from bs4 import BeautifulSoup
    
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remover scripts y styles
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    # Obtener texto
    text = soup.get_text(separator='\n', strip=True)
    
    # Limpiar líneas vacías múltiples
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def _extract_txt(content: bytes) -> str:
    """Extrae texto de archivos TXT y MD."""
    return content.decode('utf-8')


def _extract_pdf(content: bytes) -> str:
    """Extrae texto de archivos PDF."""
    from PyPDF2 import PdfReader
    
    reader = PdfReader(io.BytesIO(content))
    text_parts = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    
    return '\n'.join(text_parts)


def _extract_docx(content: bytes) -> str:
    """Extrae texto de archivos DOCX."""
    from docx import Document
    
    doc = Document(io.BytesIO(content))
    text_parts = []
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)
    
    return '\n'.join(text_parts)