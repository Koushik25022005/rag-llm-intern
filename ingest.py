import os, io, math
from typing import List, Tuple, Dict
import pdfplumber
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            texts.append(text)
    return '\n'.join(texts)

def extract_text_from_html(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    for s in soup(['script','style']):
        s.decompose()
    return soup.get_text(separator='\n')

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    return text

def chunk_text(text: str, chunk_size=500, overlap=50):
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks
