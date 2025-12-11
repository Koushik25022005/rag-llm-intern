# Local LLM-RAG Streamlit

This repository provides a **local** Retrieval-Augmented Generation (RAG) demo that:
- ingests **PDF**, **HTML**, and **image (PNG/JPEG)** files,
- extracts text (PDF/HTML via pdfplumber/BeautifulSoup; images via pytesseract),
- embeds text using **sentence-transformers** (all-MiniLM-L6-v2),
- stores vectors in a **FAISS** index,
- serves a Streamlit UI for ingestion and querying,
- uses a local HuggingFace text2text model (configurable) for answer generation.

## Quick start (local)

1. Create a Python 3.10+ venv and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\\.venv\\Scripts\\activate  # Windows (PowerShell)
