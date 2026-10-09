"""
document_processor.py
---------------------
Handles all document ingestion for Lattice:
  - PDF files: extracts text page-by-page using PyPDF, preserving page numbers
  - TXT files: reads raw text, treats everything as page 1
  - Splits extracted text into overlapping chunks
  - Attaches metadata to every chunk: source, page, chunk_id, text
"""

import re
from pathlib import Path
from typing import List, Dict, Any

import pypdf


# ──────────────────────────────────────────────
# CHUNKING CONFIGURATION
# ──────────────────────────────────────────────
CHUNK_SIZE = 500        # target characters per chunk
CHUNK_OVERLAP = 100     # characters of overlap between consecutive chunks


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF.

    Returns a list of dicts: [{"page": int, "text": str}, ...]
    Pages are 1-indexed to match what a student sees in a PDF viewer.
    """
    pages = []
    try:
        reader = pypdf.PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": page_num, "text": text})
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{file_path}': {e}")

    if not pages:
        raise ValueError(
            f"No readable text found in '{Path(file_path).name}'. "
            "The PDF may be scanned or image-only and requires OCR."
        )
    return pages


def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a plain-text file.

    Returns a single-page representation so all downstream code
    works identically for both PDF and TXT.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read TXT file '{file_path}': {e}")

    if not text:
        raise ValueError(f"The file '{Path(file_path).name}' appears to be empty.")

    return [{"page": 1, "text": text}]


def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    # Collapse runs of whitespace (but keep newlines for readability)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split *text* into overlapping character-level chunks.

    Strategy: slide a window of `chunk_size` characters across the text,
    stepping forward by (chunk_size - overlap) each time.
    Each chunk is stripped of leading/trailing whitespace.
    Empty chunks are discarded.
    """
    if not text:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def process_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract → clean → chunk for a single file.

    Returns a flat list of chunk dicts:
      {
        "text":     str,   # the chunk text
        "source":   str,   # original filename
        "page":     int,   # page number (1-indexed; 1 for TXT)
        "chunk_id": int,   # sequential ID across the whole document
      }
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    filename = path.name

    # ── 1. Extract raw text by page ──
    if suffix == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif suffix == ".txt":
        pages = extract_text_from_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Only .pdf and .txt are supported."
        )

    # ── 2. Clean + chunk each page, accumulate ──
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        cleaned = clean_text(page_data["text"])
        page_chunks = chunk_text(cleaned)

        for chunk_text_str in page_chunks:
            all_chunks.append(
                {
                    "text": chunk_text_str,
                    "source": filename,
                    "page": page_num,
                    "chunk_id": global_chunk_id,
                }
            )
            global_chunk_id += 1

    if not all_chunks:
        raise ValueError(
            f"No text chunks could be extracted from '{filename}'. "
            "Please check the file content."
        )

    return all_chunks


def process_multiple_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process a list of file paths and merge all chunks into one flat list.

    chunk_id values are reassigned globally so they are unique across all docs.
    Returns (merged_chunks, errors_dict) where errors_dict maps filename → error message.
    """
    all_chunks = []
    errors = {}

    for fp in file_paths:
        try:
            chunks = process_document(fp)
            all_chunks.extend(chunks)
        except Exception as e:
            errors[Path(fp).name] = str(e)

    # Reassign global chunk_id across all documents
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    return all_chunks, errors
