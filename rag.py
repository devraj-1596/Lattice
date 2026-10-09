"""
rag.py
------
Retrieval-Augmented Generation core for Lattice.

Responsibilities:
  - Build a FAISS vector index from document chunks
  - Persist the index + metadata to disk (vectorstore/)
  - Load the index back from disk on startup
  - Embed user queries with the same Sentence Transformer model
  - Return the top-K most relevant chunks with similarity scores
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# Primary: local copy (avoids Windows symlink issues with HF Hub cache)
# Fallback: HF model name (downloads from Hub if local copy not found)
_LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"
MODEL_NAME = (
    str(_LOCAL_MODEL_DIR)
    if _LOCAL_MODEL_DIR.exists() and any(_LOCAL_MODEL_DIR.iterdir())
    else "sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"
TOP_K = 5  # number of chunks to retrieve


# ──────────────────────────────────────────────
# MODEL LOADING  (cached so it loads once)
# ──────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load (or return cached) Sentence Transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns shape (1, embedding_dim) float32 array ready for FAISS.
    """
    return embed_texts([query])


# ──────────────────────────────────────────────
# INDEX BUILDING
# ──────────────────────────────────────────────

def build_index(chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build a FAISS IndexFlatIP (inner-product / cosine-like) index
    from a list of chunk dicts.

    chunks: list of {"text", "source", "page", "chunk_id"}

    Returns:
        index    – the FAISS index
        metadata – same list of chunk dicts (stored separately from vectors)
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks


def save_index(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    """Persist FAISS index and metadata to the vectorstore/ directory."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_index() -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, Any]]]]:
    """
    Load FAISS index and metadata from disk.

    Returns (None, None) if no index has been built yet.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    try:
        index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load vector index from disk: {e}")


def clear_index() -> None:
    """Delete the persisted FAISS index and metadata files."""
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()


# ──────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns a list of dicts, each containing all original metadata
    plus a "score" field (cosine similarity, 0–1 scale).

    Handles the edge case where fewer than top_k chunks exist.
    """
    if index is None or not metadata:
        return []

    # Guard: cannot ask for more results than we have
    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    # Embed and normalise the query
    query_vec = embed_query(query)
    faiss.normalize_L2(query_vec)

    # Search
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue  # FAISS occasionally returns -1 for padding
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    return results


# ──────────────────────────────────────────────
# CONVENIENCE: build → save → return
# ──────────────────────────────────────────────

def index_chunks(chunks: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Build a FAISS index from chunks, persist it, and return (index, metadata).
    This is the single entry point called from app.py after document processing.
    """
    index, metadata = build_index(chunks)
    save_index(index, metadata)
    return index, metadata
