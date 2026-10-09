"""
test_lattice.py
---------------
Component-level tests for Lattice.

Run:  py -3 test_lattice.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
SEP  = "-" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ──────────────────────────────────────────────────────────────
# TEST 1 — Package imports
# ──────────────────────────────────────────────────────────────
section("TEST 1: Package Imports")

packages = {
    "pypdf":                lambda: __import__("pypdf"),
    "faiss-cpu":            lambda: __import__("faiss"),
    "sentence-transformers":lambda: __import__("sentence_transformers"),
    "numpy":                lambda: __import__("numpy"),
    "streamlit":            lambda: __import__("streamlit"),
    "python-dotenv":        lambda: __import__("dotenv"),
    "google-genai":         lambda: __import__("google.genai", fromlist=["genai"]),
}

all_imports_ok = True
for name, importer in packages.items():
    try:
        importer()
        print(f"{PASS} {name}")
    except ImportError as e:
        print(f"{FAIL} {name}: {e}")
        all_imports_ok = False

if not all_imports_ok:
    print("\n[WARNING] Some packages missing. Install: py -3 -m pip install -r requirements.txt")


# ──────────────────────────────────────────────────────────────
# TEST 2 — document_processor: TXT chunking
# ──────────────────────────────────────────────────────────────
section("TEST 2: Document Processor - TXT")

try:
    from document_processor import chunk_text, clean_text, process_document

    sample_text = (
        "A Hilbert space is a real or complex inner product space that is also a "
        "complete metric space with respect to the distance function induced by the "
        "inner product. Hilbert spaces allow the methods of linear algebra and calculus "
        "to be generalised from finite-dimensional Euclidean spaces to spaces of functions. "
        "Hilbert spaces arise naturally and frequently in mathematics and physics. "
    ) * 5

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample_text)
        tmp_txt = f.name

    chunks = process_document(tmp_txt)
    os.unlink(tmp_txt)

    assert len(chunks) > 0, "No chunks produced"
    for c in chunks:
        assert "text" in c and "source" in c and "page" in c and "chunk_id" in c, \
            f"Missing metadata in chunk: {c.keys()}"
    print(f"{PASS} TXT processed: {len(chunks)} chunk(s) produced")
    print(f"{INFO} Sample: page={chunks[0]['page']}, id={chunks[0]['chunk_id']}, "
          f"text_len={len(chunks[0]['text'])}")

except Exception as e:
    print(f"{FAIL} TXT processing failed: {e}")
    import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# TEST 3 — document_processor: PDF
# ──────────────────────────────────────────────────────────────
section("TEST 3: Document Processor - PDF (reportlab)")

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from document_processor import process_document
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, "Page 1: Introduction to Hilbert Spaces")
    c.drawString(72, 700, "A Hilbert space is a complete inner product space.")
    c.showPage()
    c.drawString(72, 720, "Page 2: Hermitian Operators")
    c.drawString(72, 700, "A Hermitian operator satisfies A = A dagger.")
    c.save()
    buf.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(buf.read())
        tmp_pdf = f.name

    chunks = process_document(tmp_pdf)
    os.unlink(tmp_pdf)

    assert len(chunks) > 0, "No chunks from PDF"
    pages_seen = {c["page"] for c in chunks}
    print(f"{PASS} PDF processed: {len(chunks)} chunk(s), pages seen: {sorted(pages_seen)}")
    print(f"{INFO} Source field: '{chunks[0]['source']}'")

except ImportError:
    print(f"{INFO} reportlab not available - skipping synthetic PDF test")
except Exception as e:
    print(f"{FAIL} PDF processing failed: {e}")
    import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# TEST 4 — Embedding model
# ──────────────────────────────────────────────────────────────
section("TEST 4: Sentence Transformer Embeddings")

try:
    from rag import embed_texts, embed_query

    test_sentences = [
        "A Hilbert space is a complete inner product space.",
        "Hermitian operators are self-adjoint.",
        "Pauli matrices are fundamental in quantum mechanics.",
    ]

    print(f"{INFO} Loading model (may download on first run)...")
    embeddings = embed_texts(test_sentences)

    assert embeddings.shape[0] == 3, f"Expected 3 embeddings, got {embeddings.shape[0]}"
    assert embeddings.dtype.name == "float32", f"Wrong dtype: {embeddings.dtype}"
    print(f"{PASS} Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    q_emb = embed_query("What is a Hilbert space?")
    assert q_emb.shape == (1, embeddings.shape[1]), f"Query shape wrong: {q_emb.shape}"
    print(f"{PASS} Query embedding shape: {q_emb.shape}")

except Exception as e:
    print(f"{FAIL} Embedding test failed: {e}")
    import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# TEST 5 — FAISS indexing + retrieval
# ──────────────────────────────────────────────────────────────
section("TEST 5: FAISS Indexing & Retrieval")

try:
    from rag import build_index, retrieve

    test_chunks = [
        {"text": "A Hilbert space is a complete inner product space with a norm.",
         "source": "quantum.pdf", "page": 1, "chunk_id": 0},
        {"text": "Hermitian operators have real eigenvalues and orthogonal eigenvectors.",
         "source": "quantum.pdf", "page": 2, "chunk_id": 1},
        {"text": "Pauli matrices sigma_x, sigma_y, sigma_z are 2x2 Hermitian matrices.",
         "source": "quantum.pdf", "page": 3, "chunk_id": 2},
        {"text": "The inner product in a Hilbert space defines geometry and angles.",
         "source": "quantum.pdf", "page": 1, "chunk_id": 3},
        {"text": "Quantum states live in a Hilbert space called the state space.",
         "source": "quantum.pdf", "page": 4, "chunk_id": 4},
        {"text": "Unitary operators preserve the inner product and norm.",
         "source": "quantum.pdf", "page": 5, "chunk_id": 5},
    ]

    index, metadata = build_index(test_chunks)
    print(f"{PASS} FAISS index built: {index.ntotal} vectors, dim={index.d}")

    # Normal retrieval
    results = retrieve("What is a Hilbert space?", index, metadata, top_k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("score" in r for r in results), "Missing score in results"
    assert all(r["score"] <= 1.05 for r in results), "Scores out of range"

    print(f"{PASS} Retrieved {len(results)} chunks for 'What is a Hilbert space?'")
    for r in results:
        print(f"   [{r['score']:.3f}] Page {r['page']}: {r['text'][:60]}...")

    # Edge case: top_k > index size
    results_small = retrieve("Hilbert", index, metadata, top_k=20)
    assert len(results_small) == len(test_chunks), \
        f"Expected {len(test_chunks)}, got {len(results_small)}"
    print(f"{PASS} Edge case (top_k > size): returned {len(results_small)} (all available)")

except Exception as e:
    print(f"{FAIL} FAISS test failed: {e}")
    import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# TEST 6 — LLM module config
# ──────────────────────────────────────────────────────────────
section("TEST 6: LLM Module Configuration")

try:
    from llm import get_api_key, build_context_block, build_history_block

    key = get_api_key()
    if key:
        print(f"{PASS} GEMINI_API_KEY found (length {len(key)})")
    else:
        print(f"{INFO} GEMINI_API_KEY not set - update .env before using Gemini")

    dummy_chunks = [
        {"text": "Hilbert spaces are complete.", "source": "a.pdf", "page": 1, "chunk_id": 0},
        {"text": "Hermitian operators are key.", "source": "a.pdf", "page": 2, "chunk_id": 1},
    ]
    ctx = build_context_block(dummy_chunks)
    assert "DOCUMENT CONTEXT" in ctx
    assert "a.pdf" in ctx
    print(f"{PASS} Context block OK ({len(ctx)} chars)")

    history = [
        {"role": "user", "content": "What is a Hilbert space?"},
        {"role": "assistant", "content": "It is a complete inner product space."},
    ]
    hist = build_history_block(history)
    assert "CONVERSATION HISTORY" in hist
    print(f"{PASS} History block OK ({len(hist)} chars)")

except Exception as e:
    print(f"{FAIL} LLM config failed: {e}")
    import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  All tests complete.")
print("  To start Lattice: py -3 -m streamlit run app.py")
print(SEP)
