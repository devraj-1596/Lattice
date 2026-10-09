"""
llm.py
------
Gemini API integration for Lattice.

- Loads GEMINI_API_KEY from .env (never hardcoded)
- Builds a context-grounded prompt from retrieved chunks
- Calls the Gemini API using the google-genai SDK
- Returns the generated answer as a string
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ──────────────────────────────────────────────
# GEMINI SDK IMPORT
# ──────────────────────────────────────────────
# We use the google-genai package (google.genai)
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.5-flash"   # fast, capable, widely available

SYSTEM_INSTRUCTION = """You are Lattice, a document-grounded knowledge assistant designed for students.

Your ONLY job is to answer the user's question using the document context provided below.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "The answer to this question is not available in the uploaded documents."
3. Do not fabricate facts, citations, or sources.
4. When referencing information, mention the source document and page number when available.
5. Be clear, concise, and helpful — you are assisting first-year college students.
6. If the question is a follow-up, use the conversation history to understand the context,
   but still answer only from the provided document context."""


def get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from the environment.
    Returns None (not the key value) if missing — never raises or exposes the key.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key if key and key != "your_gemini_api_key_here" else None


def build_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a readable context block for the prompt.

    Each chunk is prefixed with its source, page, and chunk ID so the model
    can reference them in its answer.
    """
    if not retrieved_chunks:
        return "No relevant document context was found."

    lines = ["=== DOCUMENT CONTEXT ===\n"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        chunk_id = chunk.get("chunk_id", i - 1)
        text = chunk.get("text", "")
        lines.append(
            f"[Source {i}: {source} | Page {page} | Chunk #{chunk_id}]\n{text}\n"
        )
    lines.append("=== END OF CONTEXT ===")
    return "\n".join(lines)


def build_history_block(chat_history: List[Dict[str, str]]) -> str:
    """
    Format previous Q&A turns for inclusion in the prompt.
    chat_history: [{"role": "user"|"assistant", "content": str}, ...]
    """
    if not chat_history:
        return ""
    lines = ["=== CONVERSATION HISTORY ==="]
    for turn in chat_history:
        role = "Student" if turn["role"] == "user" else "Lattice"
        lines.append(f"{role}: {turn['content']}")
    lines.append("=== END HISTORY ===")
    return "\n".join(lines)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Generate a grounded answer using Gemini.

    Parameters
    ----------
    question        : The user's current question.
    retrieved_chunks: Top-K chunks from FAISS retrieval.
    chat_history    : Prior conversation turns (for follow-up awareness).

    Returns
    -------
    str : The generated answer, or a friendly error message.
    """
    if not _GENAI_AVAILABLE:
        return (
            "⚠️ The google-genai package is not installed. "
            "Please run: pip install google-genai"
        )

    api_key = get_api_key()
    if not api_key:
        return (
            "⚠️ Gemini API key is missing. Please create a `.env` file in the project "
            "folder with the following line:\n\n`GEMINI_API_KEY=your_actual_key_here`\n\n"
            "You can obtain a free key at https://aistudio.google.com/app/apikey"
        )

    # ── Build the full prompt ──
    context_block = build_context_block(retrieved_chunks)
    history_block = build_history_block(chat_history or [])

    prompt_parts = []
    if history_block:
        prompt_parts.append(history_block)
    prompt_parts.append(context_block)
    prompt_parts.append(f"\nStudent's Question: {question}")
    prompt_parts.append(
        "\nPlease answer the question based strictly on the document context above."
    )
    full_prompt = "\n\n".join(prompt_parts)

    # ── Call Gemini API ──
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,   # low temperature for factual grounding
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()

    except Exception as e:
        error_str = str(e)
        # Provide friendly messages for common errors
        if "API_KEY_INVALID" in error_str or "api_key" in error_str.lower():
            return (
                "⚠️ Your Gemini API key appears to be invalid. "
                "Please check the GEMINI_API_KEY value in your .env file."
            )
        if "quota" in error_str.lower() or "429" in error_str:
            return (
                "⚠️ Gemini API quota exceeded. Please wait a moment and try again, "
                "or check your usage at https://aistudio.google.com/"
            )
        if "model" in error_str.lower() and "not found" in error_str.lower():
            return (
                f"⚠️ The Gemini model '{GEMINI_MODEL}' was not found. "
                "Please check your API access and model availability."
            )
        return f"⚠️ Gemini API error: {error_str}"
