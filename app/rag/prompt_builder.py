from __future__ import annotations

from typing import List
from app.rag.vector_store import SearchResult

def build_rag_prompt(question: str, results: List[SearchResult]) -> str:
    """
    Build prompt using retrieved chunks
    """
    context_parts = []

    for r in results:
        context_parts.append(
            f"[CONTEXT]\n"
            f"doc_id={r.chunk.doc_id} chunk_id={r.chunk.chunk_id}\n"
            f"{r.chunk.text}\n"
        )
    context_text = "\n---\n".join(context_parts)
    #merges multiple chunks into one big string
    #below we build final prompt
    prompt = (
        "You are a helpful assistant.\n" # tone
        "Use ONLY the provided context to answer the question.\n" # prevent hallucination
        "If the context is insufficient, say you don't have enough information.\n\n" #safety
        f"{context_text}\n\n" #augmentation part
        f"[QUESTION]\n{question}\n\n"
        "[ANSWER]\n" # tell model to start generating answer here, improves output formatting
    )
    

    return prompt