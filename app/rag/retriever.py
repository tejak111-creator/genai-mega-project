from __future__ import annotations

from typing import List

from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import FaissVectorStore, SearchResult

class Retriever:
    """
    Handles query embedding + vector search.
    Keeps pipeline independent from FAISS Details.
    """

    def __init__(self, store: FaissVectorStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        results = self.store.search(query=query, embedder=self.embedder, top_k=top_k)
        return results