from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss is not installed. Run: pip install faiss-cpu") from e

from app.rag.models import DocumentChunk
from app.rag.embeddings import EmbeddingProvider
from app.rag.persistence import save_faiss_index, load_faiss_index


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float  # lower is better for L2


class FaissVectorStore:
    """
    Minimal FAISS vector store:
    - FAISS index holds vectors
    - Python list holds chunk metadata (doc_id, text, etc.)
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[DocumentChunk] = []

    def add(self, vectors: List[np.ndarray], chunks: List[DocumentChunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have same length")
        if not vectors:
            return

        mat = np.vstack([v.reshape(1, -1) for v in vectors]).astype(np.float32)

        if mat.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {mat.shape[1]}")

        self.index.add(mat)
        self.chunks.extend(chunks)

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []

        q = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q, top_k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[int(idx)], score=float(dist)))
        return results

    def search(self, query: str, embedder: EmbeddingProvider, top_k: int = 5) -> List[SearchResult]:
        q_vec = embedder.embed([query])[0]
        return self.search_by_vector(q_vec, top_k=top_k)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, dir_path: str) -> None:
        save_faiss_index(
            index=self.index,
            chunks=self.chunks,
            embedding_dim=self.embedding_dim,
            dir_path=dir_path,
        )

    @classmethod
    def load(cls, dir_path: str) -> Optional["FaissVectorStore"]:
        loaded = load_faiss_index(dir_path)
        if loaded is None:
            return None

        store = cls(embedding_dim=loaded.embedding_dim)
        store.index = loaded.index
        store.chunks = loaded.chunks
        return store