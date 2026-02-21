from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss is not installed. Run: pip install faiss-cpu") from e

from app.rag.models import DocumentChunk
from app.rag.embeddings import EmbeddingProvider

@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float #lower is better for L2 distance

class FaissVectorStore:
    """ 
    Minimal FAISS vectore store: index stores vectors, python list stores DocChunk metadata
    """
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[DocumentChunk] = []
    
    def add(self, vectors: List[np.ndarray], chunks: List[DocumentChunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Must have same length")
        if len(vectors) == 0:
            return
        #Ensure correct shape: (N,D) float32
        mat = np.vstack([v.reshape(1,-1) for v in vectors]).astype(np.float32)

        if mat.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {mat.shape[1]}")
        
        self.index.add(mat)
        self.chunks.extend(chunks)

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        q = query_vector.reshape(1,-1).astype(np.float32)
        distances, indices = self.index.search(q, top_k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[int(idx)], score=float(dist)))
        return results
    def search(self, query: str, embedder: EmbeddingProvider, top_k: int = 5) -> List[SearchResult]:
        #embed the query using the same embedding model as documents
        q_vec = embedder.embed([query])[0]
        return self.search_by_vector(q_vec, top_k=top_k)