from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.rag.loader import load_text_files
from app.rag.chunker import chunk_document
from app.rag.models import DocumentChunk
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import FaissVectorStore, SearchResult
from app.rag.retrieval_cache import get_cached_results, set_cached_results


@dataclass
class RagConfig:
    index_dir: str = "data/index"
    chunk_size: int = 400
    chunk_overlap: int = 80


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        self.embedder = get_embedding_provider()

        # Try load index, else create empty (build later)
        loaded = FaissVectorStore.load(self.config.index_dir)
        self.store = loaded

    def build_from_files(self, paths: List[str]) -> None:
        """
        Build (or rebuild) FAISS index from scratch.
        Then save to disk.
        """
        all_chunks: List[DocumentChunk] = []

        for path in paths:
            text = load_text_files(path)
            doc_id = path.split("/")[-1].split("\\")[-1]

            chunks = chunk_document(
                doc_id=doc_id,
                text=text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            all_chunks.extend(chunks)

        vectors = self.embedder.embed([c.text for c in all_chunks])
        dim = int(vectors[0].shape[0]) if vectors else 0

        store = FaissVectorStore(embedding_dim=dim)
        store.add(vectors, all_chunks)

        # Save for next startup
        store.save(self.config.index_dir)

        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        # Retrieval cache first
        cached = get_cached_results(query, top_k)
        if cached is not None:
            return cached

        if self.store is None:
            raise RuntimeError("Vector store not built. Call build_from_files() first.")

        results = self.store.search(query, self.embedder, top_k=top_k)
        set_cached_results(query, top_k, results)
        return results